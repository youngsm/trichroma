"""Vectorized builder for threaded (escape-link) bounding-volume hierarchies.

The layout is designed for stackless traversal in Triton: nodes are stored in
depth-first preorder, the first child of an inner node is the next node, and
every node stores the index of the node that follows its subtree
(``escape``, or -1 at the end). A ray that misses a node, or finishes a leaf,
jumps to ``escape``; a ray that hits an inner node steps to ``index + 1``.

Construction is a top-down linear BVH over 63-bit Morton codes of primitive
centroids, split level by level at the highest differing code bit (median
split for identical codes). Bounds are rounded outward to float32.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThreadedBVH:
    """Preorder threaded BVH.

    ``lower``/``upper``: float32 [M,3] conservative node bounds.
    ``escape``: int32 [M], next node after the subtree, -1 at the end.
    ``leaf_first``: int32 [M], first slot in ``order`` for leaves, -1 inner.
    ``leaf_count``: int32 [M], primitives in a leaf, 0 for inner nodes.
    ``order``: int32 [N], primitive index of every leaf slot.
    """

    lower: np.ndarray
    upper: np.ndarray
    escape: np.ndarray
    leaf_first: np.ndarray
    leaf_count: np.ndarray
    order: np.ndarray

    @property
    def node_count(self):
        return len(self.escape)


def _spread_bits_21(value):
    """Insert two zero bits between each of the low 21 bits (uint64)."""
    x = value.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_codes(centroids, lower, upper):
    """63-bit Morton codes of ``centroids`` inside the box [lower, upper]."""
    extent = np.maximum(upper - lower, 1e-30)
    scaled = np.clip((centroids - lower) / extent, 0.0, 1.0) * float((1 << 21) - 1)
    q = scaled.astype(np.uint64)
    return (_spread_bits_21(q[:, 0]) << np.uint64(2)) | (_spread_bits_21(q[:, 1]) << np.uint64(1)) | _spread_bits_21(q[:, 2])


def _round_down_f32(value):
    v32 = value.astype(np.float32)
    return np.where(v32.astype(np.float64) > value, np.nextafter(v32, np.float32(-np.inf)), v32).astype(np.float32)


def _round_up_f32(value):
    v32 = value.astype(np.float32)
    return np.where(v32.astype(np.float64) < value, np.nextafter(v32, np.float32(np.inf)), v32).astype(np.float32)


def _highest_bit(x):
    """Index of the highest set bit of nonzero uint64 values."""
    x = x.astype(np.uint64)
    result = np.zeros(x.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = (x >> np.uint64(shift)) != 0
        result += np.where(mask, shift, 0)
        x = np.where(mask, x >> np.uint64(shift), x)
    return result


def build_threaded_bvh(box_lower, box_upper, *, leaf_size=4):
    """Build a threaded BVH over primitives with float64 bounds [N,3]."""
    box_lower = np.asarray(box_lower, dtype=np.float64)
    box_upper = np.asarray(box_upper, dtype=np.float64)
    count = len(box_lower)
    if count == 0:
        raise ValueError("cannot build a BVH over zero primitives")
    if leaf_size < 1 or leaf_size > 15:
        raise ValueError("leaf_size must be in [1, 15]")
    centroids = 0.5 * (box_lower + box_upper)
    codes = morton_codes(centroids, centroids.min(axis=0), centroids.max(axis=0))
    order = np.argsort(codes, kind="stable").astype(np.int64)
    codes = codes[order]

    # Level-synchronous top-down split. Node arrays are in creation (BFS) order.
    begin = [np.array([0], np.int64)]
    end = [np.array([count], np.int64)]
    left_child = []
    levels = []
    first_id = 0
    cur_begin, cur_end = begin[0], end[0]
    while True:
        n = len(cur_begin)
        ids = np.arange(first_id, first_id + n)
        levels.append(ids)
        size = cur_end - cur_begin
        split_mask = size > leaf_size
        lc = np.full(n, -1, np.int64)
        if split_mask.any():
            sb, se = cur_begin[split_mask], cur_end[split_mask]
            a, b = codes[sb], codes[se - 1]
            same = a == b
            split = (sb + se) // 2
            diff = ~same
            if diff.any():
                bit = _highest_bit(a[diff] ^ b[diff]).astype(np.uint64)
                prefix = ((a[diff] >> (bit + np.uint64(1))) << (bit + np.uint64(1))) | (np.uint64(1) << bit)
                split[diff] = np.searchsorted(codes, prefix, side="left")
            split = np.clip(split, sb + 1, se - 1)
            child_begin = np.stack([sb, split], axis=1).reshape(-1)
            child_end = np.stack([split, se], axis=1).reshape(-1)
            next_first = first_id + n
            lc[split_mask] = next_first + 2 * np.arange(int(split_mask.sum()))
            left_child.append(lc)
            begin.append(child_begin)
            end.append(child_end)
            first_id = next_first
            cur_begin, cur_end = child_begin, child_end
        else:
            left_child.append(lc)
            break
    node_begin = np.concatenate(begin)
    node_end = np.concatenate(end)
    left = np.concatenate(left_child)
    total = len(node_begin)
    is_leaf = left < 0

    # Bounds: leaves reduce their primitive boxes; inner nodes their children.
    lower = np.empty((total, 3))
    upper = np.empty((total, 3))
    sorted_lower = box_lower[order]
    sorted_upper = box_upper[order]
    leaf_ids = np.flatnonzero(is_leaf)
    starts = node_begin[leaf_ids]
    ord_leaf = np.argsort(starts)
    starts_sorted = starts[ord_leaf]
    lower[leaf_ids[ord_leaf]] = np.minimum.reduceat(sorted_lower, starts_sorted, axis=0)
    upper[leaf_ids[ord_leaf]] = np.maximum.reduceat(sorted_upper, starts_sorted, axis=0)
    for ids in reversed(levels):
        inner = ids[~is_leaf[ids]]
        if len(inner):
            l, r = left[inner], left[inner] + 1
            lower[inner] = np.minimum(lower[l], lower[r])
            upper[inner] = np.maximum(upper[l], upper[r])

    # Subtree sizes (bottom-up) and preorder indices (top-down).
    subtree = np.ones(total, np.int64)
    for ids in reversed(levels):
        inner = ids[~is_leaf[ids]]
        if len(inner):
            subtree[inner] = 1 + subtree[left[inner]] + subtree[left[inner] + 1]
    pre = np.zeros(total, np.int64)
    for ids in levels:
        inner = ids[~is_leaf[ids]]
        if len(inner):
            pre[left[inner]] = pre[inner] + 1
            pre[left[inner] + 1] = pre[inner] + 1 + subtree[left[inner]]
    escape = pre + subtree
    escape[escape >= total] = -1

    out_lower = np.empty((total, 3), np.float32)
    out_upper = np.empty((total, 3), np.float32)
    out_escape = np.empty(total, np.int32)
    out_first = np.full(total, -1, np.int32)
    out_count = np.zeros(total, np.int32)
    out_lower[pre] = _round_down_f32(lower)
    out_upper[pre] = _round_up_f32(upper)
    out_escape[pre] = escape
    out_first[pre[is_leaf]] = node_begin[is_leaf]
    out_count[pre[is_leaf]] = (node_end - node_begin)[is_leaf]
    return ThreadedBVH(out_lower, out_upper, out_escape, out_first, out_count, order.astype(np.int32))


def validate_threaded_bvh(bvh, box_lower, box_upper):
    """Structural checks used by tests: containment, coverage and threading."""
    n = bvh.node_count
    assert bvh.escape.shape == (n,)
    covered = np.zeros(len(bvh.order), np.int32)
    for node in np.flatnonzero(bvh.leaf_first >= 0):
        first, cnt = bvh.leaf_first[node], bvh.leaf_count[node]
        prims = bvh.order[first:first + cnt]
        covered[first:first + cnt] += 1
        assert np.all(box_lower[prims] >= bvh.lower[node] - 1e-6 * (1 + np.abs(bvh.lower[node])))
        assert np.all(box_upper[prims] <= bvh.upper[node] + 1e-6 * (1 + np.abs(bvh.upper[node])))
    assert np.all(covered == 1), "every primitive must appear in exactly one leaf"
    assert sorted(bvh.order.tolist()) == list(range(len(bvh.order)))
    # Walking with "always descend" must visit every node exactly once in order.
    node, visited = 0, 0
    while node != -1 and visited <= n:
        visited += 1
        node = node + 1 if bvh.leaf_first[node] < 0 else bvh.escape[node]
    assert visited == n, (visited, n)
    return True
