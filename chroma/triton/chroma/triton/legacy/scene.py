"""The exact words the CUDA backend uploads for a Chroma geometry (NumPy only).

:func:`scene_words` reproduces ``chroma.gpu.GPUGeometry`` and
``chroma.gpu.GPUDetector`` (installed working tree W) without PyCUDA,
including their quirks:

* material and surface tables are ``np.interp(...).astype(float32)`` on the
  (default ``standard_wavelengths``) grid; time CDFs on
  ``np.arange(0, 1000, 0.05)``;
* materials/surfaces referenced only by analytic wire planes are appended
  after ``unique_materials``/``unique_surfaces`` (the first, pointer-array
  construction in ``GPUGeometry`` is the one the kernel reads);
* ``material_codes`` pack 8-bit indices that the kernel sign-extends, so an
  index >= 128 would be read as negative (rejected by :func:`check_legacy_limits`);
* the 124-byte ``WirePlane`` record with its FP32 frame computed by NumPy in
  float32, and ``k_min``/``k_max`` from Python floats;
* ``charge_unit = float32(charge_cdf[0][-1] / 2**16)``.

The CUDA recorder exports the same keys read back from device memory
(:func:`chroma.triton.legacy.record_cuda.export_device_scene`); the verifier
requires both to be byte-identical. ``nodes`` is the geometry's BVH exactly
as ``geometry.bvh`` holds it (split between device and mapped host memory in
CUDA; the split does not change any value).
"""

import numpy as np

TIME_STEP = 0.05


def _interp(grid, prop):
    assert prop is not None, "property must not be None"
    return np.interp(grid, prop[:, 0], prop[:, 1]).astype(np.float32)


def _wire_get(desc, key, default=None):
    return desc.get(key, default) if isinstance(desc, dict) else getattr(desc, key, default)


def material_and_surface_lists(geometry):
    """Materials/surfaces in GPUGeometry pointer order (wire-only extras appended)."""
    materials = list(geometry.unique_materials)
    for desc in (getattr(geometry, "wireplanes", None) or []):
        for mat in (desc.get("material_inner", None), desc.get("material_outer", None)):
            if mat is None or mat in materials:
                continue
            materials.append(mat)
    surfaces = list(geometry.unique_surfaces)
    for desc in (getattr(geometry, "wireplanes", None) or []):
        surface = desc.get("surface", None)
        if surface is None or surface in surfaces:
            continue
        surfaces.append(surface)
    return materials, surfaces


def wireplane_words(geometry, materials, surfaces):
    """[P,31] uint32 words of the WirePlane structs the kernel dereferences."""
    material_lookup = dict(zip(materials, range(len(materials))))
    surface_lookup = dict(zip(surfaces, range(len(surfaces))))
    rows = []
    for desc in (getattr(geometry, "wireplanes", None) or []):
        origin = np.asarray(desc["origin"], dtype=np.float32)
        u_raw = np.asarray(desc["u"], dtype=np.float32)
        v_raw = np.asarray(desc["v"], dtype=np.float32)
        pitch = np.float32(desc["pitch"])
        radius = np.float32(desc["radius"])
        umin = np.float32(desc["umin"])
        umax = np.float32(desc["umax"])
        vmin = np.float32(desc["vmin"])
        vmax = np.float32(desc["vmax"])
        v0 = np.float32(desc["v0"])
        surface = desc.get("surface", None)
        material_inner = desc.get("material_inner", None)
        material_outer = desc.get("material_outer", None)
        color = np.uint32(desc.get("color", 0))
        surface_idx = -1 if surface is None else int(surface_lookup.get(surface, -1))
        if material_outer is None or material_inner is None:
            material_outer_idx = 0
            material_inner_idx = 0
        else:
            material_outer_idx = int(material_lookup[material_outer])
            material_inner_idx = int(material_lookup[material_inner])
        u_norm = u_raw / np.linalg.norm(u_raw)
        v_orth = v_raw - np.dot(v_raw, u_norm) * u_norm
        v_norm = v_orth / np.linalg.norm(v_orth)
        n_norm = np.cross(u_norm, v_norm)
        pitch_f = float(pitch)
        v0_f = float(v0)
        vmin_f = float(vmin)
        vmax_f = float(vmax)
        k_min = int(np.ceil((vmin_f - v0_f) / pitch_f)) if pitch_f > 0 else 0
        k_max = int(np.floor((vmax_f - v0_f) / pitch_f)) if pitch_f > 0 else 0
        floats = np.concatenate([origin, u_raw, v_raw,
                                 np.asarray([pitch, radius, umin, umax, vmin, vmax, v0], np.float32)])
        ints = np.asarray([surface_idx, material_outer_idx, material_inner_idx], np.int32)
        frame = np.concatenate([u_norm.astype(np.float32), v_norm.astype(np.float32),
                                n_norm.astype(np.float32)])
        words = np.concatenate([floats.view(np.uint32), ints.view(np.uint32),
                                np.asarray([color], np.uint32), frame.view(np.uint32),
                                np.asarray([k_min, k_max], np.int32).view(np.uint32)])
        assert words.shape == (31,)
        rows.append(words)
    return np.asarray(rows, np.uint32).reshape(-1, 31)


def scene_words(geometry, wavelengths=None, times=None, require_bvh=True):
    """Dictionary of the exact uploaded words (see module docstring).

    Without ``geometry.bvh`` and with ``require_bvh=False`` the BVH keys
    (``nodes``, ``world_origin``, ``world_scale``) are omitted.
    """
    from chroma.geometry import standard_wavelengths

    if wavelengths is None:
        wavelengths = standard_wavelengths
    wavelength_step = np.unique(np.diff(wavelengths)).item()
    if times is None:
        time_step = TIME_STEP
        times = np.arange(0, 1000, time_step)
    else:
        time_step = np.unique(np.diff(times)).item()
    nw = len(wavelengths)
    nt = len(times)
    out = {}
    out["wavelength_grid"] = np.asarray(wavelengths, np.float32)
    materials, surfaces = material_and_surface_lists(geometry)
    header = []
    tables = {k: [] for k in ("refractive_index", "absorption_length", "scattering_length")}
    comps = {k: [] for k in ("comp_reemission_prob", "comp_reemission_wvl_cdf", "comp_reemission_time_cdf",
                             "comp_absorption_length")}
    num_comp = []
    for material in materials:
        if material is None:
            raise ValueError("one or more triangles is missing a material.")
        tables["refractive_index"].append(_interp(wavelengths, material.refractive_index))
        tables["absorption_length"].append(_interp(wavelengths, material.absorption_length))
        tables["scattering_length"].append(_interp(wavelengths, material.scattering_length))
        n = len(material.comp_reemission_prob)
        num_comp.append(n)
        assert n == len(material.comp_reemission_wvl_cdf) == len(material.comp_reemission_time_cdf) \
            == len(material.comp_absorption_length), "component arrays must be same length"
        comps["comp_reemission_prob"].extend(_interp(wavelengths, c) for c in material.comp_reemission_prob)
        comps["comp_reemission_wvl_cdf"].extend(_interp(wavelengths, c) for c in material.comp_reemission_wvl_cdf)
        comps["comp_reemission_time_cdf"].extend(_interp(times, c) for c in material.comp_reemission_time_cdf)
        comps["comp_absorption_length"].extend(_interp(wavelengths, c) for c in material.comp_absorption_length)
        header.append(np.asarray([n, nw], np.uint32).tolist()
                      + [int(np.float32(wavelength_step).view(np.uint32)),
                         int(np.float32(wavelengths[0]).view(np.uint32)), nt,
                         int(np.float32(time_step).view(np.uint32)),
                         int(np.float32(times[0]).view(np.uint32))])
    for key, rows in tables.items():
        out["material_" + key] = np.asarray(rows, np.float32).reshape(len(materials), nw)
    out["material_header"] = np.asarray(header, np.uint32).reshape(len(materials), 7)
    out["material_comp_offsets"] = np.concatenate([[0], np.cumsum(num_comp)]).astype(np.int64)
    for key, rows in comps.items():
        width = nt if key == "comp_reemission_time_cdf" else nw
        out[key] = np.asarray(rows, np.float32).reshape(len(rows), width)

    fields = ("detect", "absorb", "reemit", "reflect_diffuse", "reflect_specular", "eta", "k", "reemission_cdf")
    stables = {k: [] for k in fields}
    s_header = []
    present = []
    dichroic_offsets = [0]
    dichroic_angles, dichroic_reflect, dichroic_transmit = [], [], []
    angular_offsets = [0]
    ang = {k: [] for k in ("angles", "transmit", "reflect_specular", "reflect_diffuse")}
    for surface in surfaces:
        if surface is None:
            present.append(0)
            for k in fields:
                stables[k].append(np.zeros(nw, np.float32))
            s_header.append([0] * 6)
            dichroic_offsets.append(dichroic_offsets[-1])
            angular_offsets.append(angular_offsets[-1])
            continue
        present.append(1)
        for k in fields:
            stables[k].append(_interp(wavelengths, getattr(surface, k)))
        s_header.append([int(np.uint32(surface.model)), nw, int(np.uint32(surface.transmissive)),
                         int(np.float32(wavelength_step).view(np.uint32)),
                         int(np.float32(wavelengths[0]).view(np.uint32)),
                         int(np.float32(surface.thickness).view(np.uint32))])
        if surface.dichroic_props:
            props = surface.dichroic_props
            dichroic_angles.append(np.asarray(props.angles, dtype=np.float32))
            for i in range(len(props.angles)):
                dichroic_reflect.append(_interp(wavelengths, props.dichroic_reflect[i]))
                dichroic_transmit.append(_interp(wavelengths, props.dichroic_transmit[i]))
            dichroic_offsets.append(dichroic_offsets[-1] + len(props.angles))
        else:
            dichroic_offsets.append(dichroic_offsets[-1])
        if surface.angular_props:
            props = surface.angular_props
            ang["angles"].append(np.asarray(props.angles, dtype=np.float32))
            ang["transmit"].append(np.asarray(props.transmit, dtype=np.float32))
            ang["reflect_specular"].append(np.asarray(props.reflect_specular, dtype=np.float32))
            ang["reflect_diffuse"].append(np.asarray(props.reflect_diffuse, dtype=np.float32))
            angular_offsets.append(angular_offsets[-1] + len(props.angles))
        else:
            angular_offsets.append(angular_offsets[-1])
    for k in fields:
        out["surface_" + k] = np.asarray(stables[k], np.float32).reshape(len(surfaces), nw)
    out["surface_present"] = np.asarray(present, np.uint8)
    out["surface_header"] = np.asarray(s_header, np.uint32).reshape(len(surfaces), 6)
    out["dichroic_offsets"] = np.asarray(dichroic_offsets, np.int64)
    out["dichroic_angles"] = (np.concatenate(dichroic_angles) if dichroic_angles else np.zeros(0, np.float32))
    out["dichroic_reflect"] = np.asarray(dichroic_reflect, np.float32).reshape(-1, nw)
    out["dichroic_transmit"] = np.asarray(dichroic_transmit, np.float32).reshape(-1, nw)
    out["angular_offsets"] = np.asarray(angular_offsets, np.int64)
    for k, v in ang.items():
        out["angular_" + k] = np.concatenate(v).astype(np.float32) if v else np.zeros(0, np.float32)

    out["wireplanes"] = wireplane_words(geometry, materials, surfaces)
    out["vertices"] = np.ascontiguousarray(np.asarray(geometry.mesh.vertices).astype(np.float32).reshape(-1, 3))
    out["triangles"] = np.ascontiguousarray(np.asarray(geometry.mesh.triangles).astype(np.uint32).reshape(-1, 3))
    out["material_codes"] = (((geometry.material1_index & 0xff) << 24)
                             | ((geometry.material2_index & 0xff) << 16)
                             | ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
    out["colors"] = np.asarray(geometry.colors).astype(np.uint32)
    out["solid_id_map"] = np.asarray(geometry.solid_id).astype(np.uint32)
    if getattr(geometry, "bvh", None) is not None:
        out["nodes"] = np.ascontiguousarray(geometry.bvh.nodes).view(np.uint32).reshape(-1, 4)
        out["world_origin"] = np.asarray(geometry.bvh.world_coords.world_origin, np.float32).reshape(3)
        out["world_scale"] = np.asarray([geometry.bvh.world_coords.world_scale], np.float32)
    elif require_bvh:
        raise ValueError("the geometry has no Chroma BVH (geometry.bvh); bitwise mode needs the original BVH")
    if hasattr(geometry, "num_channels"):
        out["solid_id_to_channel_index"] = np.asarray(geometry.solid_id_to_channel_index).astype(np.int32)
        out["time_cdf_x"] = np.asarray(geometry.time_cdf[0]).astype(np.float32)
        out["time_cdf_y"] = np.asarray(geometry.time_cdf[1]).astype(np.float32)
        out["charge_cdf_x"] = np.asarray(geometry.charge_cdf[0]).astype(np.float32)
        out["charge_cdf_y"] = np.asarray(geometry.charge_cdf[1]).astype(np.float32)
        out["detector_header"] = np.concatenate([
            np.asarray([geometry.num_channels(), len(geometry.time_cdf[0]), len(geometry.charge_cdf[0])],
                       np.int32).view(np.uint32),
            np.asarray([np.float32(geometry.charge_cdf[0][-1] / 2**16)], np.float32).view(np.uint32)])
    return out


def check_daq_limits(words):
    """Reject DAQ tables the original run_daq reads out of bounds (fail closed).

    ``Detector._pdf_to_cdf`` (``set_time_dist*``/``set_charge_dist*``) builds
    ``cdf_y`` one entry shorter than ``cdf_x``, while ``run_daq`` searches
    ``len(cdf_x)`` entries of both, reading one float past the allocation.
    """
    header = words["detector_header"].view(np.int32)
    for name, n in (("time", int(header[1])), ("charge", int(header[2]))):
        if len(words[name + "_cdf_x"]) != n or len(words[name + "_cdf_y"]) != n:
            raise NotImplementedError(
                "the detector's %s_cdf x/y lengths differ (%d/%d); the original DAQ reads past the "
                "table (undefined device memory), which cannot be reproduced"
                % (name, len(words[name + "_cdf_x"]), len(words[name + "_cdf_y"])))
    return True


def check_legacy_limits(words):
    """Reject scenes the original kernel reads incorrectly (fail closed)."""
    codes = words["material_codes"]
    for shift, what in ((24, "inner material"), (16, "outer material"), (8, "surface")):
        idx = (codes >> shift) & 0xff
        bad = (idx >= 128) & (idx != 0xff)
        if np.any(bad):
            raise NotImplementedError(
                "a %s index >= 128 is sign-extended by the original kernel (reads outside the table)" % what)
    nodes = words["nodes"]
    if len(nodes) and (nodes[0, 3] >> 28) == 0:
        raise NotImplementedError("a root BVH node that is a leaf is not reproduced")
    return True


_MATERIAL_ROWS = ("material_refractive_index", "material_absorption_length", "material_scattering_length",
                  "material_header")
_COMP_ROWS = ("comp_absorption_length", "comp_reemission_prob", "comp_reemission_wvl_cdf",
              "comp_reemission_time_cdf")
_SURFACE_ROWS = ("surface_detect", "surface_absorb", "surface_reemit", "surface_reflect_diffuse",
                 "surface_reflect_specular", "surface_eta", "surface_k", "surface_reemission_cdf",
                 "surface_header", "surface_present")
_DICHROIC_ROWS = ("dichroic_angles", "dichroic_reflect", "dichroic_transmit")
_ANGULAR_ROWS = ("angular_angles", "angular_transmit", "angular_reflect_specular", "angular_reflect_diffuse")


def _signatures(words, count, rows, ranges):
    import hashlib

    out = []
    for i in range(count):
        h = hashlib.sha256()
        for key in rows:
            h.update(np.ascontiguousarray(np.asarray(words[key])[i]).tobytes())
        for offsets_key, keys in ranges:
            offsets = np.asarray(words[offsets_key])
            a, b = int(offsets[i]), int(offsets[i + 1])
            for key in keys:
                h.update(np.ascontiguousarray(np.asarray(words[key])[a:b]).tobytes())
        out.append(h.hexdigest())
    return out


def _matching(mine, theirs):
    """perm[i] = index in ``theirs`` of the entry equal to ``mine[i]`` (None if impossible)."""
    if len(mine) != len(theirs):
        return None
    free = {}
    for j, sig in enumerate(theirs):
        free.setdefault(sig, []).append(j)
    perm = []
    for sig in mine:
        if not free.get(sig):
            return None
        perm.append(free[sig].pop(0))
    return perm


def _reorder_ranges(words, offsets_key, keys, order):
    offsets = np.asarray(words[offsets_key], np.int64)
    parts = {k: [] for k in keys}
    new = [0]
    for old in order:
        a, b = int(offsets[old]), int(offsets[old + 1])
        for k in keys:
            parts[k].append(np.asarray(words[k])[a:b])
        new.append(new[-1] + b - a)
    out = {offsets_key: np.asarray(new, np.int64)}
    for k in keys:
        arr = np.asarray(words[k])
        out[k] = np.concatenate(parts[k]).astype(arr.dtype) if parts[k] else arr[:0]
    return out


def relabel_scene(mine, recorded):
    """``mine`` with materials and surfaces renumbered as in ``recorded``.

    W orders ``unique_materials``/``unique_surfaces`` by Python set iteration
    (``geometry.silly_unique``), i.e. by object hash, so two processes that
    build the same detector generally number its materials differently. The
    labels only select table rows; entries are matched by the content of
    everything the kernels read for them. Returns None if no matching exists.
    """
    nm = len(mine["material_header"])
    ns = len(mine["surface_header"])
    if len(recorded["material_header"]) != nm or len(recorded["surface_header"]) != ns:
        return None
    mranges = [("material_comp_offsets", _COMP_ROWS)]
    sranges = [("dichroic_offsets", _DICHROIC_ROWS[1:]), ("angular_offsets", _ANGULAR_ROWS)]
    mperm = _matching(_signatures(mine, nm, _MATERIAL_ROWS, mranges),
                      _signatures(recorded, nm, _MATERIAL_ROWS, mranges))
    # dichroic angles are stored per surface in one flat array with the same offsets
    sperm = _matching(_signatures(mine, ns, _SURFACE_ROWS, sranges + [("dichroic_offsets", ("dichroic_angles",))]),
                      _signatures(recorded, ns, _SURFACE_ROWS, sranges + [("dichroic_offsets", ("dichroic_angles",))]))
    if mperm is None or sperm is None:
        return None
    out = dict(mine)
    minv = np.argsort(mperm)  # new index -> old index
    sinv = np.argsort(sperm)
    for key in _MATERIAL_ROWS:
        out[key] = np.asarray(mine[key])[minv]
    out.update(_reorder_ranges(mine, "material_comp_offsets", _COMP_ROWS, minv))
    for key in _SURFACE_ROWS:
        out[key] = np.asarray(mine[key])[sinv]
    out.update(_reorder_ranges(mine, "dichroic_offsets", _DICHROIC_ROWS, sinv))
    out.update(_reorder_ranges(mine, "angular_offsets", _ANGULAR_ROWS, sinv))
    mmap = np.arange(256, dtype=np.uint32)
    mmap[:nm] = np.asarray(mperm, np.uint32)
    smap = np.arange(256, dtype=np.uint32)
    smap[:ns] = np.asarray(sperm, np.uint32)
    codes = np.asarray(mine["material_codes"], np.uint32)
    out["material_codes"] = ((mmap[(codes >> 24) & 0xFF] << 24) | (mmap[(codes >> 16) & 0xFF] << 16)
                             | (smap[(codes >> 8) & 0xFF] << 8) | (codes & 0xFF)).astype(np.uint32)
    planes = np.array(mine["wireplanes"], np.uint32, copy=True).reshape(-1, 31)
    if len(planes):
        surf = planes[:, 16].view(np.int32)
        planes[:, 16] = np.where(surf >= 0, smap[np.clip(surf, 0, 255)], planes[:, 16])
        rec = np.asarray(recorded["wireplanes"], np.uint32).reshape(-1, 31)
        for col in (17, 18):
            mapped = mmap[planes[:, col] & 0xFF]
            # a plane without materials is written as index 0 in every process
            if len(rec) == len(planes):
                mapped = np.where((planes[:, col] == 0) & (rec[:, col] == 0), 0, mapped)
            planes[:, col] = mapped
    out["wireplanes"] = planes
    return out


def compare_scene_relabeled(mine, recorded):
    """compare_scene after :func:`relabel_scene` (keys of ``mine`` only, no pad words)."""
    relabeled = relabel_scene(mine, recorded)
    if relabeled is None:
        return [("materials/surfaces", "no content matching between the two scenes")]
    return compare_scene(relabeled, {k: v for k, v in recorded.items() if k in relabeled})


def compare_scene(expected, actual):
    """List of (key, message) describing every difference between two scenes."""
    problems = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected or key not in actual:
            problems.append((key, "missing on one side"))
            continue
        a = np.ascontiguousarray(expected[key])
        b = np.ascontiguousarray(actual[key])
        if a.shape != b.shape:
            problems.append((key, "shape %s != %s" % (a.shape, b.shape)))
            continue
        if a.tobytes() != b.tobytes():
            flat_a = a.reshape(-1).view(np.uint8)
            flat_b = b.reshape(-1).view(np.uint8)
            first = int(np.flatnonzero(flat_a != flat_b)[0]) // max(1, a.itemsize)
            problems.append((key, "first differing element %d" % first))
    return problems
