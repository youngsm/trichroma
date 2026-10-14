"""Illustrations for docs/how_it_scales.md (hand-built SVG, no dependencies).

usage: python make_figures.py OUTDIR
"""
import math
import os
import random
import sys

W = 760
FONT = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
CUDA, TRI, TRI2 = "#D55E00", "#0072B2", "#56B4E9"
KIND = {"step": "#0072B2", "descend": "#E69F00", "leaf": "#009E73", "new": "#CC79A7"}


class Svg(object):
    def __init__(self, height, width=W):
        self.w, self.h, self.items = width, height, []

    def add(self, s):
        self.items.append(s)

    def rect(self, x, y, w, h, fill, stroke="none", rx=2, dash=None, sw=1, opacity=1):
        d = " stroke-dasharray='%s'" % dash if dash else ""
        self.add("<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' rx='%s' fill='%s' stroke='%s' "
                 "stroke-width='%s' opacity='%s'%s/>" % (x, y, w, h, rx, fill, stroke, sw, opacity, d))

    def line(self, x1, y1, x2, y2, stroke=MUTED, sw=1, dash=None, arrow=False):
        d = " stroke-dasharray='%s'" % dash if dash else ""
        a = " marker-end='url(#arrow)'" if arrow else ""
        self.add("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' stroke-width='%s'%s%s/>"
                 % (x1, y1, x2, y2, stroke, sw, d, a))

    def path(self, d, stroke=MUTED, sw=1, fill="none", dash=None, arrow=False):
        ds = " stroke-dasharray='%s'" % dash if dash else ""
        a = " marker-end='url(#arrow)'" if arrow else ""
        self.add("<path d='%s' stroke='%s' stroke-width='%s' fill='%s'%s%s/>" % (d, stroke, sw, fill, ds, a))

    def text(self, x, y, s, size=13, fill=INK, anchor="start", weight="normal", italic=False):
        st = " font-style='italic'" if italic else ""
        self.add("<text x='%.1f' y='%.1f' %s font-size='%s' fill='%s' text-anchor='%s' font-weight='%s'%s>%s</text>"
                 % (x, y, FONT, size, fill, anchor, weight, st, s))

    def circle(self, cx, cy, r, fill="none", stroke=MUTED, sw=1, dash=None):
        d = " stroke-dasharray='%s'" % dash if dash else ""
        self.add("<circle cx='%.1f' cy='%.1f' r='%.1f' fill='%s' stroke='%s' stroke-width='%s'%s/>"
                 % (cx, cy, r, fill, stroke, sw, d))

    def save(self, path):
        head = ("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d'>"
                "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
                "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='%s'/></marker>"
                "<pattern id='hatch' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
                "<line x1='0' y1='0' x2='0' y2='6' stroke='%s' stroke-width='2'/></pattern></defs>"
                "<rect width='100%%' height='100%%' fill='white' rx='8'/>" % (self.w, self.h, self.w, self.h, MUTED, GRID))
        with open(path, "w") as f:
            f.write(head + "\n".join(self.items) + "</svg>\n")


def lanes(out):
    """CUDA Chroma relaunches every step, then waits on its slowest lanes; TriChroma lanes refill."""
    rng = random.Random(4)
    nl, rh, gap = 8, 13, 4
    x0, x1 = 118, 740
    s = Svg(410)
    s.text(20, 26, "CUDA Chroma: one step per launch, then the rest of the steps in one launch", 14, weight="bold")
    top = 40
    step, pause = 13, 11
    x = x0
    for k in range(9):  # one-step launches: every lane busy, then a pause to regroup
        for i in range(nl):
            s.rect(x, top + i * (rh + gap), step, rh, CUDA, rx=1.5)
        s.rect(x + step + 1, top - 2, pause - 2, nl * (rh + gap), "url(#hatch)", rx=0)
        x += step + pause
    s.text(x0, top + nl * (rh + gap) + 14, "hatched: relaunch, reload every photon,", 11, MUTED)
    s.text(x0, top + nl * (rh + gap) + 28, "re-queue survivors, ask the host how many", 11, MUTED)
    xs = x + 4
    remaining = [rng.choice([2, 3, 4, 6, 9, 14, 21, 30]) for _ in range(nl)]
    remaining[5] = 31
    for i, r in enumerate(remaining):
        y = top + i * (rh + gap)
        s.rect(xs, y, r * step * 0.92, rh, CUDA, rx=1.5)
        end = xs + r * step * 0.92
        longest = xs + max(remaining) * step * 0.92
        if end < longest - 1:
            s.rect(end, y, longest - end, rh, "none", stroke=GRID, dash="3,2", rx=1.5)
    s.text(xs + 4, top + nl * (rh + gap) + 14, "when few remain: idle threads (dashed) wait", 11, MUTED)
    s.text(xs + 4, top + nl * (rh + gap) + 28, "for the warp's longest photon", 11, MUTED)
    s.text(22, top + nl * (rh + gap) / 2 + 4, "32 threads", 12, MUTED)
    s.text(22, top + nl * (rh + gap) / 2 + 19, "of a warp", 12, MUTED)

    top2 = 250
    s.text(20, top2 - 14, "TriChroma: a thread that finishes a photon takes the next one at once", 14, weight="bold")
    total = x1 - x0 - 190
    for i in range(nl):
        y = top2 + i * (rh + gap)
        x, k = x0, 0
        while x < x0 + total - 6:
            length = min(rng.choice([2, 3, 5, 7, 11, 17, 24]) * step * 0.92, x0 + total - x)
            s.rect(x + 0.5, y, length - 1, rh, TRI if k % 2 == 0 else TRI2, rx=1.5)
            x += length
            k += 1
    s.text(22, top2 + nl * (rh + gap) / 2 + 4, "32 threads", 12, MUTED)
    s.text(22, top2 + nl * (rh + gap) / 2 + 19, "of a warp", 12, MUTED)
    s.text(x0, top2 + nl * (rh + gap) + 14, "each colour change is a new photon: no pauses, no waiting for other threads",
           11, MUTED)
    s.line(x0, 400, x1, 400, GRID, arrow=True)
    s.text(x1, 392, "time", 11, MUTED, anchor="end")
    s.save(out)


def warp_modes(out):
    """A warp runs one instruction stream: lanes that want other work sit out, unless work is gathered by kind."""
    rng = random.Random(7)
    s = Svg(300)
    nl, rh, gap, cw = 8, 13, 4, 50
    kinds = ["step", "descend", "leaf"]
    names = {"step": "step (physics, top-level tree)", "descend": "descend into a mesh", "leaf": "test triangles"}

    def grid(x0, top, cols, busy):
        for c, kind in enumerate(cols):
            for i in range(nl):
                y, x = top + i * (rh + gap), x0 + c * (cw + 6)
                if busy(c, i):
                    s.rect(x, y, cw, rh, KIND[kind], rx=1.5)
                else:
                    s.rect(x, y, cw, rh, "none", stroke=GRID, dash="3,2", rx=1.5)

    s.text(20, 26, "Each thread does what its own photon needs", 14, weight="bold")
    need = [[rng.choice(kinds) for _ in range(3)] for _ in range(nl)]
    cols = kinds * 2
    wants = [rng.random() < 0.38 for _ in range(len(cols) * nl)]
    grid(24, 44, cols, lambda c, i: wants[c * nl + i])
    for c, kind in enumerate(cols):
        s.text(24 + c * (cw + 6) + cw / 2, 44 + nl * (rh + gap) + 14, kind if kind != "descend" else "desc.", 10,
               MUTED, anchor="middle")
    busy = sum(wants) / len(wants)
    s.text(24, 44 + nl * (rh + gap) + 34, "the warp takes turns; ~%d%% of thread slots do work" % round(100 * busy), 12,
           MUTED)

    x0 = 410
    s.text(x0 - 12, 26, "The warp picks one kind and gathers 32 of it", 14, weight="bold")
    grid(x0, 44, ["step", "step", "descend", "leaf", "leaf", "step"], lambda c, i: True)
    for c, kind in enumerate(["step", "step", "descend", "leaf", "leaf", "step"]):
        s.text(x0 + c * (cw + 6) + cw / 2, 44 + nl * (rh + gap) + 14, kind if kind != "descend" else "desc.", 10,
               MUTED, anchor="middle")
    s.text(x0, 44 + nl * (rh + gap) + 34, "photons wait in small queues until their kind runs", 12, MUTED)
    y = 250
    for k, kind in enumerate(kinds):
        s.rect(24 + k * 250, y, 14, 14, KIND[kind], rx=2)
        s.text(24 + k * 250 + 20, y + 11.5, names[kind], 12)
    s.save(out)


def random_numbers(out):
    """curand state per thread slot versus counter-based numbers indexed by (photon, step)."""
    s = Svg(290)
    s.text(20, 26, "CUDA Chroma: every thread slot owns a generator", 14, weight="bold")
    slots, bw = 5, 56
    for launch, (photons, y) in enumerate([("ABCDE", 70), ("BDE", 150)]):
        s.text(20, y + 22, "launch %d" % (launch + 1), 12, MUTED)
        for i in range(slots):
            x = 86 + i * (bw + 5)
            s.rect(x, y, bw, 34, "#fbeee6", stroke=CUDA, rx=4)
            s.text(x + bw / 2, y + 13, "slot %d" % i, 10, MUTED, anchor="middle")
            if i < len(photons):
                s.text(x + bw / 2, y + 28, "photon %s" % photons[i], 11, anchor="middle", weight="bold")
    s.path("M %d %d C %d %d, %d %d, %d %d" % (86 + 3 * 61 + 28, 106, 86 + 3 * 61 + 28, 128, 86 + 1 * 61 + 28, 124,
                                             86 + 1 * 61 + 28, 148), CUDA, 1.5, arrow=True)
    s.text(86, 212, "survivors are re-queued in whatever order the atomics", 12, MUTED)
    s.text(86, 228, "give, so photon D draws from slot 3, then from slot 1:", 12, MUTED)
    s.text(86, 244, "its numbers depend on the schedule", 12, MUTED)

    x0 = 436
    s.text(x0 - 30, 26, "TriChroma: numbers indexed by photon and step", 14, weight="bold")
    cell, rows, cols = 28, 6, 10
    for r in range(rows):
        s.text(x0 - 8, 70 + r * cell + 19, "ABCDEF"[r], 11, MUTED, anchor="end")
        for c in range(cols):
            hit = (r, c) == (3, 6)
            s.rect(x0 + c * cell, 70 + r * cell, cell - 3, cell - 3, TRI if hit else "#eaf2f8", rx=3)
    for c in range(cols):
        s.text(x0 + c * cell + 12.5, 64, str(c), 10, MUTED, anchor="middle")
    s.text(x0 + cols * cell / 2, 50, "step", 11, MUTED, anchor="middle")
    s.text(x0 - 12, 64, "photon", 11, MUTED, anchor="middle")
    s.text(x0, 256, "u = Philox(seed, photon D, step 6): any thread can compute", 12, MUTED)
    s.text(x0, 272, "it, in any order, and always gets the same number", 12, MUTED)
    s.save(out)


def pmt(s, cx, cy, r, stroke, dense):
    """A PMT-like dome drawn as a triangle fan."""
    pts = [(cx + r * math.cos(math.pi * (1 - t / dense)), cy - r * math.sin(math.pi * (1 - t / dense)))
           for t in range(dense + 1)]
    d = "M %.1f %.1f " % pts[0] + " ".join("L %.1f %.1f" % p for p in pts[1:]) + " Z"
    s.path(d, stroke, 1.2, "#ffffff")
    for p in pts[1:-1]:
        s.line(cx, cy, p[0], p[1], stroke, 0.6)
    s.rect(cx - r * 0.25, cy, r * 0.5, r * 0.45, "#ffffff", stroke=stroke, rx=1)


def instancing(out):
    s = Svg(300)
    s.text(20, 26, "Flattened: every copy is stored and indexed", 14, weight="bold")
    for i in range(3):
        for j in range(5):
            pmt(s, 50 + j * 66, 88 + i * 70, 24, CUDA, 9)
    s.text(20, 282, "30,210 PMTs x 600 triangles = 18M triangles in one tree", 12, MUTED)

    x0 = 420
    s.text(x0 - 10, 26, "Instanced: one mesh, many placements", 14, weight="bold")
    boxes = [(x0 + 10 + j * 50, 70 + i * 44) for i in range(3) for j in range(3)]
    for bx, by in boxes:
        s.rect(bx, by, 40, 32, "#eaf2f8", stroke=TRI, rx=3)
    s.text(x0 + 10, 216, "placements: a box, a rotation", 12, MUTED)
    s.text(x0 + 10, 232, "and an offset each (a small tree)", 12, MUTED)
    pmt(s, 690, 150, 40, TRI, 16)
    s.text(690, 212, "the mesh, once", 12, MUTED, anchor="middle")
    for bx, by in boxes[2::3]:
        s.path("M %.1f %.1f Q %.1f %.1f %.1f %.1f" % (bx + 42, by + 16, 620, by + 16, 646, 142), TRI, 1, dash="4,3",
               arrow=True)
    s.text(x0 + 10, 266, "a ray that enters a placement's box is moved into the mesh's", 12, MUTED)
    s.text(x0 + 10, 282, "own frame and walks the shared mesh tree", 12, MUTED)
    s.save(out)


def stackless(out):
    s = Svg(250)
    s.text(20, 26, "A tree stored in the order a ray should visit it", 14, weight="bold")
    pos = {"A": (170, 60), "B": (95, 120), "C": (245, 120), "D": (60, 180), "E": (130, 180), "F": (210, 180),
           "G": (280, 180)}
    for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]:
        s.line(pos[a][0], pos[a][1], pos[b][0], pos[b][1], GRID, 1.5)
    for k, (x, y) in pos.items():
        s.circle(x, y, 15, "#eaf2f8", TRI, 1.5)
        s.text(x, y + 4.5, k, 12, anchor="middle", weight="bold")
    s.text(40, 228, "(for rays heading right: the near child first)", 12, MUTED)

    x0, cw = 360, 50
    order = "ABDECFG"
    skip = {"A": None, "B": "C", "D": "E", "E": "C", "C": None, "F": "G", "G": None}
    for i, k in enumerate(order):
        s.rect(x0 + i * (cw + 4), 100, cw, 34, "#eaf2f8", stroke=TRI, rx=4)
        s.text(x0 + i * (cw + 4) + cw / 2, 122, k, 13, anchor="middle", weight="bold")
    for i, k in enumerate(order[:-1]):
        s.line(x0 + i * (cw + 4) + cw - 6, 150, x0 + (i + 1) * (cw + 4) + 6, 150, TRI, 1.2, arrow=True)
    for k, t in skip.items():
        if t:
            a, b = order.index(k), order.index(t)
            xa, xb = x0 + a * (cw + 4) + cw / 2, x0 + b * (cw + 4) + cw / 2
            s.path("M %.1f 98 C %.1f %.1f, %.1f %.1f, %.1f 98" % (xa, xa, 98 - 18 * (b - a), xb, 98 - 18 * (b - a), xb),
                   CUDA, 1.2, dash="4,3", arrow=True)
    s.text(x0, 176, "box hit: go to the next entry", 12, TRI)
    s.text(x0, 194, "box missed: jump past its subtree (dashed)", 12, CUDA)
    s.text(x0, 222, "no stack to keep; one copy per direction octant (8)", 12, MUTED)
    s.save(out)


def pipeline(out):
    s = Svg(240)
    rows = [("host", 60), ("GPU", 88)]

    def lane(title, top, blocks):
        s.text(20, top - 12, title, 14, weight="bold")
        for name, y in rows:
            s.text(96, top + y - 60 + 14, name, 12, MUTED, anchor="end")
        for row, x, w, colour, label in blocks:
            y = top + rows[row][1] - 60
            s.rect(x, y, w, 20, colour, rx=3)
            s.text(x + w / 2, y + 14, label, 10.5, "#ffffff", anchor="middle")

    seq = []
    x = 110
    for k in range(1, 4):
        seq += [(0, x, 60, MUTED, "draw %d" % k), (1, x + 62, 84, CUDA, "propagate %d" % k),
                (0, x + 148, 50, "#8c959f", "use %d" % k)]
        x += 202
    lane("One batch at a time, photons made on the host", 44, seq)
    ov = []
    x = 110
    for k in range(1, 6):
        ov.append((1, x, 90, TRI, "propagate %d" % k))
        if k > 1:
            ov.append((0, x, 56, "#8c959f", "use %d" % (k - 1)))
        x += 92
    ov.append((0, x, 56, "#8c959f", "use 5"))
    lane("Pipelined, photons drawn on the GPU", 140, ov)
    s.text(110, 226, "the GPU never waits for the host; batch k+1 runs while the caller reads batch k", 12, MUTED)
    s.save(out)


if __name__ == "__main__":
    out = sys.argv[1]
    for f in (lanes, warp_modes, random_numbers, instancing, stackless, pipeline):
        f(os.path.join(out, f.__name__ + ".svg"))
