"""warp_fold — watch the fold-and-merge compressor work, literally, in time.

This renders the *actual* `warp_compress.wrapfold` algorithm as it runs. The strand of symbols
(one glowing bead per byte, coloured by value) is **wrapped onto a cylinder whose circumference is
the period the compressor found** — so each coil lands directly on the one below it. Wherever a
cell matches the cell in the coil beneath, it **merges**: it flashes green and collapses away,
leaving only the template coil plus the handful of cells that differed. Then the template is
wrapped again — layer by layer — condensing into the chromosome. Run it backwards and the strand
unwraps, cells reappearing: decompression. What you see is exactly what the codec did; the frame is
the fold step.
"""

import colorsys

import numpy as np
import warp as wp

from ..engine import post
from ..particles import camera_ray, emitter, orbit_ro
from ..scene import Scene
from warp_compress import wrapfold as wf

_TWO_PI = 6.28318530718
_MAX = 128
_CYCLE = 9.0
_CACHE = {}


def _sample(kind):
    if kind == "dna":
        return (b"ACGTACGT" * 15)                       # strict period-8 motif → one clean coil
    if kind == "text":
        return b"wrap the strand into a chromosome " * 3
    rng = np.random.default_rng(4)
    return bytes(int((100 + 80 * np.sin(i * 2 * np.pi / 12) + rng.integers(-5, 5))) & 255
                 for i in range(120))


def _stages(kind):
    """(strands, levels, palette) — the per-fold-level strand + merge mask, computed once."""
    if kind in _CACHE:
        return _CACHE[kind]
    data = _sample(kind)[:_MAX]
    sym = np.frombuffer(data, np.uint8).astype(np.int32)
    core, levels = wf.fold_levels(sym, tol=0)
    strands = [sym]
    for lvl in levels:
        strands.append(strands[-1][:lvl.period].copy())
    vals = sorted(set(int(x) for x in sym))
    palette = {v: colorsys.hsv_to_rgb((k / max(1, len(vals))) % 1.0, 0.85, 1.0)
               for k, v in enumerate(vals)}
    out = (strands, levels, palette)
    _CACHE[kind] = out
    return out


def _smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _layout(kind, p):
    """Bead positions/colours/sizes + backbone for compaction progress ``p`` in [0, 1].

    Each fold level runs in two visible phases: **wrap** (flat strand → a stack of coils of
    circumference = the found period, matches flashing green) then **merge** (the redundant coils
    telescope up into the first coil and fade, leaving one bright template ring — the compressed
    chromosome). Then the template becomes the next level's strand."""
    strands, levels, palette = _stages(kind)
    K = max(1, len(levels))
    seg = 1.0 / K
    g = float(np.clip(p, 0.0, 1.0))
    si = min(int(g / seg), K - 1)
    local = (g - si * seg) / seg
    wrap = _smooth(min(local * 2.0, 1.0))
    merge = _smooth(max(local * 2.0 - 1.0, 0.0))

    strand = strands[si]
    m = len(strand)
    period = levels[si].period if levels else max(2, m)
    same = levels[si].same if levels else np.zeros(max(0, m - period), bool)
    rows = (m + period - 1) // period
    depth_g = si / K
    radius = 0.95 - 0.28 * depth_g
    rowh = 0.5
    dx = 0.26
    green = np.array([0.2, 1.0, 0.35], np.float32)

    def coiled_at(i, row):
        # row 0 (the template coil) sits at the origin so the merged strand telescopes UP into a
        # centred ring; deeper coils hang below during the wrap.
        ang = (i % period) / period * _TWO_PI
        return np.array([radius * np.cos(ang), -float(row) * rowh,
                         radius * np.sin(ang)], np.float32)

    pos = np.zeros((m, 3), np.float32)
    col = np.zeros((m, 3), np.float32)
    siz = np.zeros(m, np.float32)
    for i in range(m):
        flat = np.array([(i - 0.5 * m) * dx, 1.5 - 0.8 * depth_g, 0.0], np.float32)
        row = i // period
        pw = flat * (1.0 - wrap) + coiled_at(i, row) * wrap
        merged = (i >= period) and bool(same[i - period])
        base_c = np.array(palette.get(int(strand[i]), (0.85, 0.85, 0.85)), np.float32)
        if merged:
            # slide up onto the template ring (row 0, same column) and fade as it merges
            pos[i] = pw * (1.0 - merge) + coiled_at(i, 0) * merge
            col[i] = (base_c * (1.0 - wrap) + green * wrap) * (1.0 - 0.85 * merge)
            siz[i] = 0.075 * (1.0 - 0.8 * merge)
        else:
            pos[i] = pw
            col[i] = base_c
            siz[i] = 0.08 + 0.02 * merge                 # template/diff stays and brightens
    # centre the wrapped coil, then let it telescope up to the origin as it merges
    center_off = (rows - 1) * rowh * 0.5
    pos[:, 1] += center_off * wrap * (1.0 - merge)
    # backbone: densify the strand polyline through the current bead positions
    if m >= 2:
        reps = 3
        sei = np.linspace(0, m - 1, m * reps)
        lo = np.floor(sei).astype(int); hi = np.minimum(lo + 1, m - 1)
        f = (sei - lo)[:, None]
        path = (pos[lo] * (1 - f) + pos[hi] * f).astype(np.float32)
    else:
        path = pos.copy()
    return pos, col, siz, path


@wp.kernel
def _kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float, mouse: wp.vec2,
            pos: wp.array(dtype=wp.vec3), col: wp.array(dtype=wp.vec3),
            siz: wp.array(dtype=wp.float32), count: int,
            path: wp.array(dtype=wp.vec3), path_n: int):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    ro = orbit_ro(time, mouse, res, 7.6)
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.5)
    c = wp.vec3(0.02, 0.03, 0.06)
    rail = wp.vec3(0.30, 0.48, 0.85)
    for k in range(path_n):
        c = c + rail * (emitter(ro, rd, path[k], 0.02) * 0.30)
    for k in range(count):
        c = c + col[k] * emitter(ro, rd, pos[k], siz[k])
    img[i, j] = c


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _make(kind):
    def _render(width, height, time, mouse, device):
        pos, col, siz, path = _layout(kind, _progress(time))
        img = wp.zeros((height, width), dtype=wp.vec3, device=device)
        wp.launch(_kernel, dim=(height, width),
                  inputs=[img, width, height, float(time), mouse,
                          wp.array(pos, dtype=wp.vec3, device=device),
                          wp.array(col, dtype=wp.vec3, device=device),
                          wp.array(siz, dtype=wp.float32, device=device), len(pos),
                          wp.array(path, dtype=wp.vec3, device=device), len(path)],
                  device=device)
        wp.synchronize_device(device)
        return post.tonemap(img.numpy(), mode="aces", exposure=1.25, preserve_hue=True)
    return _render


SCENES = [
    Scene(name="warp_fold",
          description="watch the fold-and-merge compressor run in time — the strand wraps onto the "
                      "period the codec found, and cells that match the coil below flash green and "
                      "merge away, condensing into a chromosome, then unwrap on decompression.",
          renderer=_make("dna")),
    Scene(name="warp_fold_words",
          description="the same fold-and-merge on text — a repeated phrase wraps onto its period "
                      "and the matching letters merge, meaning packed layer by layer.",
          renderer=_make("text")),
]
