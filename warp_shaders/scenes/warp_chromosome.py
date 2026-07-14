"""warp_chromosome — watch data fold and coil into a chromosome *in time*, then unwind.

The companion to the ``warp_compress`` algorithm (``docs/research/44-warp-compression.md``): not a
before/after, but the **process**. A strand of symbols (each a glowing bead) is laid on a curve
that winds tighter every time the coiler wraps a frequent pair into a nucleosome — so over the
animation the loose strand condenses, layer by layer, into a compact supercoiled chromosome
(compression), then unwinds back to the raw strand (decompression). Beads are coloured by *what
they are*: literal symbols by value (the four DNA letters land on four hues), nucleosomes by how
many wrapping layers deep they sit (a cool→hot ramp), and sized by how much they unwrap to (their
mass). Several data *types* — DNA, text, bytes — each get their own scene, all animated the same
way.

Deterministic: the coil history is computed once on the host with ``warp_compress`` and replayed
frame by frame; the Warp kernel only draws the beads + backbone as glowing impostors.
"""

import colorsys

import numpy as np
import warp as wp

from ..engine import post
from ..particles import camera_ray, emitter, orbit_ro
from ..scene import Scene
import warp_compress as wc

_TWO_PI = 6.28318530718
_MAX_BEADS = 112          # cap rendered strand length so the raw state stays legible + fast
_CYCLE = 8.0              # time-units for one compress→decompress round trip


# --------------------------------------------------------------------------- host: coil history
_CACHE = {}


def _sample(kind: str) -> bytes:
    if kind == "dna":
        rng = np.random.default_rng(7)
        motifs = [b"ACGT", b"ACGTACGT", b"GGGCCC", b"TATATA", b"CGCG", b"AATT"]
        return b"".join(motifs[int(i)] for i in rng.integers(0, len(motifs), 26))
    if kind == "text":
        return (b"fold the card into a cube and coil it like a chromosome ") * 3
    # bytes: a structured-with-noise signal
    rng = np.random.default_rng(3)
    return bytes(int(min(255, max(0, 128 + 90 * np.sin(i * 0.15) + rng.integers(-8, 8))))
                 for i in range(150))


def _history(kind: str):
    """(snapshots, size_map, depth_map, palette, max_depth) — computed once per data type."""
    if kind in _CACHE:
        return _CACHE[kind]
    data = _sample(kind)
    symbols = list(data)
    if len(symbols) > _MAX_BEADS:
        symbols = symbols[:_MAX_BEADS]
    snaps, chrom = wc.coil_snapshots(symbols, base=256, max_snaps=72)
    size_map, depth_map = wc.symbol_metrics(chrom)
    max_depth = max([1] + list(depth_map.values()))
    # literal palette: distinct hue per distinct literal value present
    literals = sorted(set(symbols))
    palette = {}
    for k, v in enumerate(literals):
        h = (k / max(1, len(literals))) % 1.0
        palette[v] = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    out = (snaps, size_map, depth_map, palette, max_depth)
    _CACHE[kind] = out
    return out


def _heat(t: float):
    """Cool→hot ramp for coil depth: deep-blue → cyan → gold → white."""
    t = float(np.clip(t, 0.0, 1.0))
    stops = np.array([[0.10, 0.15, 0.55], [0.10, 0.75, 0.95],
                      [1.00, 0.80, 0.25], [1.00, 0.98, 0.90]])
    x = t * (len(stops) - 1)
    i = int(np.floor(x)); i = min(i, len(stops) - 2)
    f = x - i
    return tuple(stops[i] * (1 - f) + stops[i + 1] * f)


def _coil_point(t, g):
    """A point on the strand at fractional position ``t`` and compaction ``g`` — a two-level
    coil: a primary helix winding around a supercoiled axis, condensing as ``g`` → 1."""
    turns1 = 3.0 + 8.0 * g                 # primary winding tightens with compaction
    r1 = 0.10 + 0.52 * g
    turns2 = 2.2                           # the axis itself supercoils at high g
    r2 = 0.70 * g
    height = 6.2 - 3.7 * g                 # condenses vertically
    a2 = t * turns2 * _TWO_PI
    ax = r2 * np.cos(a2)
    ay = (t - 0.5) * height
    az = r2 * np.sin(a2)
    a1 = t * turns1 * _TWO_PI
    return np.stack([ax + r1 * np.cos(a1), ay, az + r1 * np.sin(a1)], axis=-1)


def _layout(kind: str, p: float):
    """Bead positions/colours/sizes + a dense backbone polyline at compaction progress ``p``."""
    snaps, size_map, depth_map, palette, max_depth = _history(kind)
    g = float(np.clip(p, 0.0, 1.0))
    k = int(round(g * (len(snaps) - 1)))
    strand = snaps[k]
    m = len(strand)
    base = 256

    ts = (np.arange(m) + 0.5) / max(1, m)
    pos = _coil_point(ts, g).astype(np.float32)
    col = np.zeros((m, 3), np.float32)
    siz = np.zeros(m, np.float32)
    for i, sym in enumerate(strand):
        if sym < base:
            col[i] = palette.get(sym, (0.8, 0.8, 0.8))
            siz[i] = 0.055 + 0.02 * g
        else:
            d = depth_map.get(sym, 1)
            mass = size_map.get(sym, 1)
            col[i] = _heat(d / max_depth)
            siz[i] = (0.06 + 0.032 * np.log2(1.0 + mass)) * (0.8 + 0.5 * g)

    # dense backbone samples of the SAME curve, so the coil reads even when few beads remain
    pts = 256
    path = _coil_point(np.linspace(0.0, 1.0, pts), g).astype(np.float32)
    return pos, col, siz, path


# --------------------------------------------------------------------------- device: draw beads
@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, pos: wp.array(dtype=wp.vec3), col: wp.array(dtype=wp.vec3),
                   siz: wp.array(dtype=wp.float32), count: int,
                   path: wp.array(dtype=wp.vec3), path_n: int):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    ro = orbit_ro(time, mouse, res, 8.6)
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.5)

    c = wp.vec3(0.02, 0.03, 0.06)                      # faint cold background

    # backbone: a continuous glowing rail sampled densely along the coil curve
    rail = wp.vec3(0.32, 0.5, 0.85)
    for k in range(path_n):
        c = c + rail * (emitter(ro, rd, path[k], 0.02) * 0.35)

    # nucleosome beads on the rail
    for k in range(count):
        c = c + col[k] * emitter(ro, rd, pos[k], siz[k])

    img[i, j] = c


def _progress(time: float) -> float:
    """Triangle wave: 0 → 1 (compress) → 0 (decompress) across one _CYCLE."""
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _make_renderer(kind: str):
    def _render(width, height, time, mouse, device):
        pos, col, siz, path = _layout(kind, _progress(time))
        n = len(pos)
        pos_a = wp.array(pos, dtype=wp.vec3, device=device)
        col_a = wp.array(col, dtype=wp.vec3, device=device)
        siz_a = wp.array(siz, dtype=wp.float32, device=device)
        path_a = wp.array(path, dtype=wp.vec3, device=device)
        img = wp.zeros((height, width), dtype=wp.vec3, device=device)
        wp.launch(_render_kernel, dim=(height, width),
                  inputs=[img, width, height, float(time), mouse, pos_a, col_a, siz_a, n,
                          path_a, len(path)],
                  device=device)
        wp.synchronize_device(device)
        return post.tonemap(img.numpy(), mode="aces", exposure=1.25, preserve_hue=True)
    return _render


SCENES = [
    Scene(name="warp_chromosome",
          description="watch a DNA-like strand fold and coil into a chromosome in time — each "
                      "frequent pair the warp_compress coiler wraps into a nucleosome winds the "
                      "strand tighter, layer by layer, then it unwinds on decompression; beads are "
                      "coloured by symbol and coil depth, sized by mass.",
          renderer=_make_renderer("dna")),
    Scene(name="warp_fold_text",
          description="the warp_compress chromosome forming from text — words' repeated letter "
                      "pairs coil into nucleosomes and condense, then unwind (compression as "
                      "animation, not before/after).",
          renderer=_make_renderer("text")),
    Scene(name="warp_fold_bytes",
          description="the warp_compress chromosome forming from a noisy byte signal — motifs "
                      "wrap into a supercoiled strand and back, the compression process observable "
                      "over time.",
          renderer=_make_renderer("bytes")),
]
