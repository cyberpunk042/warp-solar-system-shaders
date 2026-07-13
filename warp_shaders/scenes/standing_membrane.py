"""Standing membrane — a circular drumhead frozen in a Bessel eigenmode.

A drum skin clamped at its rim can only vibrate in discrete **normal modes**: the radial part
is a **Bessel function** ``J_m(k·r)`` (chosen so the skin is still at the edge) and the angular
part is ``cos(mθ)``, so mode ``(m, n)`` has ``m`` nodal **diameters** and ``n−1`` interior nodal
**circles** — the pattern a real drum sings. This scene ray-marches that surface as a 3-D
height field ``y = A·J_m(k·r)·cos(mθ)`` over the disk, lit and coloured by displacement (warm
crests, cool troughs), so the nodal lines read as the still grey seams between the hills and
valleys. Over ``--frames`` the whole skin oscillates in place — a standing wave, breathing. See
``docs/research/41-waves-and-resonance.md``.
"""

import math

import numpy as np

from ..engine import post
from ..scene import Scene

# first positive zeros j_{m,n} of J_m — set k = j_{m,n}/R so the rim (r=R) is a node
_ZEROS = {(2, 2): 8.417, (3, 1): 6.380, (1, 2): 7.016, (2, 1): 5.136, (0, 3): 8.654}
_MODE = (3, 1)
_R = 1.0
_AMP = 0.46


def _j0(x):
    ax = np.abs(x)
    small = ax < 3.0
    t = (x / 3.0) ** 2
    js = (1.0 - 2.2499997 * t + 1.2656208 * t**2 - 0.3163866 * t**3
          + 0.0444479 * t**4 - 0.0039444 * t**5 + 0.00021 * t**6)
    z = 3.0 / np.maximum(ax, 3.0)          # clamp: large-x branch only used where ax≥3
    f = (0.79788456 - 0.00000077 * z - 0.0055274 * z**2 - 0.00009512 * z**3
         + 0.00137237 * z**4 - 0.00072805 * z**5 + 0.00014476 * z**6)
    th = (ax - 0.78539816 - 0.04166397 * z - 0.00003954 * z**2 + 0.00262573 * z**3
          - 0.00054125 * z**4 - 0.00029333 * z**5 + 0.00013558 * z**6)
    jl = f / np.sqrt(np.maximum(ax, 1e-6)) * np.cos(th)
    return np.where(small, js, jl)


def _j1(x):
    ax = np.abs(x)
    small = ax < 3.0
    t = (x / 3.0) ** 2
    js = x * (0.5 - 0.56249985 * t + 0.21093573 * t**2 - 0.03954289 * t**3
              + 0.00443319 * t**4 - 0.00031761 * t**5 + 0.00001109 * t**6)
    z = 3.0 / np.maximum(ax, 3.0)          # clamp: large-x branch only used where ax≥3
    f = (0.79788456 + 0.00000156 * z + 0.01659667 * z**2 + 0.00017105 * z**3
         - 0.00249511 * z**4 + 0.00113653 * z**5 - 0.00020033 * z**6)
    th = (ax - 2.35619449 + 0.12499612 * z + 0.00005650 * z**2 - 0.00637879 * z**3
          + 0.00074348 * z**4 + 0.00079824 * z**5 - 0.00029166 * z**6)
    jl = f / np.sqrt(np.maximum(ax, 1e-6)) * np.cos(th) * np.sign(x)
    return np.where(small, js, jl)


def _jm(m, x):
    if m == 0:
        return _j0(x)
    if m == 1:
        return _j1(x)
    jm1, jm = _j0(x), _j1(x)          # upward recurrence J_{k+1} = (2k/x)J_k - J_{k-1}
    for k in range(1, m):
        jp = (2.0 * k) / np.maximum(np.abs(x), 1e-6) * jm - jm1
        jm1, jm = jm, jp
    return jm


def _render(width, height, time, mouse, device):
    osc = math.cos(time * 2.0)              # the whole mode breathes over frames
    amp = _AMP * (0.5 + 0.5 * osc)
    m, _n = _MODE
    k = _ZEROS[_MODE] / _R

    # dense polar sample of the drum surface, painter's-ordered splat (no cliff, no march)
    nr, nth = 420, 940
    rr = (_R * np.sqrt(np.linspace(0.0, 1.0, nr))).astype(np.float32)         # area-uniform
    tt = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False, dtype=np.float32)
    R2, T2 = np.meshgrid(rr, tt, indexing="ij")
    ct, st = np.cos(T2), np.sin(T2)
    jm = _jm(m, k * R2)
    jmm1 = _jm(m - 1, k * R2) if m >= 1 else _j1(k * R2)  # J_{m-1}
    jmp1 = _jm(m + 1, k * R2)
    dj = 0.5 * (jmm1 - jmp1)                              # J_m'(x)
    h = amp * jm * np.cos(m * T2)
    dh_dr = amp * k * dj * np.cos(m * T2)
    dh_dt = -amp * float(m) * jm * np.sin(m * T2)

    px = R2 * ct; pz = R2 * st
    pos = np.stack([px, h, pz], axis=-1).reshape(-1, 3)

    # exact surface normal from the two tangents
    Tr = np.stack([ct, dh_dr, st], axis=-1)
    Tt = np.stack([-R2 * st, dh_dt, R2 * ct], axis=-1)
    nrm = np.cross(Tt, Tr).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    nrm *= np.sign(nrm[:, 1:2] + 1e-9)                    # face up

    disp = (jm * np.cos(m * T2)).reshape(-1)             # phase-stable displacement for colour

    # camera
    ang = 0.6 + float(mouse[0]) * 0.004
    eye = np.array([3.35 * math.sin(ang), 1.78, 3.35 * math.cos(ang)], np.float32)
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0, 1, 0], np.float32)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (height * 0.5) / math.tan(math.radians(40.0) * 0.5)

    rel = pos - eye
    cz = rel @ fwd
    cx = rel @ right
    cy = rel @ up
    sx = np.round(width * 0.5 + (cx / cz) * f).astype(np.int64)
    sy = np.round(height * 0.5 - (cy / cz) * f).astype(np.int64)
    vis = (cz > 0.05) & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)

    # shading
    light = np.array([0.45, 0.8, 0.4], np.float32); light /= np.linalg.norm(light)
    diff = np.clip(nrm @ light, 0.0, 1.0)
    viewd = -(rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9))
    halfv = light[None, :] + viewd; halfv /= np.linalg.norm(halfv, axis=1, keepdims=True) + 1e-9
    spec = np.clip((nrm * halfv).sum(1), 0.0, 1.0) ** 42.0
    warm = np.array([1.0, 0.55, 0.25], np.float32)
    cool = np.array([0.2, 0.45, 0.95], np.float32)
    tc = np.clip(disp / _AMP * 0.9 + 0.5, 0.0, 1.0)[:, None]
    base = cool[None, :] * (1.0 - tc) + warm[None, :] * tc
    col = base * (0.22 + 0.85 * diff[:, None]) + spec[:, None] * np.array([1.0, 0.96, 0.86], np.float32)

    # painter's algorithm: far → near, near overwrites; 2×2 splat fills gaps
    order = np.argsort(-cz)
    order = order[vis[order]]
    bg = np.array([0.02, 0.025, 0.035], np.float32)
    frame = np.tile(bg, (height, width, 1)).astype(np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(sy[order] + dy, 0, height - 1)
            xx = np.clip(sx[order] + dx, 0, width - 1)
            frame[yy, xx] = col[order]

    return post.tonemap(frame, mode="aces", exposure=1.35, preserve_hue=True)


SCENE = Scene(
    name="standing_membrane",
    description="a circular drumhead frozen in a Bessel normal mode J_m(k·r)cos(mθ), ray-marched "
                "as a 3-D height field and coloured by displacement (warm crests, cool troughs) "
                "so the nodal diameters and circles show as still seams. The standing wave "
                "breathes in place over frames.",
    renderer=_render,
)
