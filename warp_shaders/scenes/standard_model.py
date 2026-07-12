"""The Standard Model — all seventeen fundamental particles in one chart.

The canonical "periodic table of particles" laid out as glowing tiles: three
generations of **quarks** (violet) and **leptons** (green) in a 4×3 block, the
four gauge **bosons** (orange) in their column, and the **Higgs** (gold). Tile
size ∝ log(mass) — from the near-massless neutrinos to the enormous top quark and
the heavy W/Z/Higgs. See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..scene import Scene

# families
_QK = (0.66, 0.42, 0.9)      # quark — violet
_LP = (0.42, 0.85, 0.55)     # charged lepton — green
_NU = (0.5, 0.82, 0.72)      # neutrino — teal
_BO = (0.98, 0.5, 0.3)       # gauge boson — orange
_HG = (1.0, 0.85, 0.35)      # Higgs — gold

# columns (x) : gen I, gen II, gen III, bosons, Higgs
_CX = [-1.18, -0.72, -0.26, 0.34, 0.98]
# rows (y)
_CY = [0.62, 0.2, -0.22, -0.62]

# (col, row, family_color, mass_MeV)
_P = [
    (0, 0, _QK, 2.2), (1, 0, _QK, 1270.0), (2, 0, _QK, 173000.0),      # u c t
    (0, 1, _QK, 4.7), (1, 1, _QK, 93.0), (2, 1, _QK, 4180.0),          # d s b
    (0, 2, _LP, 0.511), (1, 2, _LP, 105.7), (2, 2, _LP, 1776.9),       # e μ τ
    (0, 3, _NU, 0.001), (1, 3, _NU, 0.001), (2, 3, _NU, 0.001),        # νe νμ ντ
    (3, 0, _BO, 0.001), (3, 1, _BO, 0.001),                           # g  γ  (massless)
    (3, 2, _BO, 91200.0), (3, 3, _BO, 80400.0),                        # Z  W
    (4, 1, _HG, 125000.0),                                             # H (spans gen block)
]


def _radius(mass):
    return 0.05 + 0.017 * max(0.0, math.log10(max(mass, 0.001)) + 3.0)


@wp.kernel
def chart_kernel(img: wp.array2d(dtype=wp.vec3), centers: wp.array(dtype=wp.vec2),
                 colors: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                 n: int, aspect: float, time: float, width: int, height: int):
    i, j = wp.tid()
    x = ((2.0 * (float(j) + 0.5) / float(width)) - 1.0) * aspect
    y = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    p = wp.vec2(x, y)

    col = wp.vec3(0.02, 0.025, 0.05) * (1.0 - 0.35 * y)      # dark chart backdrop
    for k in range(n):
        d = wp.length(p - centers[k])
        r = radii[k]
        pulse = 0.9 + 0.1 * wp.sin(time * 2.0 + float(k))
        core = wp.exp(-(d / r) * (d / r) * 2.4)
        halo = wp.exp(-(d / (r * 2.2)) * (d / (r * 2.2))) * 0.4
        rim = wp.exp(-((d - r) / (r * 0.28)) * ((d - r) / (r * 0.28))) * 0.5
        glow = (core + halo) * pulse
        col = col + colors[k] * glow + wp.vec3(1.0, 1.0, 1.0) * (core * core * 0.5)
        col = col + colors[k] * rim
    img[i, j] = col


def _render(width, height, time, mouse, device):
    aspect = width / height
    cs = np.zeros((len(_P), 2), np.float32)
    cols = np.zeros((len(_P), 3), np.float32)
    rs = np.zeros(len(_P), np.float32)
    for k, (c, r, fam, mass) in enumerate(_P):
        cs[k] = (_CX[c], _CY[r])
        cols[k] = fam
        rs[k] = _radius(mass)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(chart_kernel, dim=(height, width),
              inputs=[img, wp.array(cs, dtype=wp.vec2, device=device),
                      wp.array(cols, dtype=wp.vec3, device=device),
                      wp.array(rs, dtype=wp.float32, device=device), int(len(_P)),
                      float(aspect), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    from ..subatomic import render as _r
    hdr = img.numpy().astype(np.float32)
    return _r.finish(hdr, width, height, threshold=1.2, strength=0.4, exposure=1.05)


SCENE = Scene(
    name="standard_model",
    description="The Standard Model — all 17 fundamental particles in the canonical "
                "chart: three generations of quarks (violet) + leptons (green), the "
                "four gauge bosons (orange) and the Higgs (gold); tile size ∝ log mass.",
    renderer=_render,
)
