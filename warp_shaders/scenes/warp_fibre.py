"""Process 5 — the 30 nm fibre: the nucleosome beads-on-a-string COIL into the solenoid fibre.

Chains from Process 4's actual output. ``coil_fibre`` supplies both real end states of one fibre: the
beads-on-a-string (``bead_a``, as Process 4 left them) and the coiled 30 nm solenoid (``fib_a``). This scene
animates the real **transition** between them — every base pair moves continuously from its string position to
its solenoid position (a straight interpolation of the two real lib states, the same motion the process makes)
— and renders it SOLID, tubed and ray-traced by the engine on a dark specimen background. Not a spin of a
finished object: the string visibly coils up into the fibre.
"""

from __future__ import annotations

import numpy as np

from ..genome import coil_fibre
from ..genome.render import render_strand
from ..genome.tube import tube_mesh
from ..scene import Scene

_FB = coil_fibre(sub=2, block=5)
_PER = _FB.beads_per_fibre * _FB.bp_per_bead


def _smooth(x, w):
    k = np.ones(w) / w
    pad = np.pad(x, ((w, w), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="same")[w:-w] for d in range(3)], 1)


# one real fibre, both end states from Process 5 (same pairs, same frame — do NOT re-centre separately). Split
# the coiled solenoid into its AXIS (heavy smooth) + the COIL offset, so the transition can wind the coil up
# naturally instead of collapsing it in a straight line.
_bead = (0.5 * (_FB.bead_a + _FB.bead_b))[:_PER]
_fib = (0.5 * (_FB.fib_a + _FB.fib_b))[:_PER]
_off = _fib.mean(axis=0)
_S = 3
_BEAD = _smooth(_bead - _off, 110)[::_S].astype(np.float64)      # beads-on-a-string (extended)
_SOL = _smooth(_fib - _off, 90)[::_S].astype(np.float64)         # the coiled solenoid
_AXIS = _smooth(_fib - _off, 900)[::_S].astype(np.float64)       # its central axis (coil smoothed away)
_COILR = _SOL - _AXIS                                            # the radial coil offset (grows in as it winds)

_t = (np.arange(_BEAD.shape[0]) / max(_BEAD.shape[0] - 1, 1))[:, None]
_COL = (np.array([0.55, 0.44, 0.80], np.float32) * (1.0 - _t)
        + np.array([0.82, 0.70, 0.95], np.float32) * _t).astype(np.float32)

_DUR = 4.0                                                       # transition duration (seconds)


def _ss(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _render(width, height, time, mouse, device):
    t = float(time)
    # the axis pulls in from the extended string to the compact fibre axis, THEN the coil winds up on it — a
    # natural two-phase coiling (every frame a valid partially-coiled solenoid), not a straight-line morph.
    wa = _ss((t - 0.3) / (_DUR * 0.55))                          # axis gather
    wc = _ss((t - 0.3 - _DUR * 0.35) / (_DUR * 0.6))             # coil wind-up (lags the gather)
    center = (1.0 - wa) * _BEAD + wa * _AXIS + wc * _COILR
    mesh = tube_mesh(center, radius=0.36, color=_COL, sides=14)
    eye = np.array([7.5, 3.4, 11.0], np.float32)                 # fixed 3/4 view; the coiling is the motion
    tgt = np.array([0.0, 0.0, 0.0], np.float32)
    img = render_strand(mesh, int(width), int(height), eye, tgt,
                        sun_dir=(0.45, 0.72, 0.55), device=device, fov=42.0, exposure=1.12)
    return np.clip(img, 0.0, 1.0)


SCENE = Scene(
    name="warp_fibre",
    description=(
        "Process 5 — the 30 nm fibre. The nucleosome beads-on-a-string from Process 4 coil into the 30 nm "
        "solenoid: this scene animates the real transition between coil_fibre's two end states (bead_a -> "
        "fib_a), every base pair moving continuously from its string position to its solenoid position, "
        "tubed and ray-traced solid on a dark specimen background — the string visibly coiling into the fibre."
    ),
    renderer=_render,
)
