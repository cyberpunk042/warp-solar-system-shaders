"""Process 5 — the 30 nm fibre, as a SOLID lit strand (not points).

Chains from Process 4's actual output: it takes the real nucleosome beads ``coil_fibre`` produced and renders
the conserved DNA backbone that threads them — the true coiled 30 nm solenoid — as a tube **mesh**, ray-traced
by the engine (GGX PBR + soft shadow + sky) so it reads as one continuous, opaque chromatin fibre. The tube
follows the actual lib positions; nothing is invented, the matter is only given a surface.
"""

from __future__ import annotations

import numpy as np

from ..genome import coil_fibre
from ..genome.render import render_strand
from ..genome.tube import tube_mesh
from ..scene import Scene

_FB = coil_fibre(sub=2, block=5)


def _smooth(x, w):
    """moving-average smooth an (N,3) path — removes the per-nucleosome disc wobble, leaving the clean
    ~6-beads-per-turn solenoid the beads actually ride."""
    k = np.ones(w) / w
    pad = np.pad(x, ((w, w), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="same")[w:-w] for d in range(3)], 1)


# feature one real 30 nm fibre up close — the iconic coiled solenoid. A fibre is a contiguous run of the
# strand (beads_per_fibre * bp_per_bead pairs) whose backbone coils ~6 nucleosomes per turn about its own
# axis. Take one fibre straight from Process 5's actual output (the real conserved backbone), smooth out the
# per-nucleosome disc wobble to reveal the clean solenoid it rides, centre it; nothing invented, only framed.
_PER = _FB.beads_per_fibre * _FB.bp_per_bead
_seg = (0.5 * (_FB.fib_a + _FB.fib_b))[: _PER]
_seg = _seg - _seg.mean(axis=0)
_center = _smooth(_seg, 55)[::3].astype(np.float64)              # clean solenoid centreline

_t = (np.arange(_center.shape[0]) / max(_center.shape[0] - 1, 1))[:, None]
_COL = (np.array([0.55, 0.44, 0.80], np.float32) * (1.0 - _t)
        + np.array([0.82, 0.70, 0.95], np.float32) * _t)         # violet chromatin, gently graded
_MESH = tube_mesh(_center, radius=0.36, color=_COL.astype(np.float32), sides=16)


def _camera(time: float):
    ang = 0.55 + 0.12 * float(time)                              # slow 3/4 orbit
    r = 12.0
    tgt = np.array([0.0, 0.0, 0.0], np.float32)
    eye = tgt + np.array([r * np.sin(ang), 3.4, r * np.cos(ang)], np.float32)
    return eye, tgt


def _render(width, height, time, mouse, device):
    eye, tgt = _camera(float(time))
    img = render_strand(_MESH, int(width), int(height), eye, tgt,
                        sun_dir=(0.45, 0.72, 0.55), device=device, fov=40.0, exposure=1.12)
    return np.clip(img, 0.0, 1.0)


SCENE = Scene(
    name="warp_fibre",
    description=(
        "Process 5 — the 30 nm fibre as a solid, lit strand. The real conserved DNA backbone that threads "
        "Process 4's nucleosome beads (from coil_fibre) is tubed into a mesh and ray-traced by the engine "
        "(PBR + soft shadow), so the coiled 30 nm solenoid reads as one continuous opaque chromatin fibre — "
        "the actual lib positions, given a surface, not points."
    ),
    renderer=_render,
)
