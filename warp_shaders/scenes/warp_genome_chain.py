"""The whole genome ladder as ONE continuous scene — the RTX board compressing down to a chromosome.

This is the offline ``build_genome_chain.py`` chain expressed as a normal :class:`Scene`, so the whole
take can be rendered (``render.py --scene warp_genome_chain --device auto``) and watched live
(``watch.py --scene warp_genome_chain``) like any other scene, on GPU when one is present.

It plays each dedicated genome stage's own timeline back-to-back and **cross-dissolves** every seam, so
stage i's settled body melts into stage i+1 re-forming from it (which looks like stage i) — one shape
retransforming, no cuts. It opens on ``gpu_board`` — the *exact* board ``tokenize_card`` voxel-samples
(``board_map`` is the tokenization source) — posed to the token field's angle, so the first dissolve is
that real board becoming its own tokens in place.

No cheating: the same matter flows the whole way (each genome library preserves the pair ordering), each
stage is the physics-checked output of its own library, nothing is spawned or teleported.
"""

from __future__ import annotations

import numpy as np

from ..scene import Scene, get_scene

# (scene, scene_t0, scene_t1, play_secs, mouse) — play_secs is how long the stage occupies the global
# timeline; scene time is mapped 0..play_secs -> scene_t0..scene_t1 (each stage forms then settles).
_GB_POSE = (48.0, -88.0)   # gpu_board mouse -> az=0.62, el=0.34, matching warp_tokenize's camera
_SEG = [
    ("gpu_board",       0.0, 3.0, 2.43, _GB_POSE),   # 0 the real source board (posed to the token angle)
    ("warp_tokenize",   0.0, 5.2, 3.29, (0.0, 0.0)), # 1 that same board -> its tokens
    ("warp_basepair",   0.0, 5.2, 3.29, (0.0, 0.0)), # 2 tokens -> base pairs
    ("warp_helix",      0.0, 5.6, 3.43, (0.0, 0.0)), # 3 -> double helices
    ("warp_nucleosome", 0.0, 5.6, 3.43, (0.0, 0.0)), # 4 -> nucleosome beads
    ("warp_fibre",      0.0, 5.6, 3.43, (0.0, 0.0)), # 5 -> 30 nm fibre
    ("warp_telomere",   0.0, 5.6, 3.43, (0.0, 0.0)), # 6 -> telomere-capped strand
    ("warp_chromosome", 0.0, 7.2, 4.43, (0.0, 0.0)), # 7 -> merged chromosome (held)
]
_DISSOLVE = 0.8                                        # seconds of cross-dissolve at each seam

# global start time of each segment (segments overlap by _DISSOLVE)
_START = [0.0]
for _i in range(len(_SEG) - 1):
    _START.append(_START[-1] + _SEG[_i][3] - _DISSOLVE)
TOTAL = _START[-1] + _SEG[-1][3]                       # total duration of the take (seconds)


def _seg_scene_time(i: int, gt: float) -> float:
    name, st0, st1, dur, _pose = _SEG[i]
    frac = min(max((gt - _START[i]) / dur, 0.0), 1.0)
    return st0 + (st1 - st0) * frac


def _render_seg(i: int, w: int, h: int, gt: float, device):
    name, st0, st1, dur, pose = _SEG[i]
    return get_scene(name).render(w, h, _seg_scene_time(i, gt), pose, device)


def _render(width, height, time, mouse, device):
    w, h = int(width), int(height)
    gt = float(time) % TOTAL if TOTAL > 0 else 0.0     # loop the take

    # primary segment = last one that has started
    prim = 0
    for i in range(len(_SEG)):
        if gt >= _START[i]:
            prim = i
    a = _render_seg(prim, w, h, gt, device)

    # in a dissolve window with the next segment? blend.
    nxt = prim + 1
    if nxt < len(_SEG) and gt >= _START[nxt]:
        b = _render_seg(nxt, w, h, gt, device)
        f = min(max((gt - _START[nxt]) / _DISSOLVE, 0.0), 1.0)
        f = f * f * (3.0 - 2.0 * f)                    # smoothstep
        return np.clip((1.0 - f) * a + f * b, 0.0, 1.0)
    return np.clip(a, 0.0, 1.0)


SCENE = Scene(
    name="warp_genome_chain",
    description=(
        f"The whole genome compression in one continuous {TOTAL:.0f}s take: the real gpu_board (the exact "
        "board tokenize_card samples) -> its tokens -> base pairs -> double-helix forest -> nucleosome "
        "beads -> 30 nm fibre -> telomere -> merged chromosome, every seam a cross-dissolve so it reads as "
        "one shape retransforming. Same matter throughout, each stage its library's real output, no cuts."
    ),
    renderer=_render,
)
