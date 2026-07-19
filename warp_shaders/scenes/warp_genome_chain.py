"""The whole genome ladder as ONE continuous scene — the RTX board compressing down to a chromosome.

This is the offline ``build_genome_chain.py`` chain expressed as a normal :class:`Scene`, so the whole
take can be rendered (``render.py --scene warp_genome_chain --device auto``) and watched live
(``watch.py --scene warp_genome_chain``) like any other scene, on GPU when one is present.

It plays each dedicated genome stage's own timeline back-to-back and **cross-dissolves** every seam, so
stage i's settled body melts into stage i+1 re-forming from it (which looks like stage i) — one shape
retransforming, no cuts. It opens on ``gpu_board`` — the *exact* board ``tokenize_card`` voxel-samples
(``board_map`` is the tokenization source) — posed to the token field's angle, so the first dissolve is
that real board becoming its own tokens in place. It ends on ``warp_chromosome_x``, the metaphase X.

**One continuous camera.** The seams used to jump because every stage framed its own content with its own
camera (base pairs at distance 6, helices at 52, …) — so at a seam the camera teleported and the take read
as jerky cuts, not a sequence. Here the chain drives the camera: each stage keeps its native framing during
its own core, but across every 0.8 s seam the camera's **distance and view direction interpolate smoothly**,
and *both* the outgoing and incoming stage are rendered through that same shared framing (each still aimed
at its own content's centre). Same matter, one camera path — the outgoing body and the incoming one sit at
the same place and size on screen, so one dissolves into the other in place instead of popping.

No cheating: the same matter flows the whole way (each genome library preserves the pair ordering), each
stage is the physics-checked output of its own library, nothing is spawned or teleported.
"""

from __future__ import annotations

import numpy as np

from ..scene import Scene, get_scene
from . import (
    warp_basepair,
    warp_chromosome_x,
    warp_fibre,
    warp_helix,
    warp_nucleosome,
    warp_scan_merge,
    warp_telomere,
    warp_tokenize,
)

# (scene, scene_t0, scene_t1, play_secs, mouse) — play_secs is how long the stage occupies the global
# timeline; scene time is mapped 0..play_secs -> scene_t0..scene_t1 (each stage forms then settles).
_GB_POSE = (48.0, -88.0)   # gpu_board mouse -> az=0.62, el=0.34, matching warp_tokenize's camera
_SEG = [
    ("warp_scan_merge", 0.0, 1.0, 3.60, _GB_POSE),   # 0 naked card -> a scan sweeps it, colouring every
                                                     #   element by type: the card BECOMING tokens
    ("warp_tokenize",   0.0, 5.2, 3.29, (0.0, 0.0)), # 1 the coloured card lifts off into floating tokens
    ("warp_basepair",   0.0, 5.2, 3.29, (0.0, 0.0)), # 2 tokens -> base pairs
    ("warp_helix",      0.0, 5.6, 3.43, (0.0, 0.0)), # 3 -> double helices
    ("warp_nucleosome", 0.0, 5.6, 3.43, (0.0, 0.0)), # 4 -> nucleosome beads
    ("warp_fibre",      0.0, 5.6, 3.43, (0.0, 0.0)), # 5 -> 30 nm fibre
    ("warp_telomere",   0.0, 5.6, 3.43, (0.0, 0.0)), # 6 -> telomere-capped strand
    ("warp_chromosome_x", 0.0, 7.2, 4.43, (0.0, 0.0)), # 7 -> metaphase X chromosome (held)
]
_DISSOLVE = 0.8                                        # seconds of cross-dissolve at each seam

# global start time of each segment (segments overlap by _DISSOLVE)
_START = [0.0]
for _i in range(len(_SEG) - 1):
    _START.append(_START[-1] + _SEG[_i][3] - _DISSOLVE)
TOTAL = _START[-1] + _SEG[-1][3]                       # total duration of the take (seconds)

# the splat stages (everything after the ray-marched gpu_board) share one projection and accept a camera
# override, so the chain can drive them all with one continuous camera.
_SPLAT = {
    "warp_tokenize": warp_tokenize,
    "warp_basepair": warp_basepair,
    "warp_helix": warp_helix,
    "warp_nucleosome": warp_nucleosome,
    "warp_fibre": warp_fibre,
    "warp_telomere": warp_telomere,
    "warp_chromosome_x": warp_chromosome_x,
}
# stages whose _camera() returns no dist (tokenize, basepair) — supply the fixed distance they use.
_DIST_DEFAULT = {"warp_tokenize": 7.6, "warp_basepair": 6.0}
_UP = np.array([0.0, 1.0, 0.0], np.float32)
_GB = None   # gpu_board scene, resolved lazily (avoid get_scene() at import/registry-discovery time)


def _seg_scene_time(i: int, gt: float) -> float:
    name, st0, st1, dur, _pose = _SEG[i]
    frac = min(max((gt - _START[i]) / dur, 0.0), 1.0)
    return st0 + (st1 - st0) * frac


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _native_cam(name: str, st: float):
    """This splat stage's native framing at scene-time ``st``, as (target, dist, direction).

    Reconstructed from the stage's own ``_camera``: ``ww`` is the unit forward, so the look target is
    ``ro + ww*dist`` and the view direction (target -> eye) is ``-ww``. This is the exact framing the
    stage would use standalone — the chain reuses it and only blends it across seams."""
    out = _SPLAT[name]._camera(float(st))
    ro = np.asarray(out[0], np.float32)
    ww = _norm(np.asarray(out[3], np.float32))
    dist = float(out[4]) if len(out) >= 5 else _DIST_DEFAULT[name]
    target = (ro + ww * dist).astype(np.float32)
    direction = (-ww).astype(np.float32)
    return target, dist, direction


def _build_cam(target, dist, direction):
    """A camera override tuple (ro, uu, vv, ww, dist) framing ``target`` from ``dist`` along ``direction``,
    with the same up-vector convention every splat stage uses (so it matches native framing exactly)."""
    d = _norm(np.asarray(direction, np.float32))
    ro = (np.asarray(target, np.float32) + dist * d).astype(np.float32)
    ww = _norm(np.asarray(target, np.float32) - ro)
    uu = _norm(np.cross(ww, _UP))
    vv = np.cross(uu, ww).astype(np.float32)
    return (ro, uu.astype(np.float32), vv, ww.astype(np.float32), float(dist))


# ---------------------------------------------------------------------------------------------------
# LOCKED camera. The take is watched from a STILL camera — the matter transforms, the camera never
# dollies, spins, or zooms. There is one fixed view direction for the whole take (``_DIR0``), taken from
# warp_tokenize's native 3/4 view so the opening board->tokens seam stays registered. Each stage is shown
# from that direction at the distance that makes its own content fill the frame — stages differ hugely in
# world scale (base pairs span ~6 units, the helix forest ~44), so a single world *distance* can't frame
# them all; using each stage's own fill-distance along the SAME direction keeps the shape centred and the
# same on-screen size, so the camera reads as locked. No per-frame camera motion, no seam interpolation.
# ---------------------------------------------------------------------------------------------------
_DIR0 = _native_cam("warp_tokenize", _SEG[1][2])[2]      # the one fixed view direction (tokenize's 3/4)

_FIX_T = {}   # per-segment fixed look target (its content centre, at settled time)
_FIX_D = {}   # per-segment fixed distance that frames that stage's content to fill, along _DIR0
for _i in range(1, len(_SEG)):
    _tt, _dd, _ = _native_cam(_SEG[_i][0], _SEG[_i][2])
    _FIX_T[_i] = _tt
    _FIX_D[_i] = _dd


def _render_splat(name, idx, w, h, st, device):
    """Render splat stage ``name`` (segment ``idx``) at scene-time ``st`` from the ONE locked camera:
    fixed direction ``_DIR0``, fixed fill-distance, aimed at the stage's content centre. No camera motion."""
    cam = _build_cam(_FIX_T[idx], _FIX_D[idx], _DIR0)
    return np.clip(_SPLAT[name]._render(w, h, st, (0.0, 0.0), device, cam=cam), 0.0, 1.0)


def _render_seg0(w, h, gt, device):
    # Stage 0 = the tokenization itself: the naked card, then a scan sweeps across it colouring every
    # element by its merge-codec type (identical pieces share a colour) — the card literally becoming
    # tokens. Rendered from the SAME locked camera as the token cloud, in the raw board_map frame, so the
    # coloured card and the tokens it lifts into line up exactly. Only the scan runs (no merge/gather).
    ph = _seg_scene_time(0, gt)                              # 0..1 across this stage
    sweep = _smoothstep((ph - 0.12) / 0.80)                  # hold on the bare card, then sweep the scan
    bx = warp_scan_merge._BB[1]
    front = -bx + (2.0 * bx + 0.6) * sweep                  # scan wavefront x-position across the card
    cam = _build_cam(_FIX_T[1], _FIX_D[1], _DIR0)
    state = (float(front), 0.0, 0.0, 0.0)                    # scan + classify only — no merge, no erode
    return np.clip(warp_scan_merge._render(w, h, ph, (0.0, 0.0), device, cam=cam, state=state), 0.0, 1.0)


def _render(width, height, time, mouse, device):
    w, h = int(width), int(height)
    gt = float(time) % TOTAL if TOTAL > 0 else 0.0     # loop the take

    # primary segment = last one that has started
    prim = 0
    for i in range(len(_SEG)):
        if gt >= _START[i]:
            prim = i
    nxt = prim + 1
    in_diss = nxt < len(_SEG) and gt >= _START[nxt]
    f = _smoothstep((gt - _START[nxt]) / _DISSOLVE) if in_diss else 0.0

    # --- seam 0->1 is special: gpu_board is ray-marched (its own camera model), posed to match tokenize's
    # native 3/4 view (which _DIR0 equals), so the opening board->tokens seam stays registered. ---
    if prim == 0:
        a = _render_seg0(w, h, gt, device)
        if in_diss:
            b = _render_splat("warp_tokenize", 1, w, h, _seg_scene_time(1, gt), device)
            return np.clip((1.0 - f) * a + f * b, 0.0, 1.0)
        return a

    # --- splat stages 1..7: ONE locked camera. Each stage renders from the same fixed view; a seam simply
    # cross-fades the outgoing settled body into the incoming one re-forming, both centred and same-size. ---
    a = _render_splat(_SEG[prim][0], prim, w, h, _seg_scene_time(prim, gt), device)
    if not in_diss:
        return a
    b = _render_splat(_SEG[nxt][0], nxt, w, h, _seg_scene_time(nxt, gt), device)
    return np.clip((1.0 - f) * a + f * b, 0.0, 1.0)


SCENE = Scene(
    name="warp_genome_chain",
    description=(
        f"The whole genome compression in one continuous {TOTAL:.0f}s take on one driven camera: the real "
        "gpu_board (the exact board tokenize_card samples) -> its tokens -> base pairs -> double-helix forest "
        "-> nucleosome beads -> 30 nm fibre -> telomere -> metaphase X chromosome, every seam a cross-dissolve "
        "with the camera's distance and direction interpolated across it so it reads as one shape "
        "retransforming, not cuts. Same matter throughout, each stage its library's real output, no jumps."
    ),
    renderer=_render,
)
