"""SUPER-CHROMOSOME — the recursion, made visual. Act II of the fold.

Act I (``warp_genome_thread``) folds one card into one chromosome. This is what happens *next*: give the
chromosome a type — **X** or **Y** — and let an X and a Y **merge into a new base-pair strand** that
re-enters the SAME fold as a **super-chromosome**. The whole cluster re-transforms, one level up.

The story, one continuous take:

  1. **X + Y**            — a big X chromosome and a small Y chromosome, side by side (two folded strands).
  2. **base-pair merge**  — they converge and ZIP: strand-X[i] pairs with strand-Y[i] → one base-pair ladder
                            (the two chromosomes literally become the two backbones of the next-level strand).
  3. **refold**           — that ladder winds through helix → nucleosome → fibre → chromatid via the exact
                            same engine fold, scaled up: a SUPER-chromosome (amber banding to mark the level).
  4. **super-chromosome** — held; a faint smaller X+Y ghosts in beside it — it is itself pairable again. The
                            transform recurses over the tree, relative to size and depth.

Reuses Act I's renderer (``warp_genome_thread._particles`` / ``_cam``) and the fold (``thread.frame``): the
merge is the base-pair stage read at the chromosome scale. View with play.py (``--scene warp_genome_super``).
"""
from __future__ import annotations

import math

import numpy as np

from ..genome import thread as TH
from . import warp_genome_thread as GT
from ..scene import Scene

_TH = GT._TH


def _mir(p):
    q = np.asarray(p, np.float32).copy()
    q[:, 0] = -q[:, 0]
    return q


def _ss(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _tint(col, rgb, amt):
    return (col * (1.0 - amt) + np.array(rgb, np.float32)[None] * amt).astype(np.float32)


# --- precomputed shapes (all centred; the camera auto-frames, so only relative layout matters) -----------
_posC, _colC = TH.frame(_TH, 1.0)                          # the metaphase chromatid (purple), Act I's finale
_posC = (_posC - _posC.mean(0)).astype(np.float32)
_baseX = np.concatenate([_posC, _mir(_posC)], 0).astype(np.float32)   # 2N: chromatid + sister = the X shape
_colX = np.concatenate([_colC, _colC], 0).astype(np.float32)
_colY = _tint(_colX, [0.20, 0.85, 0.80], 0.65)             # the Y tinted teal, to read as a distinct chromosome

_posL = (_TH.ladder - _TH.ladder.mean(0)).astype(np.float32)
_ladder = (np.concatenate([_posL, _mir(_posL)], 0) * 1.30).astype(np.float32)   # the shared SUPER base pairs
_colBase = np.concatenate([_TH.col_base, _TH.col_base], 0).astype(np.float32)

_TEAL = [0.20, 0.85, 0.80]
_AMBER = [0.98, 0.74, 0.36]

# beat durations (seconds): two bodies / zip-merge / refold / hold-and-recurse
_tA, _tB, _tC, _tD = 3.0, 3.4, 5.2, 3.0
TOTAL = _tA + _tB + _tC + _tD
_SEG = [("X + Y",), ("base-pair merge",), ("refold",), ("super-chromosome",)]
_START = [0.0, _tA, _tA + _tB, _tA + _tB + _tC]


def _two_bodies(sep, colX, colY):
    """The X (big, left) and Y (small, right) chromosomes as two separate folded bodies."""
    X = _baseX * 0.62; X = X.copy(); X[:, 0] -= sep
    Y = _baseX * 0.40; Y = Y.copy(); Y[:, 0] += sep
    return (np.concatenate([X, Y], 0).astype(np.float32),
            np.concatenate([colX, colY], 0).astype(np.float32))


def _compose(time):
    """Return (pos, col, az, el) for the whole take at `time`."""
    t = float(time) % TOTAL

    # 1) X + Y held, a slow turn
    if t < _tA:
        f = _ss(t / _tA)
        pos, col = _two_bodies(1.05, _colX, _colY)
        return pos, col, 0.15 + 0.30 * f, 0.42

    # 2) base-pair merge: the two bodies converge and morph onto the ONE shared base-pair ladder
    if t < _tA + _tB:
        m = _ss((t - _tA) / _tB)
        X0, _ = _two_bodies(1.05 * (1.0 - m), _colX, _colY)          # halves slide toward centre
        n = _baseX.shape[0]
        target = np.concatenate([_ladder, _ladder], 0)               # both halves land on the ladder (zip)
        pos = (X0 * (1.0 - m) + target * m).astype(np.float32)
        colX = _tint(_colX, _TEAL, 0.0) * (1.0 - m) + _colBase * m
        colY = _colY * (1.0 - m) + _colBase * m
        col = np.concatenate([colX, colY], 0).astype(np.float32)
        return pos, col, 0.45 + 0.25 * m, 0.42 - 0.16 * m

    # 3) refold: the merged ladder winds up through the engine fold, scaled up -> the super-chromosome
    if t < _tA + _tB + _tC:
        f = _ss((t - _tA - _tB) / _tC)
        prog = 0.40 + 0.60 * f                                       # pair -> helix -> ... -> chromatid
        fp, fc = TH.frame(_TH, prog)
        fp = (fp - fp.mean(0)).astype(np.float32)
        pos = (np.concatenate([fp, _mir(fp)], 0) * 1.30).astype(np.float32)
        col = _tint(np.concatenate([fc, fc], 0), _AMBER, 0.55 * f)   # ripens to amber = the new level
        # duplicate to keep the same 2N-of-both-halves density as the merge (seamless hand-off)
        pos = np.concatenate([pos, pos], 0); col = np.concatenate([col, col], 0)
        return pos, col, 0.70 - 0.60 * f, 0.26 + 0.24 * f

    # 4) super-chromosome held; a faint smaller X+Y ghosts in beside it -> it can pair again (recursion)
    f = _ss((t - _tA - _tB - _tC) / _tD)
    fp, fc = TH.frame(_TH, 1.0)
    fp = (fp - fp.mean(0)).astype(np.float32)
    sup = (np.concatenate([fp, _mir(fp)], 0) * 1.30).astype(np.float32)
    sup = sup.copy(); sup[:, 0] -= 0.55 * f                          # drift left to make room for the ghost
    scol = _tint(np.concatenate([fc, fc], 0), _AMBER, 0.55)
    gx, gcolx = _baseX * 0.34, _tint(_colX, _AMBER, 0.3)
    gy, gcoly = _baseX * 0.22, _colY
    gx = gx.copy(); gx[:, 0] += 1.15; gy = gy.copy(); gy[:, 0] += 1.75
    ghost_fade = _ss(max(0.0, (f - 0.35) / 0.65))                    # fade the next generation in late
    gcol = np.concatenate([gcolx, gcoly], 0) * (0.20 + 0.30 * ghost_fade)
    pos = np.concatenate([sup, gx, gy], 0).astype(np.float32)
    col = np.concatenate([scol, gcol], 0).astype(np.float32)
    return pos, col, 0.10, 0.50


def _core(W, H, time, mx, my, zoomf, device):
    pos, col, az, el = _compose(time)
    cam = GT._cam(pos, az, el, mx, my, zoomf)
    return GT._particles(int(W), int(H), pos, col, cam, device)


def render_view(width, height, time, mx, my, zoomf, device):
    return _core(width, height, time, mx, my, zoomf, device)


def _render(width, height, time, mouse, device):
    return _core(width, height, time, float(mouse[0]), float(mouse[1]), 1.0, device)


SCENE = Scene(
    name="warp_genome_super",
    description="The recursion, visual: an X and a Y chromosome merge into a base-pair strand and refold into "
                "a super-chromosome, which is itself pairable again — the fold read one level up.",
    renderer=_render,
)
