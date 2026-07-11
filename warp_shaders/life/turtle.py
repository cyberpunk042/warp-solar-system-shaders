"""3D turtle interpreter — turns an L-System word into geometry.

Walks the module string carrying an orthonormal **H / L / U** frame (heading /
left / up) and a pen state (radius, colour), emitting tapered branch *segments*
and *leaves*. This is the classic bracketed-turtle interpretation (ABOP §1.6),
in 3D (ABOP §1.6.2 — the ``& ^ \\ /`` pitch/roll commands).

Commands (a trailing ``(x)`` overrides the default where noted):

====  ==========================================================
sym   action
====  ==========================================================
F     move forward ``x`` (or ``step``) drawing a segment
f     move forward without drawing
+ -   yaw left / right around U by ``x`` (or ``angle``)
& ^   pitch down / up around L
\\ /   roll left / right around H
\|     yaw 180°
[ ]   push / pop the full turtle state
!     set the branch radius to ``x``
'     set the colour to palette index ``x``
L     emit a leaf at the current position + orientation
====  ==========================================================

Any other symbol (non-terminals left in the word) draws nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .lsystem import Module

Vec3 = np.ndarray

# default colour palette: 0 = bark, 1..2 = greens, 3 = flower
_PALETTE = np.array([
    [0.42, 0.28, 0.16],   # 0 bark
    [0.20, 0.52, 0.16],   # 1 leaf green
    [0.38, 0.68, 0.24],   # 2 light green
    [0.90, 0.35, 0.55],   # 3 flower
    [0.75, 0.68, 0.30],   # 4 dry / straw
], dtype=np.float32)


@dataclass
class TurtleConfig:
    step: float = 1.0          # default F length
    angle: float = 25.0        # default rotation (degrees)
    radius: float = 0.08       # starting branch radius
    leaf_size: float = 0.5     # leaf blade length
    palette: np.ndarray = field(default_factory=lambda: _PALETTE.copy())


@dataclass
class Segment:
    p0: Vec3
    p1: Vec3
    r0: float
    r1: float
    color: Vec3


@dataclass
class Leaf:
    pos: Vec3
    h: Vec3        # blade forward
    u: Vec3        # blade up (normal-ish)
    size: float
    color: Vec3


@dataclass
class Geometry:
    segments: List[Segment] = field(default_factory=list)
    leaves: List[Leaf] = field(default_factory=list)

    def bounds(self) -> Tuple[Vec3, Vec3]:
        pts = []
        for s in self.segments:
            pts.append(s.p0); pts.append(s.p1)
        for lf in self.leaves:
            pts.append(lf.pos)
        if not pts:
            return np.zeros(3, np.float32), np.zeros(3, np.float32)
        a = np.array(pts, np.float32)
        return a.min(0), a.max(0)


def _rot(v: Vec3, axis: Vec3, deg: float) -> Vec3:
    """Rodrigues rotation of `v` around unit `axis` by `deg` degrees."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1.0 - c)


class _State:
    __slots__ = ("pos", "H", "L", "U", "r", "color")

    def __init__(self, pos, H, L, U, r, color):
        self.pos = pos.copy(); self.H = H.copy(); self.L = L.copy()
        self.U = U.copy(); self.r = r; self.color = color.copy()


def interpret(word: Sequence[Module], cfg: TurtleConfig = None) -> Geometry:
    """Interpret an L-System word into :class:`Geometry` (segments + leaves)."""
    if cfg is None:
        cfg = TurtleConfig()
    geo = Geometry()
    pos = np.zeros(3, np.float32)
    H = np.array([0.0, 1.0, 0.0], np.float32)     # grow up (+Y)
    L = np.array([-1.0, 0.0, 0.0], np.float32)
    U = np.array([0.0, 0.0, 1.0], np.float32)
    r = float(cfg.radius)
    color = cfg.palette[0].copy()
    stack: List[_State] = []

    for m in word:
        sym = m.sym
        a = m.params[0] if m.params else None
        if sym == "F":
            length = a if a is not None else cfg.step
            p1 = pos + H * length
            geo.segments.append(Segment(pos.copy(), p1.copy(), r, r, color.copy()))
            pos = p1
        elif sym == "f":
            pos = pos + H * (a if a is not None else cfg.step)
        elif sym == "+":
            H = _rot(H, U, a if a is not None else cfg.angle); L = _rot(L, U, a if a is not None else cfg.angle)
        elif sym == "-":
            d = -(a if a is not None else cfg.angle); H = _rot(H, U, d); L = _rot(L, U, d)
        elif sym == "&":
            H = _rot(H, L, a if a is not None else cfg.angle); U = _rot(U, L, a if a is not None else cfg.angle)
        elif sym == "^":
            d = -(a if a is not None else cfg.angle); H = _rot(H, L, d); U = _rot(U, L, d)
        elif sym == "/":
            L = _rot(L, H, a if a is not None else cfg.angle); U = _rot(U, H, a if a is not None else cfg.angle)
        elif sym == "\\":
            d = -(a if a is not None else cfg.angle); L = _rot(L, H, d); U = _rot(U, H, d)
        elif sym == "|":
            H = _rot(H, U, 180.0); L = _rot(L, U, 180.0)
        elif sym == "[":
            stack.append(_State(pos, H, L, U, r, color))
        elif sym == "]":
            if stack:
                st = stack.pop()
                pos, H, L, U, r, color = st.pos, st.H, st.L, st.U, st.r, st.color
        elif sym == "!":
            if a is not None:
                r = a
        elif sym == "'":
            idx = int(a) if a is not None else 1
            color = cfg.palette[idx % len(cfg.palette)].copy()
        elif sym == "L":
            size = a if a is not None else cfg.leaf_size
            geo.leaves.append(Leaf(pos.copy(), H.copy(), U.copy(), size, color.copy()))
        # any other symbol: structural, no geometry
    return geo
