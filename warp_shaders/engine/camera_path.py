"""Keyframed camera paths — the motion half of the cinematics layer (host).

A :class:`CameraPath` is a list of :class:`Keyframe` s (a normalized time in
``[0, 1]`` plus an eye, a look-at target and a vertical FOV). ``sample(t)``
returns the interpolated ``(eye, target, fov)``; ``camera(t, aspect)`` builds the
engine :class:`~warp_shaders.engine.uniforms.Camera` you feed the kernel.

The **eye** is interpolated with a Catmull-Rom spline through the keyframe
positions, so a handful of stops become a smooth curved move (a straight lerp
would give robotic corners). **Target** and **FOV** ease between the bracketing
keyframes. Segment progress is shaped by a named **easing** (``linear`` /
``smoothstep`` / ``smoother`` / ``ease_in`` / ``ease_out`` / ``ease_in_out``) so a
dolly can accelerate and settle.

Convenience builders cover the common shots — :func:`orbit` (circle the subject),
:func:`dolly` (push in / pull out), :func:`fly` (an arbitrary keyframe list).
All maths is NumPy; nothing here touches the GPU.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


# --------------------------------------------------------------------------- #
# easing                                                                      #
# --------------------------------------------------------------------------- #

def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def linear(t):        return _clamp01(t)
def smoothstep(t):    t = _clamp01(t); return t * t * (3.0 - 2.0 * t)
def smoother(t):      t = _clamp01(t); return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
def ease_in(t):       t = _clamp01(t); return t * t
def ease_out(t):      t = _clamp01(t); return 1.0 - (1.0 - t) * (1.0 - t)
def ease_in_out(t):   return smoothstep(t)


EASINGS = {
    "linear": linear, "smoothstep": smoothstep, "smoother": smoother,
    "ease_in": ease_in, "ease_out": ease_out, "ease_in_out": ease_in_out,
}


def ease(name, t):
    """Apply a named easing to `t` (unknown name -> linear)."""
    return EASINGS.get(name, linear)(t)


# --------------------------------------------------------------------------- #
# keyframes + path                                                            #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Keyframe:
    t: float                       # normalized time in [0, 1]
    eye: tuple                     # camera position (x, y, z)
    target: tuple = (0.0, 0.0, 0.0)
    fov: float = 45.0              # vertical field of view, degrees


def _catmull_rom(p0, p1, p2, p3, u):
    """Centripetal-ish Catmull-Rom point at local parameter u in [0,1]."""
    u2 = u * u
    u3 = u2 * u
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3)


class CameraPath:
    """A timed sequence of camera keyframes with smooth interpolation."""

    def __init__(self, keyframes=None, easing="ease_in_out"):
        self.keys = sorted(keyframes or [], key=lambda k: k.t)
        self.easing = easing

    def add(self, t, eye, target=(0.0, 0.0, 0.0), fov=45.0):
        self.keys.append(Keyframe(t, eye, target, fov))
        self.keys.sort(key=lambda k: k.t)
        return self

    def sample(self, t):
        """Return ``(eye, target, fov)`` at global time `t` in ``[0, 1]``."""
        if not self.keys:
            raise ValueError("CameraPath has no keyframes")
        if len(self.keys) == 1:
            k = self.keys[0]
            return np.asarray(k.eye, float), np.asarray(k.target, float), float(k.fov)
        t = _clamp01(t)
        # bracketing segment [i, i+1]
        n = len(self.keys)
        i = 0
        while i < n - 1 and t > self.keys[i + 1].t:
            i += 1
        a, b = self.keys[i], self.keys[min(i + 1, n - 1)]
        span = max(b.t - a.t, 1e-9)
        u = ease(self.easing, (t - a.t) / span)
        # Catmull-Rom on the eye through neighbouring keyframes
        p0 = np.asarray(self.keys[max(i - 1, 0)].eye, float)
        p1 = np.asarray(a.eye, float)
        p2 = np.asarray(b.eye, float)
        p3 = np.asarray(self.keys[min(i + 2, n - 1)].eye, float)
        eye = _catmull_rom(p0, p1, p2, p3, u)
        target = (1.0 - u) * np.asarray(a.target, float) + u * np.asarray(b.target, float)
        fov = (1.0 - u) * a.fov + u * b.fov
        return eye, target, float(fov)

    def camera(self, t, aspect=1.0):
        """Build an engine :class:`Camera` at time `t`."""
        from .uniforms import make_camera
        eye, target, fov = self.sample(t)
        return make_camera(tuple(eye), tuple(target), fov_deg=fov, aspect=aspect)


# --------------------------------------------------------------------------- #
# shot builders                                                               #
# --------------------------------------------------------------------------- #

def orbit(center=(0.0, 0.0, 0.0), radius=5.0, elevation=0.3, turns=1.0,
          start_az=0.0, fov=45.0, samples=12, easing="linear") -> CameraPath:
    """Circle `center` at `radius`, looking inward — `turns` revolutions."""
    cx, cy, cz = center
    keys = []
    for k in range(samples + 1):
        f = k / samples
        az = start_az + turns * 2.0 * math.pi * f
        eye = (cx + radius * math.cos(elevation) * math.sin(az),
               cy + radius * math.sin(elevation),
               cz + radius * math.cos(elevation) * math.cos(az))
        keys.append(Keyframe(f, eye, center, fov))
    return CameraPath(keys, easing=easing)


def dolly(eye0, eye1, target=(0.0, 0.0, 0.0), fov0=45.0, fov1=None,
          easing="ease_in_out") -> CameraPath:
    """Straight push-in / pull-out from `eye0` to `eye1` (optional FOV move)."""
    fov1 = fov0 if fov1 is None else fov1
    return CameraPath([Keyframe(0.0, eye0, target, fov0),
                       Keyframe(1.0, eye1, target, fov1)], easing=easing)


def fly(keyframes, easing="ease_in_out") -> CameraPath:
    """A path from ``(t, eye[, target, fov])`` tuples."""
    keys = []
    for row in keyframes:
        t, eye = row[0], row[1]
        target = row[2] if len(row) > 2 else (0.0, 0.0, 0.0)
        fov = row[3] if len(row) > 3 else 45.0
        keys.append(Keyframe(t, eye, target, fov))
    return CameraPath(keys, easing=easing)
