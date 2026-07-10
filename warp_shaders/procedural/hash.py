"""GPU hash functions (portable CPU+CUDA).

Compact `fract(sin(dot()))` family — chosen for portability across Warp's CPU and
CUDA backends over bit-integer hashes. Ref: Morgan McGuire, "Hash Functions for
GPU Rendering" (JCGT 2020). All return values in [0, 1); the *NN variants return
per-component hashes suitable for gradients (map to [-1,1] via 2x-1).
"""

import warp as wp


@wp.func
def fract(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def hash11(p: float) -> float:
    return fract(wp.sin(p * 127.1) * 43758.5453)


@wp.func
def hash21(p: wp.vec2) -> float:
    return fract(wp.sin(wp.dot(p, wp.vec2(127.1, 311.7))) * 43758.5453)


@wp.func
def hash31(p: wp.vec3) -> float:
    return fract(wp.sin(wp.dot(p, wp.vec3(127.1, 311.7, 74.7))) * 43758.5453)


@wp.func
def hash22(p: wp.vec2) -> wp.vec2:
    x = wp.dot(p, wp.vec2(127.1, 311.7))
    y = wp.dot(p, wp.vec2(269.5, 183.3))
    return wp.vec2(fract(wp.sin(x) * 43758.5453), fract(wp.sin(y) * 43758.5453))


@wp.func
def hash33(p: wp.vec3) -> wp.vec3:
    x = wp.dot(p, wp.vec3(127.1, 311.7, 74.7))
    y = wp.dot(p, wp.vec3(269.5, 183.3, 246.1))
    z = wp.dot(p, wp.vec3(113.5, 271.9, 124.6))
    return wp.vec3(fract(wp.sin(x) * 43758.5453),
                   fract(wp.sin(y) * 43758.5453),
                   fract(wp.sin(z) * 43758.5453))
