"""Uniform blocks passed to render kernels — the "UBO" pattern (@wp.struct).

Inspired by the-virus-block-mc's `ubo/{camera,light,frame}_ubo.glsl`: instead of
threading a dozen scalars through every kernel, group them into typed structs.
This is the clean-API backbone of the engine — a scene kernel takes
(img, cam, light, frame, qual) and everything it needs is inside.
"""

import math

import warp as wp

from ..lod import QualityTier


@wp.struct
class Camera:
    eye: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_half_fov: float
    aspect: float
    aperture: float      # lens radius; 0 = pinhole (no depth of field)
    focus_dist: float    # distance to the sharp focal plane


@wp.struct
class Light:
    dir: wp.vec3      # unit vector TOWARD the light
    color: wp.vec3
    intensity: float


@wp.struct
class Frame:
    time: float
    width: int
    height: int


@wp.struct
class Quality:
    raymarch_steps: int
    shadow_steps: int
    ao_steps: int
    noise_octaves: int
    volumetric_steps: int
    mip_bias: float


@wp.func
def camera_ray_dir(cam: Camera, u: float, v: float) -> wp.vec3:
    """Ray direction for normalized screen coords u,v in [-1,1] (v up)."""
    d = cam.forward + cam.right * (u * cam.aspect * cam.tan_half_fov) + cam.up * (v * cam.tan_half_fov)
    return wp.normalize(d)


@wp.func
def lens_offset(cam: Camera, s1: float, s2: float) -> wp.vec3:
    """World-space aperture offset from the eye for lens samples s1,s2 in [0,1].

    Uniform sample of the aperture disk (radius ``cam.aperture``) in the camera's
    right/up plane. Zero when ``aperture == 0`` (pinhole). Used with
    :func:`focus_point` for depth of field.
    """
    r = wp.sqrt(s1) * cam.aperture
    a = 6.2831853 * s2
    return cam.right * (r * wp.cos(a)) + cam.up * (r * wp.sin(a))


@wp.func
def focus_point(cam: Camera, u: float, v: float) -> wp.vec3:
    """World point on the focal plane along the pinhole ray for pixel (u,v).

    Objects here stay sharp; a lens ray from ``eye + lens_offset`` aimed at this
    point defocuses everything nearer/farther.
    """
    return cam.eye + camera_ray_dir(cam, u, v) * cam.focus_dist


# ---- host builders ---------------------------------------------------------

def make_camera(eye, target, fov_deg=45.0, aspect=1.0, up=(0.0, 1.0, 0.0),
                aperture=0.0, focus_dist=None) -> Camera:
    import numpy as np
    eye = np.asarray(eye, np.float32)
    tgt = np.asarray(target, np.float32)
    fwd = tgt - eye
    dist = float(np.linalg.norm(fwd))
    fwd /= (dist + 1e-9)
    right = np.cross(fwd, np.asarray(up, np.float32))
    right /= (np.linalg.norm(right) + 1e-9)
    upv = np.cross(right, fwd)
    c = Camera()
    c.eye = wp.vec3(*[float(x) for x in eye])
    c.forward = wp.vec3(*[float(x) for x in fwd])
    c.right = wp.vec3(*[float(x) for x in right])
    c.up = wp.vec3(*[float(x) for x in upv])
    c.tan_half_fov = float(math.tan(math.radians(fov_deg) * 0.5))
    c.aspect = float(aspect)
    c.aperture = float(aperture)
    # default focus on the look-at target (distance eye -> target)
    c.focus_dist = float(focus_dist) if focus_dist is not None else (dist if dist > 1e-6 else 1.0)
    return c


def make_light(direction, color=(1.0, 1.0, 1.0), intensity=1.0) -> Light:
    import numpy as np
    d = np.asarray(direction, np.float32)
    d /= (np.linalg.norm(d) + 1e-9)
    lt = Light()
    lt.dir = wp.vec3(*[float(x) for x in d])
    lt.color = wp.vec3(*[float(x) for x in color])
    lt.intensity = float(intensity)
    return lt


def make_frame(time, width, height) -> Frame:
    fr = Frame()
    fr.time = float(time)
    fr.width = int(width)
    fr.height = int(height)
    return fr


def make_quality(tier: QualityTier) -> Quality:
    q = Quality()
    q.raymarch_steps = int(tier.raymarch_steps)
    q.shadow_steps = int(tier.shadow_steps)
    q.ao_steps = int(tier.ao_steps)
    q.noise_octaves = int(tier.noise_octaves)
    q.volumetric_steps = int(tier.volumetric_steps)
    q.mip_bias = float(tier.mip_bias)
    return q
