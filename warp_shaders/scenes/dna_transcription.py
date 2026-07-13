"""DNA transcription — reading a gene into messenger RNA.

The double helix (two phosphate backbones + base-pair rungs); an **RNA polymerase**
(a bright bead) travels along it, unzipping the strands and threading out a single
**mRNA** copy that spirals away. Ray-marched. See
``docs/research/24-the-living-body.md``. --frames animates the polymerase.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.field import sd_capsule
from ..subatomic.render import orbit_camera
from ..scene import Scene

_R = 0.7
_K = 2.4
_TUBE = 0.11
_STEP = 0.52                                            # rung spacing along the axis


@wp.func
def _bb(y: float, phase: float) -> wp.vec3:
    return wp.vec3(_R * wp.cos(_K * y + phase), y, _R * wp.sin(_K * y + phase))


@wp.func
def _helix(p: wp.vec3, unzip_y: float) -> float:
    y = p[1]
    d1 = wp.length(p - _bb(y, 0.0)) - _TUBE
    d2 = wp.length(p - _bb(y, 3.1416)) - _TUBE
    # base-pair rungs at discrete heights (fade where already unzipped, above the polymerase)
    yy = wp.floor(y / _STEP + 0.5) * _STEP
    rung = sd_capsule(p, _bb(yy, 0.0), _bb(yy, 3.1416), 0.05)
    if yy > unzip_y:
        rung = 1.0e9                                    # unzipped — no rungs yet
    return wp.min(wp.min(d1, d2), rung)


@wp.func
def _mrna(p: wp.vec3, top_y: float) -> float:
    # a single strand spiralling out to the side, emerging near the polymerase
    y = p[1]
    if y > top_y:
        return 1.0e9
    c = wp.vec3(1.4 + 0.35 * wp.cos(_K * y * 0.8), y, 0.35 * wp.sin(_K * y * 0.8))
    return wp.length(p - c) - 0.09


@wp.func
def _map(p: wp.vec3, unzip_y: float, top_y: float) -> float:
    return wp.min(_helix(p, unzip_y), _mrna(p, top_y))


@wp.func
def _nrm(p: wp.vec3, uz: float, ty: float) -> wp.vec3:
    e = 0.012
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0), uz, ty) - _map(p - wp.vec3(e, 0.0, 0.0), uz, ty),
        _map(p + wp.vec3(0.0, e, 0.0), uz, ty) - _map(p - wp.vec3(0.0, e, 0.0), uz, ty),
        _map(p + wp.vec3(0.0, 0.0, e), uz, ty) - _map(p - wp.vec3(0.0, 0.0, e), uz, ty)))


@wp.kernel
def dna_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, poly_y: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    bg = wp.vec3(0.02, 0.03, 0.05) + wp.vec3(0.03, 0.03, 0.06) * (0.5 + 0.5 * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(110):
        p = ro + rd * t
        d = _map(p, poly_y, poly_y)
        if d < 0.002:
            hit = 1
            break
        t = t + d
        if t > 16.0:
            break
    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _nrm(p, poly_y, poly_y)
        key = wp.normalize(wp.vec3(0.5, 0.7, 0.6))
        ndl = wp.max(wp.dot(n, key), 0.0)
        # colour: backbones tan; rungs cycle A/T/G/C; mRNA green
        base = wp.vec3(0.85, 0.7, 0.45)
        axr = wp.length(wp.vec2(p[0], p[2]))
        if axr < _R * 0.75:                              # inner region = a rung
            bp = wp.floor(p[1] / _STEP + 0.5)
            sel = bp - 4.0 * wp.floor(bp / 4.0)
            if sel < 1.0:
                base = wp.vec3(0.9, 0.3, 0.3)
            elif sel < 2.0:
                base = wp.vec3(0.95, 0.8, 0.3)
            elif sel < 3.0:
                base = wp.vec3(0.3, 0.6, 0.95)
            else:
                base = wp.vec3(0.4, 0.85, 0.5)
        if p[0] > 1.0:                                   # the mRNA strand
            base = wp.vec3(0.4, 0.9, 0.55)
        col = base * (0.25 + 0.85 * ndl)
        # the polymerase glow
        pd = wp.abs(p[1] - poly_y)
        col = col + wp.vec3(1.0, 0.9, 0.5) * (wp.exp(-(pd * pd) / 0.06) * wp.exp(-axr * axr * 0.5) * 1.2)
    img[i, j] = col


def _render(width, height, time, mouse, device, period=9.0):
    prog = (time % period) / period
    poly_y = 2.4 - 4.8 * prog                            # travels down the helix
    cam = orbit_camera(width, height, time, mouse, dist=6.5, fov=46.0, el0=0.08,
                       auto=0.12)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(dna_kernel, dim=(height, width),
              inputs=[img, cam, float(poly_y), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="dna_transcription",
    description="DNA transcription — the double helix (tan backbones + coloured base "
                "pairs) read by an RNA polymerase bead travelling along it, unzipping "
                "the strands and threading out a green mRNA copy. --frames animates.",
    renderer=_render,
)
