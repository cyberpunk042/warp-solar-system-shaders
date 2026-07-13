"""A virus — a protein shell with a corona of spikes.

A virus is barely alive: a protein **capsid** (often an icosahedron) around a strand of
DNA or RNA, studded with **spike** proteins it uses to grab a host cell — the corona of
SARS-CoV-2. It hijacks the cell's machinery to copy itself. See
``docs/research/33-the-cell-up-close.md``. --frames rotates the virion.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import worley3, fbm3
from ..subatomic.field import void
from ..subatomic.render import orbit_camera
from ..scene import Scene

_R = 0.8


def _spike_dirs(n=52):
    dirs = np.empty((n, 3), np.float32)
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for k in range(n):
        y = 1.0 - 2.0 * (k + 0.5) / n
        rad = math.sqrt(max(1.0 - y * y, 0.0))
        th = ga * k
        dirs[k] = (math.cos(th) * rad, y, math.sin(th) * rad)
    return dirs


_DIRS = _spike_dirs()


@wp.kernel
def virus_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, dirs: wp.array(dtype=wp.vec3),
                 ndir: int, spin: wp.mat33, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = void(rd)
    g = _rs(ro, rd, _R)
    thit = float(1e30)
    if g[0] > 0.0 and g[0] < 1.0e29:
        thit = g[0]
        p = ro + rd * g[0]
        n = wp.normalize(p)
        # protein capsomers (worley) + facet shading
        caps = worley3(n * 7.0)
        rough = fbm3(n * 5.0, 3)
        base = wp.vec3(0.5, 0.42, 0.46) * (0.5 + 0.6 * caps) + wp.vec3(0.15, 0.1, 0.12) * rough
        dif = wp.max(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.6, 0.7))), 0.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        col = base * (0.35 + 0.8 * dif) + wp.vec3(0.6, 0.4, 0.5) * fres * 0.5

    # spike proteins (glowing clubs radiating out), occluded by the near capsid surface
    for k in range(ndir):
        d = spin * dirs[k]
        for seg in range(3):
            rr = _R + 0.12 + 0.11 * float(seg)
            bulb = float(seg) / 2.0
            pk = d * rr
            tp = wp.dot(pk - ro, rd)
            if tp > 0.0 and (thit > 1.0e29 or tp < thit + 0.06):
                dist = wp.length(wp.cross(pk - ro, rd))
                w = 0.02 + 0.02 * bulb
                gl = wp.exp(-(dist / w) * (dist / w))
                col = col + wp.vec3(1.0, 0.55, 0.35) * gl * (0.5 + 1.2 * bulb) * 0.9

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=3.4, fov=42.0, el0=0.12, auto=0.0)
    a = time * 0.25
    c, s = math.cos(a), math.sin(a)
    spin = wp.mat33(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
    dirs = wp.array(_DIRS, dtype=wp.vec3, device=device)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(virus_kernel, dim=(height, width),
              inputs=[img, cam, dirs, int(_DIRS.shape[0]), spin, float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="virus",
    description="A virus particle — a protein capsid shell (capsomer-textured) studded "
                "with a corona of glowing spike proteins it uses to grab host cells, "
                "floating and slowly rotating. --frames rotates the virion.",
    renderer=_render,
)
