"""Engine render path for the genome meshes — ray-trace a tube/strand :class:`Mesh` as a SOLID, lit specimen
on a dark background, the same look as the finished chromosome (GGX PBR key light + soft cast shadow + rim +
a waxy subsurface term). Every genome stage (fibre, telomere, chromosome …) renders through this one path, so
the whole ladder is consistent, solid, and opaque — never points.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..life.mesh import Mesh


@wp.kernel
def _strand_kernel(
    img: wp.array2d(dtype=wp.vec3),
    cam: Camera,
    mesh_id: wp.uint64,
    indices: wp.array(dtype=wp.int32),
    vnormals: wp.array(dtype=wp.vec3),
    vcolors: wp.array(dtype=wp.vec3),
    sun: wp.vec3,
    sun_col: wp.vec3,
    width: int,
    height: int,
):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # dark specimen background — a soft vertical gradient, no bright sky
    vy = float(i) / float(height)
    bg = wp.vec3(0.020, 0.023, 0.034) * (1.0 - 0.55 * vy) + wp.vec3(0.006, 0.004, 0.010)

    q = wp.mesh_query_ray(mesh_id, ro, rd, 1.0e6)
    if not q.result:
        img[i, j] = bg
        return

    p = ro + rd * q.t
    f = q.face
    i0 = indices[3 * f + 0]
    i1 = indices[3 * f + 1]
    i2 = indices[3 * f + 2]
    w1 = q.u
    w2 = q.v
    w0 = 1.0 - w1 - w2
    n = wp.normalize(vnormals[i0] * w0 + vnormals[i1] * w1 + vnormals[i2] * w2)
    if wp.dot(n, rd) > 0.0:
        n = -n
    albedo = vcolors[i0] * w0 + vcolors[i1] * w1 + vcolors[i2] * w2

    # soft cast shadow toward the key light (one mesh query; slightly softened)
    sh = float(1.0)
    sq = wp.mesh_query_ray(mesh_id, p + n * 0.02, sun, 1.0e6)
    if sq.result:
        sh = 0.28

    v_dir = -rd
    direct = shade_pbr(n, v_dir, sun, albedo, 0.46, 0.0, sun_col) * (2.9 * sh)
    amb = wp.cw_mul(wp.vec3(0.30, 0.33, 0.45), albedo) * 0.34
    sss = albedo * (0.16 * wp.clamp(wp.dot(n, sun) * 0.5 + 0.5, 0.0, 1.0))   # waxy subsurface
    fres = wp.pow(wp.clamp(1.0 + wp.dot(rd, n), 0.0, 1.0), 3.0)
    rim = wp.vec3(0.45, 0.50, 0.72) * (0.30 * fres)
    img[i, j] = direct + amb + sss + rim


def render_strand(mesh: Mesh, width: int, height: int, eye, target,
                  sun_dir=(0.42, 0.72, 0.55), device: str = "cpu", fov: float = 38.0,
                  exposure: float = 1.12) -> np.ndarray:
    """Ray-trace a genome :class:`Mesh` as a solid lit specimen on a dark background → ``(H, W, 3)`` image."""
    W, H = int(width), int(height)
    if mesh.n_tris == 0:
        return np.zeros((H, W, 3), np.float32)
    wmesh = wp.Mesh(points=wp.array(mesh.verts, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.indices, dtype=wp.int32, device=device))
    vnormals = wp.array(mesh.normals, dtype=wp.vec3, device=device)
    vcolors = wp.array(mesh.colors, dtype=wp.vec3, device=device)
    idx = wp.array(mesh.indices, dtype=wp.int32, device=device)

    s = np.asarray(sun_dir, np.float32); s /= np.linalg.norm(s) + 1e-9
    cam = make_camera(eye, target, fov_deg=fov, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(_strand_kernel, dim=(H, W),
              inputs=[img, cam, wmesh.id, idx, vnormals, vcolors,
                      wp.vec3(float(s[0]), float(s[1]), float(s[2])),
                      wp.vec3(1.0, 0.96, 0.88), W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=1.05, strength=0.30, radius=max(2, int(min(W, H) * 0.01)), passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=exposure, preserve_hue=True)
    return post.vignette(ldr, amount=0.3)
