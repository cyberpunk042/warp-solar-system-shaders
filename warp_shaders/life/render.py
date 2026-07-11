"""Render a plant mesh with the Warp engine — ray-cast a `wp.Mesh` per pixel.

Uploads the tessellated plant (:class:`~warp_shaders.life.mesh.Mesh`) as a
`wp.Mesh` (which builds a BVH) and casts a camera ray per pixel with
`wp.mesh_query_ray`. Hits interpolate the vertex normal + colour (barycentric)
and shade with GGX PBR + a sun, over a ground plane that catches the plant's
cast shadow, under the engine's sky and post pipeline. This is how the engine
"shows life": real generated geometry, ray-traced.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from .mesh import Mesh


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.72, 0.80, 0.92), wp.vec3(0.30, 0.52, 0.85))
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.96, 0.88), 0.9994, 0.4)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, mesh_id: wp.uint64,
                  indices: wp.array(dtype=wp.int32), vnormals: wp.array(dtype=wp.vec3),
                  vcolors: wp.array(dtype=wp.vec3), sun: wp.vec3, sun_col: wp.vec3,
                  ground_y: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    q = wp.mesh_query_ray(mesh_id, ro, rd, 1.0e6)
    t_mesh = float(1.0e30)
    if q.result:
        t_mesh = q.t

    t_gnd = float(1.0e30)
    if rd[1] < -1.0e-4:
        tg = (ground_y - ro[1]) / rd[1]
        if tg > 0.0:
            t_gnd = tg

    col = _sky(rd, sun)

    if q.result and t_mesh <= t_gnd:
        # --- plant surface ---
        p = ro + rd * t_mesh
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
        # cast shadow toward the sun
        sh = float(1.0)
        sq = wp.mesh_query_ray(mesh_id, p + n * 0.01, sun, 1.0e6)
        if sq.result:
            sh = 0.3
        direct = shade_pbr(n, -rd, sun, albedo, 0.55, 0.0, sun_col) * (2.6 * sh)
        amb = wp.cw_mul(_sky(n, sun), albedo) * (0.35 * (0.5 + 0.5 * n[1]))
        col = direct + amb
    elif t_gnd < 1.0e29:
        # --- ground plane (soil), catches the plant's shadow ---
        p = ro + rd * t_gnd
        up = wp.vec3(0.0, 1.0, 0.0)
        sh = float(1.0)
        sq = wp.mesh_query_ray(mesh_id, p + up * 0.01, sun, 1.0e6)
        if sq.result:
            sh = 0.35
        soil = wp.vec3(0.20, 0.17, 0.12)
        direct = soil * (wp.max(wp.dot(up, sun), 0.0) * 1.6 * sh + 0.25)
        col = apply_fog(direct, t_gnd, _sky(rd, sun), 0.02)

    img[i, j] = col


def render_plant(mesh: Mesh, width: int, height: int, eye, target,
                 sun_dir=(0.4, 0.8, 0.35), device: str = "cpu",
                 fov: float = 38.0, exposure: float = 1.05,
                 ground_y: float = 0.0) -> np.ndarray:
    """Ray-cast a plant mesh into an ``(H, W, 3)`` image via the Warp engine."""
    if mesh.n_tris == 0:                              # nothing grew yet
        img = np.zeros((height, width, 3), np.float32)
        return img
    wmesh = wp.Mesh(points=wp.array(mesh.verts, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.indices, dtype=wp.int32, device=device))
    vnormals = wp.array(mesh.normals, dtype=wp.vec3, device=device)
    vcolors = wp.array(mesh.colors, dtype=wp.vec3, device=device)
    idx = wp.array(mesh.indices, dtype=wp.int32, device=device)

    cam = make_camera(eye, target, fov_deg=fov, aspect=width / height)
    s = np.asarray(sun_dir, np.float32); s /= np.linalg.norm(s) + 1e-9
    sun = wp.vec3(float(s[0]), float(s[1]), float(s[2]))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, wmesh.id, idx, vnormals, vcolors, sun,
                      wp.vec3(1.0, 0.96, 0.86), float(ground_y),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=exposure)
