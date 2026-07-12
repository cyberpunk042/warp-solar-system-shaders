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
                  ground_y: float, draw_ground: int, width: int, height: int):
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
    if draw_ground != 0 and rd[1] < -1.0e-4:
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


@wp.func
def _world_sky(rd: wp.vec3, sky_lo: wp.vec3, sky_hi: wp.vec3,
               suns: wp.array(dtype=wp.vec3), sun_cols: wp.array(dtype=wp.vec3),
               n_suns: int) -> wp.vec3:
    base = sky_gradient(rd, sky_lo, sky_hi)
    for k in range(n_suns):
        base = base + sun_disk(rd, suns[k], sun_cols[k], 0.9994, 0.4)
    return base


@wp.kernel
def render_world_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, mesh_id: wp.uint64,
                        indices: wp.array(dtype=wp.int32), vnormals: wp.array(dtype=wp.vec3),
                        vcolors: wp.array(dtype=wp.vec3), suns: wp.array(dtype=wp.vec3),
                        sun_cols: wp.array(dtype=wp.vec3), sun_ints: wp.array(dtype=float),
                        n_suns: int, sky_lo: wp.vec3, sky_hi: wp.vec3, ground_col: wp.vec3,
                        ground_y: float, draw_ground: int, width: int, height: int):
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
    if draw_ground != 0 and rd[1] < -1.0e-4:
        tg = (ground_y - ro[1]) / rd[1]
        if tg > 0.0:
            t_gnd = tg

    col = _world_sky(rd, sky_lo, sky_hi, suns, sun_cols, n_suns)

    if q.result and t_mesh <= t_gnd:
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
        lit = wp.vec3(0.0, 0.0, 0.0)
        for k in range(n_suns):                          # sum every sun + its own shadow
            sh = float(1.0)
            sq = wp.mesh_query_ray(mesh_id, p + n * 0.01, suns[k], 1.0e6)
            if sq.result:
                sh = 0.3
            lit = lit + shade_pbr(n, -rd, suns[k], albedo, 0.55, 0.0, sun_cols[k]) \
                * (2.6 * sun_ints[k] * sh)
        amb = wp.cw_mul(sky_gradient(n, sky_lo, sky_hi), albedo) * (0.35 * (0.5 + 0.5 * n[1]))
        col = lit + amb
    elif t_gnd < 1.0e29:
        p = ro + rd * t_gnd
        up = wp.vec3(0.0, 1.0, 0.0)
        lit = wp.vec3(0.0, 0.0, 0.0)
        for k in range(n_suns):
            sh = float(1.0)
            sq = wp.mesh_query_ray(mesh_id, p + up * 0.01, suns[k], 1.0e6)
            if sq.result:
                sh = 0.35
            ndl = wp.max(wp.dot(up, suns[k]), 0.0)
            lit = lit + wp.cw_mul(ground_col, sun_cols[k]) * (ndl * 1.6 * sun_ints[k] * sh)
        amb = wp.cw_mul(ground_col, sky_gradient(up, sky_lo, sky_hi)) * 0.3
        col = apply_fog(lit + amb, t_gnd, _world_sky(rd, sky_lo, sky_hi, suns, sun_cols, n_suns), 0.02)

    img[i, j] = col


def _sky_tint(primary_el):
    """Horizon + zenith sky colours from the primary sun's elevation (its `y`):
    warm + low-contrast near the horizon (dawn/dusk), blue overhead (noon)."""
    day = float(np.clip(primary_el * 2.2, 0.0, 1.0))
    warm = np.array([1.0, 0.5, 0.26], np.float32)
    pale = np.array([0.72, 0.80, 0.92], np.float32)
    dusk = np.array([0.34, 0.26, 0.42], np.float32)
    deep = np.array([0.26, 0.46, 0.85], np.float32)
    horizon = warm + (pale - warm) * day
    zenith = dusk + (deep - dusk) * day
    return horizon, zenith


def render_world(mesh: Mesh, width: int, height: int, eye, target, suns,
                 device: str = "cpu", fov: float = 46.0, exposure: float = 1.08,
                 ground_y: float = 0.0, ground_col=(0.10, 0.24, 0.09)) -> np.ndarray:
    """Ray-cast a life mesh lit by **N suns** — a planet surface under its solar
    system. `suns` is a list of ``(dir3, colour3, intensity)`` — each casts its
    own shadow and tints the sky with its disc. The sky day/twilight colour is
    derived from the first (primary) sun's elevation. See
    ``docs/research/16-a-living-world.md``."""
    if mesh.n_tris == 0:
        return np.zeros((height, width, 3), np.float32)
    wmesh = wp.Mesh(points=wp.array(mesh.verts, dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.indices, dtype=wp.int32, device=device))
    vnormals = wp.array(mesh.normals, dtype=wp.vec3, device=device)
    vcolors = wp.array(mesh.colors, dtype=wp.vec3, device=device)
    idx = wp.array(mesh.indices, dtype=wp.int32, device=device)

    dirs = np.zeros((len(suns), 3), np.float32)
    cols = np.zeros((len(suns), 3), np.float32)
    ints = np.zeros(len(suns), np.float32)
    for k, (d, c, inten) in enumerate(suns):
        dd = np.asarray(d, np.float32); dd /= np.linalg.norm(dd) + 1e-9
        dirs[k] = dd
        cols[k] = np.asarray(c, np.float32)
        ints[k] = float(inten)
    horizon, zenith = _sky_tint(dirs[0][1])

    cam = make_camera(eye, target, fov_deg=fov, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_world_kernel, dim=(height, width),
              inputs=[img, cam, wmesh.id, idx, vnormals, vcolors,
                      wp.array(dirs, dtype=wp.vec3, device=device),
                      wp.array(cols, dtype=wp.vec3, device=device),
                      wp.array(ints, dtype=wp.float32, device=device), int(len(suns)),
                      wp.vec3(*horizon.tolist()), wp.vec3(*zenith.tolist()),
                      wp.vec3(*ground_col), float(ground_y), int(1),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=exposure)


def render_plant(mesh: Mesh, width: int, height: int, eye, target,
                 sun_dir=(0.4, 0.8, 0.35), device: str = "cpu",
                 fov: float = 38.0, exposure: float = 1.05,
                 ground_y: float = 0.0, ground: bool = True) -> np.ndarray:
    """Ray-cast a mesh into an ``(H, W, 3)`` image via the Warp engine.

    Set ``ground=False`` for a floating subject (a molecule / cell) — the
    shadow-catching soil plane is dropped and the mesh renders on pure sky.
    """
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
                      int(1 if ground else 0), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=exposure)
