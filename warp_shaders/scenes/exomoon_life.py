"""Exomoon life — a living meadow under a looming ringed gas-giant parent.

Cross-strand: the **life** ecosystem (ray-traced L-System plants) growing on the
surface of an **exomoon**, with the ringed gas-giant it orbits filling the sky —
the ``ringed_planet`` cosmos strand behind a real meadow. One warm sun rakes the
grass; the giant planet and its ring hang low, tinting the twilight. Animate with
``--frames`` (the sun drifts to dusk). See
``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

import math
from dataclasses import replace

import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..life import ecosystem as _eco
from ..life import plants as _plants
from ..life.mesh import merge_meshes
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene

_ECO = _eco.Ecosystem(seed=11, pool=95, radius=11.0)

# the ringed gas-giant parent, hung low in the sky (world space, far away)
_PC = wp.constant(wp.vec3(-42.0, 34.0, -120.0))
_PR = 30.0
_RING_N = wp.constant(wp.vec3(0.16, 0.6, 0.78))        # ring-plane normal (opens toward viewer)
_RING_IN = 40.0
_RING_OUT = 66.0


@wp.func
def sun_glow(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    d = wp.max(wp.dot(rd, sun), 0.0)
    disc = wp.pow(d, 900.0) * 6.0
    halo = wp.pow(d, 6.0) * 0.4
    return wp.vec3(1.0, 0.72, 0.42) * (disc + halo)


@wp.func
def _ring_density(r: float) -> float:
    if r < _RING_IN or r > _RING_OUT:
        return 0.0
    x = (r - _RING_IN) / (_RING_OUT - _RING_IN)
    bands = 0.5 + 0.3 * wp.sin(r * 2.2) + 0.25 * wp.sin(r * 0.9) + 0.15 * wp.sin(r * 5.0)
    cassini = wp.smoothstep(0.46, 0.5, x) * (1.0 - wp.smoothstep(0.54, 0.58, x))
    edge = wp.smoothstep(0.0, 0.05, x) * (1.0 - wp.smoothstep(0.9, 1.0, x))
    return wp.clamp(bands * (1.0 - 0.9 * cassini) * edge, 0.0, 1.0)


@wp.func
def _planet_sky(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3,
                sky_lo: wp.vec3, sky_hi: wp.vec3) -> wp.vec3:
    col = sky_gradient(rd, sky_lo, sky_hi)
    # stars high in the twilight
    s = hash21(wp.vec2(wp.floor(rd[0] * 220.0), wp.floor(rd[2] * 220.0)))
    col = col + wp.vec3(1.0, 1.0, 1.0) * (wp.step(0.9968 - s) * wp.clamp(rd[1] * 3.0, 0.0, 1.0))

    # the ring (behind or in front of the planet body) — plane through _PC
    dn = wp.dot(rd, _RING_N)
    t_ring = float(1.0e30)
    if wp.abs(dn) > 1.0e-4:
        t_ring = wp.dot(_PC - ro, _RING_N) / dn
    r_a = float(0.0)
    r_col = wp.vec3(0.0, 0.0, 0.0)
    if t_ring > 0.0 and t_ring < 1.0e29:
        pr = ro + rd * t_ring
        rr = wp.length(pr - _PC)
        dens = _ring_density(rr)
        if dens > 0.001:
            r_a = wp.clamp(dens * 1.3, 0.0, 1.0)
            r_col = wp.vec3(0.95, 0.86, 0.68) * (0.6 + 0.6 * dens)

    # the gas-giant body
    g = _rs(ro - _PC, rd, _PR)
    planet_hit = g[0] > 0.0 and g[0] < 1.0e29
    p_col = wp.vec3(0.0, 0.0, 0.0)
    if planet_hit:
        n = wp.normalize((ro - _PC) + rd * g[0])
        band = 0.5 + 0.5 * wp.sin(n[1] * 9.0 + 1.2 * fbm3(n * 1.5, 4))
        base = wp.vec3(0.66, 0.44, 0.3) * (1.0 - band) + wp.vec3(0.86, 0.72, 0.5) * band
        ndl = wp.max(wp.dot(n, sun), 0.0)
        limb = wp.pow(wp.max(wp.dot(n, -rd), 0.0), 0.4)
        p_col = base * (0.08 + 0.95 * ndl) * limb

    # composite: ring in front of body if nearer
    if planet_hit and r_a > 0.0:
        if t_ring < g[0]:
            col = p_col * (1.0 - r_a) + r_col * r_a
        else:
            col = p_col
    elif planet_hit:
        col = p_col
    elif r_a > 0.0:
        col = col * (1.0 - r_a) + r_col * r_a

    col = col + sun_glow(rd, sun)
    return col


@wp.func
def sun_glow(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    d = wp.max(wp.dot(rd, sun), 0.0)
    disc = wp.pow(d, 900.0) * 6.0
    halo = wp.pow(d, 6.0) * 0.4
    return wp.vec3(1.0, 0.72, 0.42) * (disc + halo)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, mesh_id: wp.uint64,
                  indices: wp.array(dtype=wp.int32), vnormals: wp.array(dtype=wp.vec3),
                  vcolors: wp.array(dtype=wp.vec3), sun: wp.vec3, sun_col: wp.vec3,
                  sky_lo: wp.vec3, sky_hi: wp.vec3, ground_col: wp.vec3,
                  width: int, height: int):
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
        tg = -ro[1] / rd[1]
        if tg > 0.0:
            t_gnd = tg

    col = _planet_sky(ro, rd, sun, sky_lo, sky_hi)

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
        sh = float(1.0)
        sq = wp.mesh_query_ray(mesh_id, p + n * 0.01, sun, 1.0e6)
        if sq.result:
            sh = 0.3
        lit = shade_pbr(n, -rd, sun, albedo, 0.55, 0.0, sun_col) * (2.8 * sh)
        amb = wp.cw_mul(sky_gradient(n, sky_lo, sky_hi), albedo) * (0.4 * (0.5 + 0.5 * n[1]))
        col = lit + amb
    elif t_gnd < 1.0e29:
        p = ro + rd * t_gnd
        up = wp.vec3(0.0, 1.0, 0.0)
        sh = float(1.0)
        sq = wp.mesh_query_ray(mesh_id, p + up * 0.01, sun, 1.0e6)
        if sq.result:
            sh = 0.35
        ndl = wp.max(wp.dot(up, sun), 0.0)
        lit = wp.cw_mul(ground_col, sun_col) * (ndl * 1.7 * sh)
        amb = wp.cw_mul(ground_col, sky_gradient(up, sky_lo, sky_hi)) * 0.32
        bg = _planet_sky(ro, rd, sun, sky_lo, sky_hi)
        col = apply_fog(lit + amb, t_gnd, bg, 0.02)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    day = max(0.0, min(time, 1.0))
    phase = 0.16
    pal = _eco.season_palette(phase)
    vig = _eco.vigor(phase)
    # a single warm sun drifting toward dusk
    saz = -0.5 - 0.5 * day + float(mouse[0]) * 0.01
    sel = 0.26 - 0.14 * day
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), -math.cos(sel) * math.cos(saz))
    sun_col = wp.vec3(1.0, 0.6, 0.32)
    sdir = (float(sun[0]), float(sun[1]), float(sun[2]))

    meshes, offsets = [], []
    for st in _ECO.standing(2.0 + phase):
        spec = _plants.get_spec(st.plant.species)
        light_pt = (st.plant.x + sdir[0] * 6.0, 6.0, st.plant.z + sdir[2] * 6.0)
        cfg = replace(spec.cfg, palette=pal,
                      leaf_size=spec.cfg.leaf_size * (0.5 + 0.5 * vig),
                      light=light_pt, light_e=0.09,
                      tropism=(0.0, 1.0, 0.0), tropism_e=0.02)
        mesh, _b = _plants.grow_mesh_env(spec, st.gen, cfg)
        if mesh.n_tris:
            meshes.append(mesh)
            offsets.append((st.plant.x, 0.0, st.plant.z))
    field = merge_meshes(meshes, offsets)
    if field.n_tris == 0:
        import numpy as np
        return np.zeros((height, width, 3), np.float32)

    wmesh = wp.Mesh(points=wp.array(field.verts, dtype=wp.vec3, device=device),
                    indices=wp.array(field.indices, dtype=wp.int32, device=device))
    vnormals = wp.array(field.normals, dtype=wp.vec3, device=device)
    vcolors = wp.array(field.colors, dtype=wp.vec3, device=device)
    idx = wp.array(field.indices, dtype=wp.int32, device=device)

    # twilight sky, dimmed by the drifting sun
    d = 1.0 - 0.5 * day
    sky_lo = wp.vec3(0.5 * d, 0.34 * d, 0.4 * d)
    sky_hi = wp.vec3(0.16 * d, 0.2 * d, 0.42 * d)

    dist = _ECO.radius * 1.1 + 5.0
    eye = (float(mouse[0]) * 0.02, 0.7, dist)
    cam = make_camera(eye, (0.0, 2.2, -2.0), fov_deg=60.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, wmesh.id, idx, vnormals, vcolors, sun, sun_col,
                      sky_lo, sky_hi, wp.vec3(0.09, 0.21, 0.08),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="exomoon_life",
    description="A living L-System meadow on an exomoon, under a looming ringed "
                "gas-giant parent filling the twilight sky — the life strand meets "
                "the ringed-planet cosmos strand. --frames drifts the sun to dusk.",
    renderer=_render,
)
