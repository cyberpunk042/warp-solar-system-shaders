"""Volumetric clouds / smoke (Warp device functions).

A density-field raymarcher: march the view ray through a cloud slab, accumulating
scattered light with Beer-Lambert extinction; at each step light-march toward the
sun for self-shadowing. Uses the Henyey-Greenstein phase function and a "powder"
term for the dark-edge look. Density follows the Schneider "Nubis" recipe:
low-frequency billow shape, a coverage remap, a height gradient, and high-frequency
Worley erosion at the edges.

Sources (docs/research): Schneider & Vos, "The Real-time Volumetric Cloudscapes of
Horizon Zero Dawn" (SIGGRAPH 2015); Henyey-Greenstein 1941; Beer-Lambert.
Sample counts scale with the quality tier.
"""

import warp as wp

from ..procedural.noise import fbm3, value_tiled3
from ..textures import sample3d

_PI = 3.14159265
_DETAIL_TILE = 64.0     # world units per repeat of the baked detail volume


@wp.func
def hg_phase(cos_theta: float, g: float) -> float:
    g2 = g * g
    return (1.0 - g2) / (4.0 * _PI * wp.pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5))


# ---- baked tileable detail volume -------------------------------------------
# The high-frequency edge erosion is the same seamless 3D field every step, so
# bake it ONCE into a wp.array3d and sample it with trilinear `sample3d` (cheap)
# instead of recomputing Worley/fBm per march step. `value_tiled3` keeps the
# volume seamless so wrap-repeat sampling has no visible tile joins.

@wp.kernel
def bake_cloud_detail(vol: wp.array3d(dtype=float), size: int):
    z, y, x = wp.tid()
    q = wp.vec3(float(x), float(y), float(z)) / float(size)     # [0,1)^3
    p = 5.0                                                     # base features / tile
    d = (value_tiled3(q * p, p) * 0.55
         + value_tiled3(q * p * 2.0, p * 2.0) * 0.30
         + value_tiled3(q * p * 3.0, p * 3.0) * 0.15)           # tileable fbm
    vol[z, y, x] = d


def build_cloud_detail(size=96, device="cpu"):
    """Bake a seamless 3D cloud-erosion detail volume once (host)."""
    vol = wp.zeros((size, size, size), dtype=float, device=device)
    wp.launch(bake_cloud_detail, dim=(size, size, size), inputs=[vol, int(size)],
              device=device)
    wp.synchronize_device(device)
    return vol


@wp.func
def cloud_density(p: wp.vec3, time: float, coverage: float,
                  base_y: float, top_y: float,
                  vol: wp.array3d(dtype=float)) -> float:
    """Cloud density in [0,1] at a point in the slab [base_y, top_y].

    `vol` is a seamless detail volume from :func:`build_cloud_detail`.
    """
    hf = wp.clamp((p[1] - base_y) / (top_y - base_y), 0.0, 1.0)
    wind = wp.vec3(time * 0.6, 0.0, time * 0.2)
    q = p * 0.022 + wind * 0.03
    shape = fbm3(q, 5)
    d = wp.clamp((shape - (1.0 - coverage)) / wp.max(coverage, 1e-3), 0.0, 1.0)
    # height gradient: rounded bottoms, softer tops (cumulus-ish)
    gh = wp.smoothstep(0.0, 0.12, hf) * wp.smoothstep(1.0, 0.5, hf)
    d = d * gh
    # high-frequency edge erosion from the baked tileable volume (billowy look)
    wu = time * 0.02
    det = sample3d(vol, p[0] / _DETAIL_TILE + wu, p[1] / _DETAIL_TILE,
                   p[2] / _DETAIL_TILE + wu * 0.3, 1)
    d = wp.clamp(d - (1.0 - d) * det * 0.30, 0.0, 1.0)
    return d


@wp.func
def march_clouds(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, time: float,
                 coverage: float, base_y: float, top_y: float,
                 steps: int, light_steps: int,
                 sun_col: wp.vec3, amb: wp.vec3,
                 vol: wp.array3d(dtype=float)) -> wp.vec4:
    """Raymarch a horizontal cloud slab. Returns (scattered_rgb, transmittance).

    `vol` is a seamless detail volume from :func:`build_cloud_detail`.
    """
    if wp.abs(rd[1]) < 1e-4:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    t_base = (base_y - ro[1]) / rd[1]
    t_top = (top_y - ro[1]) / rd[1]
    t_enter = wp.max(wp.min(t_base, t_top), 0.0)
    t_exit = wp.max(t_base, t_top)
    if t_exit <= t_enter:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    t_exit = wp.min(t_exit, 900.0)

    seg = (t_exit - t_enter) / float(steps)
    sigma = 0.11
    trans = float(1.0)
    scat = wp.vec3(0.0, 0.0, 0.0)
    sun_up = wp.max(sun[1], 0.35)
    seg_l = (top_y - base_y) / float(light_steps) / sun_up
    g_phase = hg_phase(wp.dot(rd, sun), 0.45)

    t = t_enter + 0.5 * seg
    for _ in range(steps):
        p = ro + rd * t
        d = cloud_density(p, time, coverage, base_y, top_y, vol)
        if d > 0.002:
            od_l = float(0.0)
            tl = seg_l * 0.5
            for _ in range(light_steps):
                od_l += cloud_density(p + sun * tl, time, coverage, base_y, top_y, vol) * seg_l
                tl += seg_l
            t_light = wp.exp(-sigma * od_l)
            powder = 1.0 - wp.exp(-2.0 * d)
            sample = sun_col * (t_light * g_phase * powder * 6.0) + amb
            d_trans = wp.exp(-sigma * d * seg)
            scat += sample * (trans * (1.0 - d_trans))
            trans *= d_trans
            if trans < 0.01:
                break
        t += seg
    return wp.vec4(scat[0], scat[1], scat[2], trans)
