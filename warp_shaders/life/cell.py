"""A living cell — and its division — in the glow-impostor style of the atom strand.

Rendered like `scenes/electron.py` / `scenes/elements.py` (no mesh): one Warp
kernel accumulates glow along each camera ray. The cell is a **metaball
membrane** (two blobs whose centres separate as it divides), a translucent
**cytoplasm** volume, a bright **nucleus**, and **organelle** emitters packed on
a Fibonacci sphere and partitioned into the two daughter cells as ``divide``
rises from 0 (one cell) to 1 (two cells) — mitosis.

Reuses `warp_shaders.particles` (`emitter`, `ray_sphere`, `orbit_ro`,
`camera_ray`, `noise3`) and the `fib_dir` packing from `scenes/elements.py`.
"""

import warp as wp

from ..particles import camera_ray, emitter, noise3, orbit_ro, ray_sphere

_GOLDEN = 2.399963
_MARCH = 48
_N_ORG = 26


@wp.func
def _fib(k: int, n: int) -> wp.vec3:
    y = 1.0 - 2.0 * (float(k) + 0.5) / float(n)
    r = wp.sqrt(wp.max(1.0 - y * y, 0.0))
    th = _GOLDEN * float(k)
    return wp.vec3(r * wp.cos(th), y, r * wp.sin(th))


@wp.func
def _field(p: wp.vec3, cl: wp.vec3, cr: wp.vec3, sig: float) -> float:
    """Two-blob metaball field (gaussians at the daughter centres)."""
    dl = p - cl
    dr = p - cr
    fl = wp.exp(-wp.dot(dl, dl) / sig)
    fr = wp.exp(-wp.dot(dr, dr) / sig)
    return fl + fr


@wp.kernel
def cell_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int,
                time: float, mouse: wp.vec2, divide: float, n_org: int):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    ro = orbit_ro(time, mouse, res, 6.5)
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.7)

    sep = divide * 1.15
    cl = wp.vec3(-sep, 0.0, 0.0)
    cr = wp.vec3(sep, 0.0, 0.0)
    r_cell = 1.5
    sig = r_cell * r_cell * 0.62
    iso = 0.85

    col = wp.vec3(0.0, 0.0, 0.0)

    # --- cytoplasm + membrane: march the metaball volume ---
    bound = r_cell + sep + 0.6
    tn, tf, hit = ray_sphere(ro, rd, wp.vec3(0.0, 0.0, 0.0), bound)
    if hit != 0:
        t0 = wp.max(tn, 0.0)
        dt = (tf - t0) / float(_MARCH)
        t = t0
        for _s in range(_MARCH):
            p = ro + rd * t
            f = _field(p, cl, cr, sig)
            inside = wp.max(f - iso, 0.0)
            # translucent cytoplasm (soft interior fill)
            col = col + wp.vec3(0.28, 0.62, 0.52) * (inside * dt * 0.55)
            # membrane rim: a thin bright shell where f ~= iso
            rim = wp.exp(-((f - iso) / 0.10) * ((f - iso) / 0.10))
            col = col + wp.vec3(0.55, 0.85, 0.80) * (rim * dt * 0.9)
            t = t + dt

    # --- nuclei: coincide at the origin (one cell) then part into two ---
    col = col + wp.vec3(0.75, 0.55, 0.9) * (emitter(ro, rd, cl, 0.42) * 1.1)
    col = col + wp.vec3(0.75, 0.55, 0.9) * (emitter(ro, rd, cr, 0.42) * 1.1)

    # --- organelles: packed on a sphere, partitioned to each daughter ---
    for k in range(n_org):
        d = _fib(k, n_org)
        side = cl
        if (k % 2) == 1:
            side = cr
        jit = 0.05 * noise3(wp.vec3(time * 0.6, float(k) * 2.3, 0.0))
        p = side + d * (r_cell * 0.72 + jit)
        e = emitter(ro, rd, p, 0.10)
        col = col + wp.vec3(0.95, 0.75, 0.35) * (e * 0.55)   # warm organelles

    # gamma lift (matches the atom strand grade)
    img[i, j] = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                        wp.pow(wp.max(col[1], 0.0), 0.4545),
                        wp.pow(wp.max(col[2], 0.0), 0.4545))


def render_cell(width, height, time, mouse, divide, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(cell_kernel, dim=(height, width),
              inputs=[img, int(width), int(height), float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(divide), int(_N_ORG)], device=device)
    wp.synchronize_device(device)
    return img.numpy()
