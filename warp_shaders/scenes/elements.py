"""Elements — stylized (non-realistic) Bohr-model atoms, one scene per element.

A deliberately artistic take (the opposite of the physics-diagram proton/neutron
scenes): a glowing packed nucleus of protons (warm) and neutrons (cool), wrapped
by neon electron shells with orbiting electrons on tilted rings — the iconic
"atom logo" look.

One generic Warp kernel renders any element from runtime parameters
(Z protons, N neutrons, electrons-per-shell), so all elements share one kernel
and one code path. Correct shell occupancies and common-isotope neutron counts
are baked into the data table below; the aesthetic is stylized, not to scale.
"""

import functools

import warp as wp

from ..particles import camera_ray, emitter, noise3, orbit_ro
from ..scene import Scene

_RING_SAMPLES = 40
_GOLDEN = 2.399963  # golden angle, for Fibonacci-sphere nucleon packing


@wp.func
def fib_dir(k: int, n: int) -> wp.vec3:
    """A near-uniform direction on the sphere (Fibonacci lattice)."""
    y = 1.0 - 2.0 * (float(k) + 0.5) / float(n)
    r = wp.sqrt(wp.max(1.0 - y * y, 0.0))
    theta = _GOLDEN * float(k)
    return wp.vec3(r * wp.cos(theta), y, r * wp.sin(theta))


@wp.func
def ring_point(radius: float, a: float, tx: float, ty: float) -> wp.vec3:
    """A point on a ring of given radius at angle ``a``, tilted by (tx, ty)."""
    p = wp.vec3(radius * wp.cos(a), radius * wp.sin(a), 0.0)
    cy = wp.cos(tx)
    sy = wp.sin(tx)
    p = wp.vec3(p[0], p[1] * cy - p[2] * sy, p[1] * sy + p[2] * cy)
    cx = wp.cos(ty)
    sx = wp.sin(ty)
    return wp.vec3(p[0] * cx + p[2] * sx, p[1], -p[0] * sx + p[2] * cx)


@wp.kernel
def element_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
    z: int,           # proton count
    nn: int,          # neutron count
    n_shells: int,
    shells: wp.array(dtype=wp.int32),  # electrons per shell
):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))

    shell_base = 0.95
    shell_step = 0.6
    r_out = shell_base + float(n_shells - 1) * shell_step
    dist = 2.2 * r_out + 2.6

    ro = orbit_ro(time, mouse, res, dist)
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.6)

    col = wp.vec3(0.0, 0.0, 0.0)

    # --- Nucleus: packed protons (warm) + neutrons (cool), gently vibrating.
    a_num = z + nn
    r_nuc = 0.22 * wp.pow(float(a_num), 0.3333)
    for k in range(a_num):
        d = fib_dir(k, a_num)
        rad = r_nuc * wp.pow((float(k) + 0.5) / float(a_num), 0.3333)
        jit = 0.03 * wp.sin(time * 3.0 + float(k) * 1.7)
        p = d * (rad + jit)
        e = emitter(ro, rd, p, 0.07)
        if k < z:
            col = col + wp.vec3(1.0, 0.45, 0.2) * (e * 0.9)   # proton
        else:
            col = col + wp.vec3(0.4, 0.7, 1.0) * (e * 0.9)    # neutron

    # --- Electron shells: neon rings + orbiting electrons on tilted planes.
    for s in range(n_shells):
        rs = shell_base + float(s) * shell_step
        tx = 0.6 * float(s) + 0.3
        ty = 1.1 * float(s)
        # faint ring glow
        for rsamp in range(_RING_SAMPLES):
            a = 6.2831 * float(rsamp) / float(_RING_SAMPLES)
            col = col + wp.vec3(0.15, 0.55, 0.95) * (emitter(ro, rd, ring_point(rs, a, tx, ty), 0.02) * 0.12)
        # electrons
        ne = shells[s]
        speed = 1.4 / (float(s) + 1.0)
        for e_i in range(ne):
            a = 6.2831 * float(e_i) / float(ne) + time * speed
            pe = ring_point(rs, a, tx, ty)
            col = col + wp.vec3(0.6, 1.0, 1.0) * (emitter(ro, rd, pe, 0.06) * 1.1)

    # Stylized grade: subtle lift + gamma.
    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


def _render_element(z, nn, shells_list, width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    shells = wp.array(shells_list, dtype=wp.int32, device=device)
    wp.launch(
        element_kernel,
        dim=(height, width),
        inputs=[img, width, height, float(time),
                wp.vec2(float(mouse[0]), float(mouse[1])),
                int(z), int(nn), len(shells_list), shells],
        device=device,
    )
    wp.synchronize_device(device)
    return img.numpy()


# (Z, symbol, name, neutrons [common isotope], electrons-per-shell)
_ELEMENTS = [
    (1, "H", "hydrogen", 0, [1]),
    (2, "He", "helium", 2, [2]),
    (3, "Li", "lithium", 4, [2, 1]),
    (4, "Be", "beryllium", 5, [2, 2]),
    (5, "B", "boron", 6, [2, 3]),
    (6, "C", "carbon", 6, [2, 4]),
    (7, "N", "nitrogen", 7, [2, 5]),
    (8, "O", "oxygen", 8, [2, 6]),
    (9, "F", "fluorine", 10, [2, 7]),
    (10, "Ne", "neon", 10, [2, 8]),
    (11, "Na", "sodium", 12, [2, 8, 1]),
    (12, "Mg", "magnesium", 12, [2, 8, 2]),
    (13, "Al", "aluminium", 14, [2, 8, 3]),
    (14, "Si", "silicon", 14, [2, 8, 4]),
    (15, "P", "phosphorus", 16, [2, 8, 5]),
    (16, "S", "sulfur", 16, [2, 8, 6]),
    (17, "Cl", "chlorine", 18, [2, 8, 7]),
    (18, "Ar", "argon", 22, [2, 8, 8]),
]


def _make_scenes():
    scenes = []
    for z, sym, name, nn, shells_list in _ELEMENTS:
        cfg = "-".join(str(c) for c in shells_list)
        scenes.append(Scene(
            name=name,
            description=f"{name.capitalize()} ({sym}, Z={z}, N={nn}) — stylized Bohr atom, shells {cfg}. iMouse orbits.",
            renderer=functools.partial(_render_element, z, nn, shells_list),
        ))
    return scenes


SCENES = _make_scenes()
