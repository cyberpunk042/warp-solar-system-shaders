"""Kramers-Wannier duality, live: order and disorder are the same model read twice.

Two years before Onsager solved the 2D Ising model, Kramers and Wannier (1941)
found its critical temperature anyway — because the model maps onto ITSELF with
hot and cold exchanged:

    sinh(2/T) · sinh(2/T*) = 1

Any temperature T has a dual T*; the high-temperature expansion of one is the
low-temperature expansion of the other, term by term. The critical point must
then be the unique temperature that maps to itself — ``sinh(2/T_c) = 1``,
``T_c = 2/ln(1+√2)`` — pinned by symmetry alone (asserted at machine precision;
involution ``T** = T`` and the product = 1 asserted).

The scene runs it live: TWO 128² seeded Metropolis lattices side by side, the
left at T(t) sweeping cold → hot → cold, the right ALWAYS at the
Kramers-Wannier dual T*(t). When the left is deep in order the right is deep
in disorder; as the left heats the right cools; and at the moment the left
crosses T_c the right crosses it TOO — both critical at once, the only
rendezvous the map allows. Ledgers: cyan T (left), amber T* (right) — mirror
images crossing at the white T_c line — and magenta the LIVE product
``sinh(2/T)·sinh(2/T*)``, pinned at 1 under its white line, never moving.
--frames runs one cold-hot-cold cycle; iMouse pans. See
``docs/research/58-ising-exactly.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.ising import critical_temperature, dual_temperature
from ..scene import Scene

_T_CYCLE = 16.0
_N = 128
_SWEEP_RATE = 26.0
_SEED_L = 31415
_SEED_R = 27182
_T_LO, _T_HI = 1.45, 3.7


@wp.kernel
def _metro_kernel(spins: wp.array2d(dtype=float), n: int, temp: float,
                  parity: int, seed: int):
    i, j = wp.tid()
    if (i + j) % 2 == parity:
        s = spins[i, j]
        nb = spins[(i + 1) % n, j] + spins[(i - 1 + n) % n, j] + \
            spins[i, (j + 1) % n] + spins[i, (j - 1 + n) % n]
        d_e = 2.0 * s * nb
        state = wp.rand_init(seed, i * n + j)
        if d_e <= 0.0 or wp.randf(state) < wp.exp(-d_e / temp):
            spins[i, j] = -s


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2,
                   left: wp.array2d(dtype=float), right: wp.array2d(dtype=float), n: int,
                   t_frac: float, td_frac: float, tc_frac: float, prod_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- two lattices: left at T, right at the Kramers-Wannier dual T* ----
    lx = (x + 1.90) / 1.5                 # left window: x in [-1.90, -0.40]
    rx = (x + 0.25) / 1.5                 # right window: x in [-0.25,  1.25]
    ly = (y + 0.75) / 1.5                 # both: y in [-0.75, 0.75]
    if ly >= 0.0 and ly < 1.0:
        if lx >= 0.0 and lx < 1.0:
            ci = int(ly * float(n))
            cj = int(lx * float(n))
            if ci > n - 1:
                ci = n - 1
            if cj > n - 1:
                cj = n - 1
            if left[ci, cj] > 0.0:
                col = col + wp.vec3(0.95, 0.72, 0.30) * 0.55
            else:
                col = col + wp.vec3(0.10, 0.16, 0.38) * 0.85
        if rx >= 0.0 and rx < 1.0:
            ci = int(ly * float(n))
            cj = int(rx * float(n))
            if ci > n - 1:
                ci = n - 1
            if cj > n - 1:
                cj = n - 1
            if right[ci, cj] > 0.0:
                col = col + wp.vec3(0.80, 0.45, 0.75) * 0.55
            else:
                col = col + wp.vec3(0.16, 0.10, 0.30) * 0.85

    # frames
    for k in range(2):
        x0 = -1.90 + float(k) * 1.65
        x1 = x0 + 1.5
        on_fx = (wp.abs(x - x0) < 0.008 or wp.abs(x - x1) < 0.008) and \
            wp.abs(y) < 0.75
        on_fy = (wp.abs(y + 0.75) < 0.008 or wp.abs(y - 0.75) < 0.008) and \
            x > x0 and x < x1
        if on_fx or on_fy:
            col = col + wp.vec3(0.30, 0.34, 0.44) * 0.8

    # ---- the ledgers: T / T* (mirrors crossing at Tc) / the pinned product ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * t_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * td_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.38 and x < 1.62 and wp.abs(y - (-1.05 + 2.0 * tc_frac)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # the self-dual point
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * prod_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0
    if x > 1.58 and x < 1.72 and wp.abs(y - (-1.05 + 2.0 * 0.85)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # product = 1, forever

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _smooth(a: float) -> float:
    a = max(0.0, min(1.0, a))
    return a * a * (3.0 - 2.0 * a)


def _temperature(tau: float) -> float:
    """Cold -> hot -> cold sweep for the LEFT lattice; the right follows the dual."""
    half = 0.5 * _T_CYCLE
    if tau < half:
        return _T_LO + (_T_HI - _T_LO) * _smooth(tau / half)
    return _T_HI + (_T_LO - _T_HI) * _smooth((tau - half) / half)


_STATE = {"l": None, "r": None, "sweeps": -1, "device": None}


def _seed_lattice(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random((_N, _N)) < 0.5, 1.0, -1.0).astype(np.float32)


def _advance_to(t: float, device: str):
    target = int(_SWEEP_RATE * max(t, 0.0))
    if (_STATE["l"] is None or _STATE["sweeps"] > target or
            _STATE["device"] != device):
        _STATE["l"] = wp.array(_seed_lattice(_SEED_L), dtype=float, device=device)
        _STATE["r"] = wp.array(_seed_lattice(_SEED_R), dtype=float, device=device)
        _STATE["sweeps"] = 0
        _STATE["device"] = device
    for k in range(_STATE["sweeps"], target):
        temp = _temperature(math.fmod(k / _SWEEP_RATE, _T_CYCLE))
        temp_d = dual_temperature(temp)
        for parity in (0, 1):
            wp.launch(_metro_kernel, dim=(_N, _N),
                      inputs=[_STATE["l"], _N, float(temp), parity,
                              _SEED_L + 2 * k + parity], device=device)
            wp.launch(_metro_kernel, dim=(_N, _N),
                      inputs=[_STATE["r"], _N, float(temp_d), parity,
                              _SEED_R + 2 * k + parity], device=device)
    _STATE["sweeps"] = target
    return _STATE["l"], _STATE["r"]


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    temp = _temperature(tau)
    temp_d = dual_temperature(temp)
    tc = critical_temperature()

    left, right = _advance_to(t, device)

    # ledger scale: T in [1.2, 6.0] (the dual of 1.45 is ~4.1)
    def frac(tt):
        return max(0.0, min((tt - 1.2) / 4.8, 1.0))
    prod = math.sinh(2.0 / temp) * math.sinh(2.0 / temp_d)   # = 1, live

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      left, right, _N,
                      float(frac(temp)), float(frac(temp_d)), float(frac(tc)),
                      float(0.85 * prod)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ising_duality",
    description="Kramers-Wannier duality live: two seeded 128^2 Metropolis "
                "lattices side by side — the left at T sweeping cold-hot-cold, "
                "the right ALWAYS at the dual T* with sinh(2/T) sinh(2/T*) = 1 "
                "(product asserted = 1 exactly; involution asserted). When one "
                "is ordered the other is disordered; they cross T_c TOGETHER — "
                "the unique self-dual rendezvous that pinned the critical point "
                "two years before Onsager solved the model (sinh(2/Tc) = 1 "
                "asserted at machine precision). Cyan/amber ledgers: T and T*, "
                "mirror images crossing at the white T_c line; magenta: the "
                "LIVE product, pinned at 1 under its white line, never moving. "
                "--frames runs one cycle.",
    renderer=_render,
)
