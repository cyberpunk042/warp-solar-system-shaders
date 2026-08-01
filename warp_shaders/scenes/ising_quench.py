"""A live Ising lattice quenched through Onsager's exact critical point.

A 256² square-lattice Ising ferromagnet (J = k_B = 1) simulated LIVE on the
Warp lattice — deterministic checkerboard Metropolis, seeded, replayable — while
the temperature is swept through one 16-second cycle:

* **hot** (T = 4.0) — paramagnetic snow: spins flip freely, no order, |M| ≈ 0;
* **cooling to T_c = 2/ln(1+√2) ≈ 2.269** (Onsager 1944, self-duality
  ``sinh(2/T_c) = 1`` asserted at machine precision) — as T_c approaches,
  correlated clusters bloom AT EVERY SCALE: the critical point is the only
  temperature with no characteristic size;
* **cold** (T = 1.2) — the symmetry breaks: domains coarsen, one color wins
  locally, and |M| climbs toward Yang's exact curve;
* **reheat** — order melts again and the cycle closes.

The ledgers carry the exact solution next to the live measurement: amber is the
temperature (against its white T_c line), CYAN is Yang's closed form
``M = (1 − sinh(2/T)^{−4})^{1/8}`` at the current T (asserted: exact β = 1/8),
and MAGENTA is the lattice's live |M| — watch the measurement chase the theorem
as the lattice cools. Measured, not asserted — twice over: the suite also runs
the same seeded dynamics and asserts it lands on Yang. --frames runs one cycle;
iMouse pans. See ``docs/research/58-ising-exactly.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.ising import critical_temperature, magnetization_exact
from ..scene import Scene

_T_CYCLE = 16.0
_N = 256
_SWEEP_RATE = 26.0            # Metropolis sweeps per scene-second
_SEED = 20260801


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
                   mouse: wp.vec2, spins: wp.array2d(dtype=float), n: int,
                   temp_frac: float, tc_frac: float, yang_frac: float, m_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the lattice: a square window, spins as two-tone cells ----
    lx = (x + 1.35) / 2.2                 # lattice occupies x in [-1.35, 0.85]
    ly = (y + 1.10) / 2.2                 # and y in [-1.10, 1.10]
    if lx >= 0.0 and lx < 1.0 and ly >= 0.0 and ly < 1.0:
        ci = int(ly * float(n))
        cj = int(lx * float(n))
        if ci > n - 1:
            ci = n - 1
        if cj > n - 1:
            cj = n - 1
        s = spins[ci, cj]
        if s > 0.0:
            col = col + wp.vec3(0.95, 0.72, 0.30) * 0.55
        else:
            col = col + wp.vec3(0.10, 0.16, 0.38) * 0.85
        # domain walls: glow where a neighbor disagrees
        nb = spins[(ci + 1) % n, cj] + spins[(ci - 1 + n) % n, cj] + \
            spins[ci, (cj + 1) % n] + spins[ci, (cj - 1 + n) % n]
        if s * nb < 4.0 * s * s - 0.5:
            col = col + wp.vec3(0.25, 0.75, 0.85) * 0.22
    # frame around the lattice
    on_fx = (wp.abs(x + 1.35) < 0.008 or wp.abs(x - 0.85) < 0.008) and \
        wp.abs(y) < 1.10
    on_fy = (wp.abs(y + 1.10) < 0.008 or wp.abs(y - 1.10) < 0.008) and \
        x > -1.35 and x < 0.85
    if on_fx or on_fy:
        col = col + wp.vec3(0.30, 0.34, 0.44) * 0.8

    # ---- the ledgers: T (with Tc line) / Yang exact / live |M| ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * temp_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.38 and x < 1.52 and wp.abs(y - (-1.05 + 2.0 * tc_frac)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # Onsager's Tc, exactly
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * yang_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0      # the theorem
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * m_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0      # the measurement

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _smooth(a: float) -> float:
    a = max(0.0, min(1.0, a))
    return a * a * (3.0 - 2.0 * a)


def _temperature(tau: float) -> float:
    """The quench schedule: hot -> critical -> cold -> reheat, one cycle."""
    tc = critical_temperature()
    if tau < 4.0:                                  # hot, cooling toward Tc
        return 4.0 + (tc + 0.15 - 4.0) * _smooth(tau / 4.0)
    if tau < 7.0:                                  # hover just above/at Tc
        return tc + 0.15 - 0.15 * _smooth((tau - 4.0) / 3.0)
    if tau < 11.0:                                 # the quench: through Tc to cold
        return tc + (1.2 - tc) * _smooth((tau - 7.0) / 4.0)
    if tau < 14.0:                                 # cold: domains coarsen
        return 1.2
    return 1.2 + (4.0 - 1.2) * _smooth((tau - 14.0) / 2.0)   # reheat


_STATE = {"spins": None, "sweeps": -1, "device": None}


def _seed_lattice() -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    return np.where(rng.random((_N, _N)) < 0.5, 1.0, -1.0).astype(np.float32)


def _advance_to(t: float, device: str) -> wp.array:
    """Deterministic lattice state at absolute time t: sweep k runs at
    temperature T(k/rate) with RNG seed derived from k — so any t is
    replayable from the seed lattice, and sequential frames advance cheaply."""
    target = int(_SWEEP_RATE * max(t, 0.0))
    if (_STATE["spins"] is None or _STATE["sweeps"] > target or
            _STATE["device"] != device):
        _STATE["spins"] = wp.array(_seed_lattice(), dtype=float, device=device)
        _STATE["sweeps"] = 0
        _STATE["device"] = device
    spins = _STATE["spins"]
    for k in range(_STATE["sweeps"], target):
        temp = _temperature(math.fmod(k / _SWEEP_RATE, _T_CYCLE))
        for parity in (0, 1):
            wp.launch(_metro_kernel, dim=(_N, _N),
                      inputs=[spins, _N, float(temp), parity,
                              _SEED + 2 * k + parity],
                      device=device)
    _STATE["sweeps"] = target
    return spins


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    temp = _temperature(tau)
    tc = critical_temperature()

    spins = _advance_to(t, device)
    # the live order parameter, measured BLOCK-LOCALLY (32^2 blocks): a fast
    # quench traps opposite domains, so the global mean would hide the local
    # order the lattice has actually built — Yang's M is what each domain
    # carries inside
    lat = spins.numpy()
    blocks = lat.reshape(_N // 32, 32, _N // 32, 32).mean(axis=(1, 3))
    m_live = float(np.abs(blocks).mean())

    temp_frac = (temp - 1.0) / 3.2
    tc_frac = (tc - 1.0) / 3.2
    yang_frac = magnetization_exact(temp)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      spins, _N,
                      float(temp_frac), float(tc_frac),
                      float(yang_frac), float(m_live)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ising_quench",
    description="a LIVE 256^2 Ising lattice (deterministic seeded Metropolis "
                "on the Warp lattice) quenched through Onsager's exact critical "
                "point T_c = 2/ln(1+sqrt2) ~ 2.269 (self-duality sinh(2/Tc)=1 "
                "asserted at machine precision): paramagnetic snow when hot, "
                "correlated clusters at EVERY scale as T_c approaches, then the "
                "symmetry breaks and domains coarsen. Ledgers: amber "
                "temperature against its white T_c line; cyan Yang's exact "
                "M = (1 - sinh(2/T)^-4)^(1/8) at the current T (exact beta = "
                "1/8 asserted); magenta the lattice's LIVE measured |M| — the "
                "measurement chasing the theorem. The suite runs the same "
                "seeded dynamics and asserts it lands on Yang's closed form. "
                "--frames runs one hot-critical-cold-reheat cycle.",
    renderer=_render,
)
