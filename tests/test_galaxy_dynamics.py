"""Tests for cosmos.galaxy_dynamics — the restricted N-body encounter (host).

Run: `python -m tests.test_galaxy_dynamics` (or under pytest).
"""

import numpy as np

from warp_shaders.cosmos.galaxy_dynamics import (Collision, EncounterConfig,
                                                 GalaxyConfig, simulate)


def _encounter(n=600, seed=3):
    g0 = GalaxyConfig(mass=1.0, n=n, r_in=0.4, r_out=2.0, incl_deg=20, spin=1,
                      center=(-4, 0, 0), vel=(0.35, 0, 0), color=(1, 0.85, 0.6))
    g1 = GalaxyConfig(mass=0.7, n=n, r_in=0.4, r_out=1.8, incl_deg=30, spin=1,
                      center=(4, 1.4, 0), vel=(-0.35, 0, 0), color=(0.6, 0.75, 1))
    return EncounterConfig(g0, g1, soft=0.25, seed=seed)


def test_isolated_disk_stays_bound():
    g0 = GalaxyConfig(mass=1.0, n=500, r_in=0.4, r_out=2.0, incl_deg=0, spin=1)
    far = GalaxyConfig(mass=1e-6, n=1, center=(1000, 0, 0))
    c = simulate(EncounterConfig(g0, far, soft=0.2), frames=40, substeps=12, dt=0.05)
    assert np.all(np.isfinite(c.part_pos))
    core = c.core_pos[:, 0, :]
    rad = np.linalg.norm(c.part_pos[:, :500, :] - core[:, None, :], axis=2)
    # no particle escapes far past the initial disk edge (orbits stay bound)
    assert rad.max() < 2.0 * 1.4
    assert abs(rad[0].mean() - rad[-1].mean()) < 0.3    # mean radius ~conserved


def test_encounter_grows_tails():
    c = simulate(_encounter(), frames=60, substeps=10, dt=0.06)
    assert isinstance(c, Collision) and c.frames == 60
    spread0 = np.linalg.norm(c.part_pos[0] - c.part_pos[0].mean(0), axis=1).mean()
    spread1 = np.linalg.norm(c.part_pos[-1] - c.part_pos[-1].mean(0), axis=1).mean()
    assert spread1 > 1.5 * spread0                      # debris flung into tails


def test_pericenter():
    c = simulate(_encounter(), frames=60, substeps=10, dt=0.06)
    sep = np.linalg.norm(c.core_pos[:, 0, :] - c.core_pos[:, 1, :], axis=1)
    assert sep.min() < sep[0] and sep.min() < sep[-1]   # a close approach in between
    assert sep[-1] > sep[0]                              # ...then recede


def test_deterministic():
    a = simulate(_encounter(seed=5), frames=20, substeps=8, dt=0.06)
    b = simulate(_encounter(seed=5), frames=20, substeps=8, dt=0.06)
    assert np.array_equal(a.part_pos, b.part_pos)


def test_particles_are_massless():
    # test particles exert no force on the cores: core trajectory is independent
    # of how many particles there are
    a = simulate(_encounter(n=300, seed=7), frames=30, substeps=8, dt=0.06)
    b = simulate(_encounter(n=1500, seed=7), frames=30, substeps=8, dt=0.06)
    assert np.allclose(a.core_pos, b.core_pos, atol=1e-6)


if __name__ == "__main__":
    test_isolated_disk_stays_bound()
    print("  isolated disk stays bound: OK")
    test_encounter_grows_tails()
    print("  encounter grows tidal tails: OK")
    test_pericenter()
    print("  fly-by has a pericenter then recedes: OK")
    test_deterministic()
    print("  deterministic from seed: OK")
    test_particles_are_massless()
    print("  test particles are massless (cores independent of N): OK")
    print("ALL PASSED")
