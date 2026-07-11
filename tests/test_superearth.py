"""Tests for the configurable super-earth (:mod:`warp_shaders.superearth`).

Run: `python -m tests.test_superearth` (or under pytest). Verifies that the
config knobs are wired, that the planet is deterministic from its config, that
different knobs actually change the image (with/without mountains, rocky vs
gas), that the moon and bombardment helpers are coherent, and that every preset
scene is registered.
"""

import math

import numpy as np
import warp as wp

from warp_shaders.superearth import bombardment as B
from warp_shaders.superearth import moons as M
from warp_shaders.superearth import presets
from warp_shaders.superearth.planet import make_config, render_planet

wp.init()

_W, _H = 64, 48


def _img(cfg, time=0.5):
    return render_planet(cfg, _W, _H, time=time, device="cpu", quality="low")


def test_presets_all_build():
    names = presets.names()
    for want in ("barren", "earthlike", "gas_giant", "windstorm",
                 "electrostorm", "flatland", "volcanic", "living"):
        assert want in names, f"preset {want} missing"
    for n in names:
        cfg = presets.get(n)
        assert cfg is not None


def test_make_config_overrides():
    cfg = make_config(mountain=0.0, has_ocean=0, gas=1.0, seed=3.0)
    assert cfg.mountain == 0.0 and cfg.has_ocean == 0
    assert cfg.gas == 1.0 and cfg.seed == 3.0
    # untouched fields keep their defaults
    assert cfg.spin == 0.05


def test_render_is_finite():
    img = _img(presets.get("earthlike"))
    assert img.shape == (_H, _W, 3)
    assert np.all(np.isfinite(img)) and img.min() >= 0.0


def test_deterministic():
    # same config + time -> identical image (no hidden RNG in the planet path)
    a = _img(presets.get("earthlike"))
    b = _img(presets.get("earthlike"))
    assert np.array_equal(a, b), "planet render is not deterministic"


def test_mountains_knob_changes_image():
    # the with/without-mountains knob must actually change the surface
    flat = _img(presets.get("flatland"))
    mount = _img(make_config(seed=1.0, mountain=1.0, has_ocean=1,
                             has_rivers=1, has_atmo=1, atmo=1.0, veg=0.9))
    assert not np.allclose(flat, mount, atol=1e-3)


def test_gas_differs_from_rocky():
    gas = _img(presets.get("gas_giant"))
    rock = _img(presets.get("earthlike"))
    assert not np.allclose(gas, rock, atol=1e-3)
    # a gas world has no solid-terrain path; its background is clean space
    assert float(np.asarray(gas[0, 0]).max()) < 0.05


def test_moon_state():
    ms = M.moonset("many")
    assert len(ms) >= 1
    pos, rad, typ = M.moon_state(ms, time=1.0)
    assert pos.shape[0] == len(ms) and pos.shape[1] == 3
    assert rad.shape[0] == len(ms) and typ.shape[0] == len(ms)
    assert np.all(np.isfinite(pos)) and np.all(rad > 0.0)
    # deterministic in time
    pos2, _, _ = M.moon_state(ms, time=1.0)
    assert np.array_equal(pos, pos2)


def test_bombardment_sites():
    n = 40
    front = np.array([0.3, 0.2, 0.93], np.float32)
    front /= np.linalg.norm(front)
    st = B.sites(n, "clustered", seed=2, front=front)
    assert st.shape == (n, 3)
    norms = np.linalg.norm(st, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "sites must be unit vectors"
    # front-bias: the strong majority land on the camera-facing hemisphere
    facing = (st @ front) > 0.0
    assert facing.mean() > 0.8, "front-bias failed"


def test_bombardment_fire_frames_monotone():
    bcfg = B.BombConfig(n=12, delay=0.3, interval=0.28, parallel=3)
    ff = B._fire_frames(bcfg, dt=0.07)
    assert len(ff) == 12
    assert ff == sorted(ff), "wave fire-frames must be non-decreasing"
    assert min(ff) >= 0


def test_scenes_registered():
    import warp_shaders.scenes.super_earth as se
    names = {s.name for s in se.SCENES}
    for want in ("super_earth", "se_gas", "se_windstorm", "se_electrostorm",
                 "se_flat", "se_moons"):
        assert want in names, f"scene {want} not registered"


if __name__ == "__main__":
    test_presets_all_build()
    print("  presets all build:", len(presets.names()), "OK")
    test_make_config_overrides()
    print("  make_config overrides: OK")
    test_render_is_finite()
    print("  render finite: OK")
    test_deterministic()
    print("  deterministic render: OK")
    test_mountains_knob_changes_image()
    print("  mountains knob changes image: OK")
    test_gas_differs_from_rocky()
    print("  gas differs from rocky + clean space bg: OK")
    test_moon_state()
    print("  moon state: OK")
    test_bombardment_sites()
    print("  bombardment sites (unit + front-bias): OK")
    test_bombardment_fire_frames_monotone()
    print("  bombardment fire-frames monotone: OK")
    test_scenes_registered()
    print("  preset scenes registered: OK")
    print("ALL PASSED")
