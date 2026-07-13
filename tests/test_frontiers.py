"""Tests for the four-strand 'frontiers' round — chemistry, the origin & large-
scale universe, the living body, and Earth & weather.

Run: `python -m tests.test_frontiers`. Each scene renders tiny on the CPU and is
checked to be finite, lit, and (where cheap) to show a defining signature.
"""

import numpy as np

import warp_shaders as ws
from warp_shaders.molecules.data import water, benzene   # noqa: F401 (import smoke)

ws.set_active("low")


def _render(name, time=1.0, w=96, h=72):
    img = np.asarray(ws.render(name, width=w, height=h, time=time))
    assert img.shape == (h, w, 3), f"{name}: shape {img.shape}"
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.max() > 0.05, f"{name}: nothing lit"
    return img


def test_molecules_render():
    for m in ["water", "carbon_dioxide", "methane", "ammonia", "benzene"]:
        _render(m, 1.5)


def test_molecule_geometry():
    # water is bent: the two H are not diametrically opposite the O
    atoms, bonds = water()
    o = np.array(atoms[0][0]); h1 = np.array(atoms[1][0]); h2 = np.array(atoms[2][0])
    v1 = h1 - o; v2 = h2 - o
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ang = np.degrees(np.arccos(cosang))
    assert 100.0 < ang < 110.0                        # ~104.5°
    ba, bo = benzene()
    assert len(ba) == 12 and len(bo) == 12            # 6 C + 6 H, ring + C–H bonds


def test_salt_and_reaction():
    _render("salt_crystal", 1.0)
    _render("periodic_table", 1.0, w=140, h=90)
    early = _render("combustion", 1.0).mean()
    flash = _render("combustion", 3.6).mean()
    assert flash > early                              # exothermic flash brightens


def test_universe_scenes():
    for s in ["big_bang", "cmb", "cosmic_web", "first_stars", "structure_formation"]:
        _render(s, 3.0)


def test_cmb_has_blue_and_red():
    img = _render("cmb", 1.0)
    # the Planck palette spans cold (blue) to hot (red) — both channels present
    assert img[..., 2].max() > 0.3 and img[..., 0].max() > 0.3


def test_body_scenes():
    for s in ["neural_net", "neuron", "heartbeat", "dna_transcription", "red_blood_cells"]:
        _render(s, 1.0)


def test_heart_is_red():
    img = _render("heartbeat", 0.05)
    assert img[..., 0].sum() > img[..., 2].sum()      # red dominant


def test_earth_scenes():
    for s in ["hurricane", "lightning_storm", "plate_tectonics", "ocean_currents", "water_cycle"]:
        _render(s, 1.0)


if __name__ == "__main__":
    test_molecules_render(); print("  molecules render: OK")
    test_molecule_geometry(); print("  molecule geometry (water bent ~104.5°, benzene ring): OK")
    test_salt_and_reaction(); print("  salt / periodic table / combustion flash: OK")
    test_universe_scenes(); print("  universe scenes render: OK")
    test_cmb_has_blue_and_red(); print("  CMB spans blue↔red: OK")
    test_body_scenes(); print("  body scenes render: OK")
    test_heart_is_red(); print("  heart is red: OK")
    test_earth_scenes(); print("  earth/weather scenes render: OK")
    print("ALL PASSED")
