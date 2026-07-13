"""Tests for the four-strand 'frontiers III' round — light & optics, states of matter,
electromagnetism & fields, and the cell up close.

Run: `python -m tests.test_frontiers3`. Each scene renders tiny on the CPU and is
checked to be finite, lit, and (where cheap) to show a defining signature.
"""

import numpy as np

import warp_shaders as ws
from warp_shaders.engine.color import wavelength_rgb   # noqa: F401 (import smoke)
from warp_shaders.scenes.bar_magnet import _LINES as _MAG_LINES
from warp_shaders.scenes.virus import _DIRS as _VIRUS_DIRS

ws.set_active("low")


def _render(name, time=1.0, w=96, h=72):
    img = np.asarray(ws.render(name, width=w, height=h, time=time))
    assert img.shape == (h, w, 3), f"{name}: shape {img.shape}"
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.max() > 0.05, f"{name}: nothing lit"
    return img


def test_optics_scenes():
    for s in ["prism", "rainbow", "thin_film", "diffraction_grating", "caustics",
              "interferometer"]:
        _render(s, 1.0)


def test_prism_has_full_spectrum():
    img = _render("prism", 2.0, w=160, h=130)
    # a dispersed spectrum lights every channel strongly
    assert img[..., 0].max() > 0.4 and img[..., 1].max() > 0.4 and img[..., 2].max() > 0.4


def test_matter_scenes():
    for s in ["plasma_arc", "crystallization", "ferrofluid", "boiling", "bose_einstein",
              "glass_vs_crystal"]:
        _render(s, 2.0)


def test_em_scenes():
    for s in ["bar_magnet", "electric_dipole", "em_wave", "solenoid",
              "magnetic_reconnection", "cyclotron"]:
        _render(s, 2.0)


def test_field_line_geometry():
    # each seeded field line traces many points, and the dipole is 16 lines
    assert len(_MAG_LINES) == 16
    assert all(len(ln) > 5 for ln in _MAG_LINES)
    assert _VIRUS_DIRS.shape[0] == 52 and abs(np.linalg.norm(_VIRUS_DIRS[0]) - 1.0) < 1e-4


def test_cell_scenes():
    for s in ["virus", "mitochondrion", "ribosome", "bacterium", "lipid_bilayer",
              "immune_cell"]:
        _render(s, 2.0)


if __name__ == "__main__":
    test_optics_scenes(); print("  optics scenes render: OK")
    test_prism_has_full_spectrum(); print("  prism disperses full spectrum: OK")
    test_matter_scenes(); print("  states-of-matter scenes render: OK")
    test_em_scenes(); print("  electromagnetism scenes render: OK")
    test_field_line_geometry(); print("  field-line / spike geometry: OK")
    test_cell_scenes(); print("  cell scenes render: OK")
    print("ALL PASSED")
