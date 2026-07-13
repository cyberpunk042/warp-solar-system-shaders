"""Tests for the four-strand 'frontiers II' round — the machine, mathematics made
visible, the deep ocean, and megastructures & the far future.

Run: `python -m tests.test_frontiers2`. Each scene renders tiny on the CPU and is
checked to be finite, lit, and (where cheap) to show a defining signature.
"""

import numpy as np

import warp_shaders as ws
from warp_shaders.scenes.tesseract import _EDGES
from warp_shaders.scenes.penrose_tiling import _TILES
from warp_shaders.scenes.ai_training import _TRAJ, _loss_np

ws.set_active("low")


def _render(name, time=1.0, w=96, h=72):
    img = np.asarray(ws.render(name, width=w, height=h, time=time))
    assert img.shape == (h, w, 3), f"{name}: shape {img.shape}"
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.max() > 0.05, f"{name}: nothing lit"
    return img


def test_machine_scenes():
    for s in ["transistor", "logic_gates", "cpu_die", "data_flow", "internet",
              "quantum_computer", "ai_training"]:
        _render(s, 2.0)


def test_transistor_switches():
    on = _render("transistor", 2.0).mean()
    off = _render("transistor", 4.0).mean()
    assert on > off                                   # channel glows when ON


def test_gradient_descent_converges():
    # the Lorenz-free loss trajectory reaches the wide global minimum
    x0, z0 = _TRAJ[0]
    xe, ze = _TRAJ[-1]
    assert _loss_np(xe, ze) < _loss_np(x0, z0) - 0.5
    assert abs(xe - 0.15) < 0.25 and abs(ze + 0.05) < 0.25


def test_math_scenes():
    for s in ["strange_attractor", "torus_knot", "klein_bottle", "tesseract",
              "penrose_tiling", "domain_coloring"]:
        _render(s, 1.0)


def test_math_structure_counts():
    assert len(_EDGES) == 32                          # a tesseract has 32 edges
    assert len(_TILES) > 200                          # deflated Penrose has many tiles


def test_domain_coloring_full_hue():
    img = _render("domain_coloring", 1.0, w=140, h=120)
    # a phase portrait cycles the whole colour wheel: all channels strongly present
    assert img[..., 0].max() > 0.4 and img[..., 1].max() > 0.4 and img[..., 2].max() > 0.4


def test_ocean_scenes():
    for s in ["jellyfish", "hydrothermal_vent", "bioluminescent", "coral_reef",
              "mariana_trench", "whale_fall"]:
        _render(s, 2.0)


def test_megastructure_scenes():
    for s in ["dyson_sphere", "ringworld", "oneill_cylinder", "space_elevator",
              "generation_ship", "matrioshka_brain"]:
        _render(s, 2.0)


if __name__ == "__main__":
    test_machine_scenes(); print("  machine scenes render: OK")
    test_transistor_switches(); print("  transistor ON brighter than OFF: OK")
    test_gradient_descent_converges(); print("  gradient descent reaches global min: OK")
    test_math_scenes(); print("  math scenes render: OK")
    test_math_structure_counts(); print("  tesseract 32 edges, Penrose deflation: OK")
    test_domain_coloring_full_hue(); print("  domain colouring spans full hue: OK")
    test_ocean_scenes(); print("  deep-ocean scenes render: OK")
    test_megastructure_scenes(); print("  megastructure scenes render: OK")
    print("ALL PASSED")
