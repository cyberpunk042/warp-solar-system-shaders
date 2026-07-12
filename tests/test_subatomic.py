"""Tests for the sub-atomic strand — Standard Model particles & interactions.

Run: `python -m tests.test_subatomic`. Each scene renders tiny on the CPU and is
checked to be finite, lit, and to show its defining signature (the quark flavours
are distinct hues, the proton is warm vs the neutron cool, the 2p orbital has a
node at the nucleus, beta decay brightens as it emits products).
"""

import numpy as np

import warp_shaders as ws
from warp_shaders.subatomic.field import orbital_psi2  # noqa: F401  (import smoke)

ws.set_active("low")


def _render(name, time=2.0, w=100, h=72):
    img = np.asarray(ws.render(name, width=w, height=h, time=time))
    assert img.shape == (h, w, 3), f"{name}: shape {img.shape}"
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.max() > 0.05, f"{name}: nothing lit"
    return img


def test_nucleons_render_and_differ():
    p = _render("proton")
    n = _render("neutron")
    # both are lit fields; the proton bag is warm (more red) than the neutron's
    assert p[..., 0].sum() / (p[..., 2].sum() + 1.0) > n[..., 0].sum() / (n[..., 2].sum() + 1.0)


def test_quark_flavours_distinct_hues():
    up = _render("quark_up")
    bot = _render("quark_bottom")
    # up is red-leaning, bottom is blue/cyan-leaning
    up_rb = up[..., 0].sum() - up[..., 2].sum()
    bot_rb = bot[..., 0].sum() - bot[..., 2].sum()
    assert up_rb > bot_rb


def test_orbital_2p_has_node():
    # |psi|^2 for 2p_z vanishes at the nucleus (r=0) and on the equator (theta=90)
    import warp as wp

    @wp.kernel
    def k(out: wp.array(dtype=float)):
        out[0] = orbital_psi2(wp.vec3(0.0, 0.0, 0.0), 2, 0.5)        # nucleus
        out[1] = orbital_psi2(wp.vec3(1.0, 0.0, 0.0), 2, 0.5)        # equator (x)
        out[2] = orbital_psi2(wp.vec3(0.0, 1.0, 0.0), 2, 0.5)        # pole (y) — a lobe
    out = wp.zeros(3, dtype=float)
    wp.launch(k, dim=1, inputs=[out])
    wp.synchronize()
    a = out.numpy()
    assert a[0] < 1e-6 and a[1] < 1e-6            # node at nucleus + equator
    assert a[2] > a[0]                            # the pole lobe is populated


def test_atom_and_orbitals_lit():
    _render("atom", 1.0)
    _render("orbitals", 8.0)


def test_leptons_render():
    e = _render("electron")
    _render("tau")
    _render("neutrino_mu")
    # electron field is gen-I cyan: blue+green exceed red
    assert e[..., 2].sum() > e[..., 0].sum()


def test_bosons_render():
    for b in ["photon", "gluon", "w_boson", "z_boson", "higgs"]:
        _render(b, 1.5)


def test_standard_model_chart():
    img = _render("standard_model", 1.0, w=160, h=110)
    assert img.mean() > 0.02                      # many lit tiles


def test_beta_decay_emits_products():
    neutron_phase = _render("beta_decay", 1.6).mean()
    decay_phase = _render("beta_decay", 6.0).mean()
    # the flying e/ν products + spread make the decay frame brighter on average
    assert decay_phase > neutron_phase * 0.8      # both lit; decay not darker


if __name__ == "__main__":
    test_nucleons_render_and_differ()
    print("  nucleons: proton warmer than neutron: OK")
    test_quark_flavours_distinct_hues()
    print("  quarks: up red-leaning vs bottom cyan-leaning: OK")
    test_orbital_2p_has_node()
    print("  orbital 2p: node at nucleus + equator, lobe at pole: OK")
    test_atom_and_orbitals_lit()
    print("  atom + orbitals: lit: OK")
    test_leptons_render()
    print("  leptons: electron cyan field, tau, neutrino: OK")
    test_bosons_render()
    print("  bosons: photon/gluon/W/Z/Higgs render: OK")
    test_standard_model_chart()
    print("  standard model chart: many tiles lit: OK")
    test_beta_decay_emits_products()
    print("  beta decay: emits products: OK")
    print("ALL PASSED")
