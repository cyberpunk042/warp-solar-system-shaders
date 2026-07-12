"""The bosons — photon, gluon, W, Z, and the Higgs.

The Standard Model's force carriers plus the Higgs, each with a physically-motivated
look: the photon a travelling EM wave packet, the gluon a colour+anticolour double
helix, the W/Z heavy cores decaying to back-to-back jets, the Higgs a golden core
over its field lattice decaying to two photons. See
``docs/research/21-standard-model.md``. Animate with ``--frames``.
"""

from ..scene import Scene
from ..subatomic import boson


def _mk(name, fn, desc):
    def _r(width, height, time, mouse, device, _fn=fn):
        return _fn(width, height, time, mouse, device)
    return Scene(name=name, description=desc, renderer=_r)


SCENES = [
    _mk("photon", boson.render_photon,
        "The photon (γ) — a travelling transverse EM wave packet, oscillating E "
        "(gold) ⊥ B (cyan) sweeping along its axis. Massless. --frames animates."),
    _mk("gluon", boson.render_gluon,
        "The gluon (g) — the strong-force carrier, a colour + anticolour double "
        "helix (it carries both). Massless. --frames animates."),
    _mk("w_boson", boson.render_w,
        "The W boson (80.4 GeV) — a heavy, short-lived weak boson: a dense orange "
        "core decaying into a back-to-back pair of jets. --frames animates."),
    _mk("z_boson", boson.render_z,
        "The Z boson (91.2 GeV) — the neutral weak boson: a dense blue core "
        "decaying into a back-to-back fermion pair. --frames animates."),
    _mk("higgs", boson.render_higgs,
        "The Higgs boson (125 GeV) — an excitation of the all-pervading Higgs "
        "field (faint lattice), decaying to two photon jets (H→γγ). --frames."),
]
