"""Tests for cosmos.stellar_evolution — the phase timeline (host).

Run: `python -m tests.test_stellar_evolution` (or under pytest).
"""

import numpy as np

from warp_shaders.cosmos import bodies as B
from warp_shaders.cosmos import stellar_evolution as SE


def test_mass_fork():
    assert SE.remnant_kind(1.0) == B.WHITE_DWARF
    assert SE.remnant_kind(12.0) == B.NEUTRON
    assert SE.remnant_kind(30.0) == B.BLACK_HOLE
    # the timeline's end state matches the fork
    assert SE.phase_state(1.0, 1.0).kind == B.WHITE_DWARF
    assert SE.phase_state(1.0, 12.0).kind == B.NEUTRON
    assert SE.phase_state(1.0, 30.0).kind == B.BLACK_HOLE


def test_timeline_endpoints():
    assert SE.phase_state(0.0, 1.0).phase.startswith("protostar")
    assert SE.phase_state(0.0, 20.0).phase.startswith("protostar")
    # the birth cradle envelope is present at the start, gone on the main sequence
    assert SE.phase_state(0.02, 1.0).env == "cradle"
    assert SE.phase_state(0.30, 1.0).env == "none"


def test_fields_bounded_and_finite():
    for mass in (1.0, 15.0, 30.0):
        prev_seen = set()
        for t in np.linspace(0, 1, 101):
            s = SE.phase_state(t, mass)
            assert 0.0 <= s.temp <= 1.0
            assert 0.05 < s.radius < 6.0
            assert np.isfinite(s.radius) and np.isfinite(s.hr_temp) and np.isfinite(s.hr_lum)
            assert 0.0 <= s.flash < 3.0
            prev_seen.add(s.phase)
        assert len(prev_seen) >= 4                    # several distinct phases occur


def test_massive_ms_is_hotter():
    # an O/B main-sequence star is bluer (hotter) than the sun's G main sequence
    sun_ms = SE.phase_state(0.30, 1.0)
    massive_ms = SE.phase_state(0.25, 15.0)
    assert massive_ms.temp > sun_ms.temp
    assert massive_ms.hr_temp > sun_ms.hr_temp


def test_planetary_nebula_shell_expands():
    # the ejected envelope radius grows through the planetary-nebula phase
    r_early = SE.phase_state(0.905, 1.0).env_radius
    r_late = SE.phase_state(0.95, 1.0).env_radius
    assert SE.phase_state(0.92, 1.0).env == "planetary"
    assert r_late > r_early > 0.0


def test_supernova_flash():
    # the supernova phase spikes a flash near its head, decaying after
    head = SE.phase_state(0.685, 20.0)
    late = SE.phase_state(0.73, 20.0)
    assert head.env == "supernova" and head.flash > 0.3
    assert late.flash < head.flash


def test_hr_inset():
    import numpy as np
    frame = np.full((120, 180, 3), 0.2, np.float32)
    out = SE.draw_hr_inset(frame, 0.66, 1.0, "red giant")
    assert out.shape == frame.shape and np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0
    # the panel is drawn into the bottom-right, so that corner changes
    br = (slice(-40, None), slice(-70, None))
    assert not np.allclose(out[br], frame[br])
    # the rest of the frame is untouched
    assert np.allclose(out[:40, :70], frame[:40, :70])
    # the marker colour tracks temperature (hot bluer than cool)
    hot = SE._hr_color(0.95)
    cool = SE._hr_color(0.05)
    assert hot[2] > cool[2] and cool[0] > cool[2]


if __name__ == "__main__":
    test_mass_fork()
    print("  mass fork (WD/NS/BH) + timeline end state: OK")
    test_timeline_endpoints()
    print("  timeline endpoints + birth cradle: OK")
    test_fields_bounded_and_finite()
    print("  fields bounded + finite across the sweep: OK")
    test_massive_ms_is_hotter()
    print("  massive main sequence is hotter/bluer: OK")
    test_planetary_nebula_shell_expands()
    print("  planetary-nebula shell expands: OK")
    test_supernova_flash()
    print("  supernova flash spikes then decays: OK")
    print("ALL PASSED")
