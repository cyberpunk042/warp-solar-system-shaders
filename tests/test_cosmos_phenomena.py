"""Tests for the extraordinary-cosmos phenomena (wormhole / quasar / TDE).

Run: `python -m tests.test_cosmos_phenomena`. Renders each at a tiny size on the
CPU and checks the image is finite, non-trivial, and behaves as designed (the
wormhole's throat differs from its rim; the TDE brightens as the star is devoured).
"""

import numpy as np

import warp_shaders as ws

ws.set_active("low")


def _render(name, time):
    img = np.asarray(ws.render(name, width=96, height=64, time=time))
    assert img.shape == (64, 96, 3)
    assert np.all(np.isfinite(img))
    assert img.max() > 0.05                             # something is lit
    return img


def test_wormhole_portal():
    img = _render("wormhole", 0.0)
    h, w, _ = img.shape
    centre = img[h // 2 - 6:h // 2 + 6, w // 2 - 6:w // 2 + 6].mean()
    corner = img[:8, :8].mean()
    # the throat (centre) shows the other universe; the corners are open sky —
    # they should not be identical (the portal is doing something)
    assert abs(centre - corner) > 1e-3


def test_quasar_renders():
    img = _render("quasar", 2.0)
    # the disk/jets put real signal well above the starfield floor
    assert img.mean() > 0.01


def test_tde_brightens_as_it_feeds():
    early = _render("tidal_disruption", 2.0).mean()
    late = _render("tidal_disruption", 10.0).mean()
    assert late > early                                 # the flare grows as it accretes


if __name__ == "__main__":
    test_wormhole_portal()
    print("  wormhole: throat differs from open sky (a portal): OK")
    test_quasar_renders()
    print("  quasar: finite, lit (disk + jets): OK")
    test_tde_brightens_as_it_feeds()
    print("  tidal disruption: brightens as the star is devoured: OK")
    print("ALL PASSED")
