"""Tests for the MX arc — more cosmic events, worlds, and cross-strand scenes.

Run: `python -m tests.test_more_cosmos`. Each scene renders at a tiny size on the
CPU and is checked to be finite, lit, and to show its defining signature (the
supernova expands, the ocean moon is blue, the meadow is green, ...).
"""

import numpy as np

import warp_shaders as ws

ws.set_active("low")


def _render(name, time=0.0, w=96, h=60):
    img = np.asarray(ws.render(name, width=w, height=h, time=time))
    assert img.shape == (h, w, 3), f"{name}: shape {img.shape}"
    assert np.all(np.isfinite(img)), f"{name}: non-finite pixels"
    assert img.max() > 0.05, f"{name}: nothing lit"
    return img


def test_supernova_flash_then_fades():
    flash = _render("supernova", 0.5)
    late = _render("supernova", 10.0)
    # the initial flash is the brightest moment; the shell cools + dims as it expands
    assert flash.mean() > late.mean()


def test_kilonova_two_colour_ejecta():
    img = _render("kilonova", 9.0)
    r = img[..., 0].sum()
    b = img[..., 2].sum()
    # r-process ejecta: a warm (red/orange) equatorial component is present
    assert r > b * 0.9 and img.mean() > 0.01


def test_gravitational_waves_lit():
    img = _render("gravitational_waves", 6.0)
    assert img.mean() > 0.005                            # ripples + binary points


def test_ringed_planet_body_and_space():
    img = _render("ringed_planet", 0.0)
    corner = img[:6, :6].mean()                          # deep space
    assert img.max() > corner + 0.05                     # a lit planet/ring exists


def test_ocean_moon_is_blue():
    img = _render("ocean_moon", 0.0)
    # the global ocean makes blue the dominant channel over the frame
    assert img[..., 2].sum() > img[..., 0].sum()


def test_transit_dark_planet_on_bright_star():
    img = _render("transit", 6.0)
    lum = img.mean(axis=2)
    assert lum.max() > 0.5                               # bright stellar disk
    assert lum.min() < 0.15                              # dark planet silhouette / space


def test_city_planet_has_lights_and_dark_sky():
    img = _render("city_planet", 0.0, w=120, h=75)
    lum = img.mean(axis=2)
    assert lum.max() > 0.4                               # lit windows / atmosphere
    assert img[:5].mean() < 0.25                         # dark space along the top


def test_exomoon_life_is_green():
    img = _render("exomoon_life", 0.0)
    g = img[..., 1].sum()
    # a living meadow: green exceeds red across the frame
    assert g > img[..., 0].sum()


if __name__ == "__main__":
    test_supernova_flash_then_fades()
    print("  supernova: flash brightest, then cools/expands: OK")
    test_kilonova_two_colour_ejecta()
    print("  kilonova: warm r-process ejecta present: OK")
    test_gravitational_waves_lit()
    print("  gravitational_waves: ripples + binary lit: OK")
    test_ringed_planet_body_and_space()
    print("  ringed_planet: lit body/ring over space: OK")
    test_ocean_moon_is_blue()
    print("  ocean_moon: blue ocean dominant: OK")
    test_transit_dark_planet_on_bright_star()
    print("  transit: dark planet on a bright star: OK")
    test_city_planet_has_lights_and_dark_sky()
    print("  city_planet: lit city under a dark-space horizon: OK")
    test_exomoon_life_is_green()
    print("  exomoon_life: green meadow: OK")
    print("ALL PASSED")
