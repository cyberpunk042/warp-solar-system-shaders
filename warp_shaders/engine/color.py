"""Colour science — blackbody temperature, sRGB transfer, luminance, ramps.

Reusable `@wp.func` colour helpers plus host (NumPy) twins, consolidating the
temperature-colour code that was re-implemented across the star shaders, the
particle blackbody ramp, and the lava palettes:

- `kelvin_to_rgb` — a physically-grounded blackbody colour from a temperature in
  Kelvin (the Planckian locus, via the standard Tanner-Helland approximation),
  normalised so the brightest channel is ~1.
- `blackbody` — a cheap artistic 0..1 "heat" ramp (deep-red → orange → white →
  blue-white) for when you just want a hot-to-cold gradient.
- `luminance`, `linear_to_srgb` / `srgb_to_linear` — the everyday transfer +
  weighting helpers.

Device functions are for calling inside your own kernel; the ``*_np`` host
variants are for baking LUTs / palettes.
"""

from __future__ import annotations

import numpy as np
import warp as wp


# --------------------------------------------------------------------------- #
# blackbody                                                                    #
# --------------------------------------------------------------------------- #

@wp.func
def kelvin_to_rgb(kelvin: float) -> wp.vec3:
    """Approximate blackbody colour for `kelvin` (~1000..40000 K), normalised so
    the peak channel is ~1. Tanner-Helland fit to the Planckian locus."""
    k = wp.clamp(kelvin, 1000.0, 40000.0) / 100.0
    r = float(1.0)
    g = float(1.0)
    b = float(1.0)
    if k <= 66.0:
        r = 1.0
        g = wp.clamp((99.4708025861 * wp.log(k) - 161.1195681661) / 255.0,
                     0.0, 1.0)
    else:
        r = wp.clamp((329.698727446 * wp.pow(k - 60.0, -0.1332047592)) / 255.0,
                     0.0, 1.0)
        g = wp.clamp((288.1221695283 * wp.pow(k - 60.0, -0.0755148492)) / 255.0,
                     0.0, 1.0)
    if k >= 66.0:
        b = 1.0
    elif k <= 19.0:
        b = 0.0
    else:
        b = wp.clamp((138.5177312231 * wp.log(k - 10.0) - 305.0447927307)
                     / 255.0, 0.0, 1.0)
    return wp.vec3(r, g, b)


@wp.func
def blackbody(t: float) -> wp.vec3:
    """Artistic 0..1 heat ramp: deep red → orange → yellow → white → blue-white.
    (A cheap stand-in for `kelvin_to_rgb` when you only have a 0..1 heat.)"""
    t = wp.clamp(t, 0.0, 1.0)
    red = wp.vec3(0.11, 0.10, 0.09)
    orange = wp.vec3(0.85, 0.32, 0.08)
    yellow = wp.vec3(1.0, 0.80, 0.32)
    white = wp.vec3(1.0, 0.97, 0.90)
    blue = wp.vec3(0.80, 0.88, 1.0)
    if t < 0.25:
        k = t / 0.25
        return red * (1.0 - k) + orange * k
    if t < 0.5:
        k = (t - 0.25) / 0.25
        return orange * (1.0 - k) + yellow * k
    if t < 0.75:
        k = (t - 0.5) / 0.25
        return yellow * (1.0 - k) + white * k
    k = (t - 0.75) / 0.25
    return white * (1.0 - k) + blue * k


# --------------------------------------------------------------------------- #
# transfer + weighting                                                         #
# --------------------------------------------------------------------------- #

@wp.func
def wavelength_rgb(nm: float) -> wp.vec3:
    """Approximate visible-spectrum colour for a wavelength in nanometres
    (~380–750 nm), the classic piecewise Bruton approximation. Returns an
    un-normalised linear RGB (dim at the spectrum's violet/red ends). Used by the
    prism, rainbow and diffraction scenes to colour light by wavelength."""
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)
    if nm >= 380.0 and nm < 440.0:
        r = -(nm - 440.0) / (440.0 - 380.0)
        b = 1.0
    elif nm >= 440.0 and nm < 490.0:
        g = (nm - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif nm >= 490.0 and nm < 510.0:
        g = 1.0
        b = -(nm - 510.0) / (510.0 - 490.0)
    elif nm >= 510.0 and nm < 580.0:
        r = (nm - 510.0) / (580.0 - 510.0)
        g = 1.0
    elif nm >= 580.0 and nm < 645.0:
        r = 1.0
        g = -(nm - 645.0) / (645.0 - 580.0)
    elif nm >= 645.0 and nm <= 750.0:
        r = 1.0
    # intensity fall-off near the ends of the visible range
    fall = float(1.0)
    if nm < 420.0:
        fall = 0.3 + 0.7 * (nm - 380.0) / (420.0 - 380.0)
    elif nm > 700.0:
        fall = 0.3 + 0.7 * (750.0 - nm) / (750.0 - 700.0)
    return wp.vec3(r, g, b) * wp.max(fall, 0.0)


@wp.func
def luminance(c: wp.vec3) -> float:
    """Rec. 709 relative luminance."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@wp.func
def linear_to_srgb(c: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.pow(wp.clamp(c[0], 0.0, 1.0), 1.0 / 2.2),
                   wp.pow(wp.clamp(c[1], 0.0, 1.0), 1.0 / 2.2),
                   wp.pow(wp.clamp(c[2], 0.0, 1.0), 1.0 / 2.2))


@wp.func
def srgb_to_linear(c: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.pow(wp.clamp(c[0], 0.0, 1.0), 2.2),
                   wp.pow(wp.clamp(c[1], 0.0, 1.0), 2.2),
                   wp.pow(wp.clamp(c[2], 0.0, 1.0), 2.2))


# --------------------------------------------------------------------------- #
# host twins                                                                   #
# --------------------------------------------------------------------------- #

def kelvin_to_rgb_np(kelvin) -> np.ndarray:
    """Host blackbody colour (same fit as `kelvin_to_rgb`); accepts a scalar or
    an array, returns ``(..., 3)`` float32 in [0, 1]."""
    k = np.clip(np.asarray(kelvin, np.float64), 1000.0, 40000.0) / 100.0
    r = np.where(k <= 66.0, 255.0,
                 329.698727446 * np.power(np.maximum(k - 60.0, 1e-6), -0.1332047592))
    g = np.where(k <= 66.0,
                 99.4708025861 * np.log(np.maximum(k, 1e-6)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(k - 60.0, 1e-6), -0.0755148492))
    b = np.where(k >= 66.0, 255.0,
                 np.where(k <= 19.0, 0.0,
                          138.5177312231 * np.log(np.maximum(k - 10.0, 1e-6))
                          - 305.0447927307))
    rgb = np.stack([r, g, b], -1) / 255.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def linear_to_srgb_np(c) -> np.ndarray:
    return np.clip(np.asarray(c, np.float32), 0.0, 1.0) ** (1.0 / 2.2)


def srgb_to_linear_np(c) -> np.ndarray:
    return np.clip(np.asarray(c, np.float32), 0.0, 1.0) ** 2.2


def luminance_np(c) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
