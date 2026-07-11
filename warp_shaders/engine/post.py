"""Host-side post-processing over the HDR framebuffer (NumPy).

A small compositing pipeline, no SciPy dependency. Typical order:

    hdr -> exposure / auto_exposure -> bloom / godrays -> tonemap
        -> chromatic_aberration -> sharpen -> vignette -> film_grain

- **exposure / auto_exposure** — scale the HDR buffer (photographic stops, or a
  geometric-mean-luminance auto key) *before* tonemapping.
- **bloom / godrays** — threshold-masked glow / radial light shafts (HDR).
- **tonemap** — ACES / Reinhard / AgX-ish, then gamma encode to [0,1].
- **chromatic_aberration / sharpen / vignette / film_grain** — the lens + film
  look, on the display-range buffer.

Sources (docs/research): Narkowicz 2016 (ACES fit); Reinhard et al. 2002;
Troy Sobotka AgX (approximation here); integral-image box blur ≈ Gaussian.
"""

import numpy as np


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _agx(x):
    # cheap AgX-like: log-ish shoulder then a slight contrast s-curve
    x = np.clip(x, 0.0, None)
    x = x / (x + 0.155)
    return np.clip(x * (x * (x * 0.9 + 0.3) + 0.05) / (x * 0.9 + 0.35), 0.0, 1.0)


def tonemap(frame, mode="aces", exposure=1.0, gamma=2.2):
    c = np.clip(np.asarray(frame, np.float32) * exposure, 0.0, None)
    if mode == "reinhard":
        c = c / (1.0 + c)
    elif mode == "agx":
        c = _agx(c)
    else:
        c = aces(c)
    return np.clip(c, 0.0, 1.0) ** (1.0 / gamma)


def _box_blur(img, r):
    """Separable box blur via integral image (fast, exact)."""
    if r < 1:
        return img
    h, w = img.shape[:2]
    pad = np.pad(img, ((r + 1, r), (r + 1, r), (0, 0)), mode="edge")
    ii = pad.cumsum(0).cumsum(1)
    y0 = np.arange(h)
    x0 = np.arange(w)
    Y1, X1 = np.meshgrid(y0 + 2 * r + 1, x0 + 2 * r + 1, indexing="ij")
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    area = (2 * r + 1) ** 2
    s = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
    return (s / area).astype(np.float32)


def bloom(hdr, threshold=1.0, strength=0.6, radius=6, passes=3):
    """Threshold the bright parts, blur, and add back (on the HDR buffer)."""
    c = np.asarray(hdr, np.float32)
    lum = c.max(axis=2, keepdims=True)
    bright = c * np.clip((lum - threshold) / max(threshold, 1e-3), 0.0, 1.0)
    b = bright
    for _ in range(passes):
        b = _box_blur(b, radius)
    return c + b * strength


def godrays(hdr, cx, cy, samples=28, density=0.9, decay=0.95, weight=0.6,
            threshold=1.2):
    """Radial light shafts from a screen-space light at (cx, cy) in [0,1].

    Accumulates a threshold-masked bright image sampled at progressively
    center-ward positions (Mitchell/GPU Gems 3 volumetric light scattering)."""
    c = np.asarray(hdr, np.float32)
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return c
    h, w = c.shape[:2]
    lum = c.max(axis=2, keepdims=True)
    bright = c * np.clip((lum - threshold) / max(threshold, 1e-3), 0.0, 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lx, ly = cx * w, cy * h
    acc = np.zeros_like(c)
    illum = 1.0
    for i in range(samples):
        s = 1.0 - density * (i + 1) / samples
        ix = np.clip((lx + (xx - lx) * s).astype(np.int64), 0, w - 1)
        iy = np.clip((ly + (yy - ly) * s).astype(np.int64), 0, h - 1)
        acc += bright[iy, ix] * (illum * weight)
        illum *= decay
    return c + acc / samples


def vignette(frame, amount=0.35):
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx / (w - 1) - 0.5)
    dy = (yy / (h - 1) - 0.5)
    r2 = (dx * dx + dy * dy) * 2.0
    v = (1.0 - amount * r2)[..., None]
    return frame * np.clip(v, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# exposure (operate on the HDR buffer, before tonemap)                        #
# --------------------------------------------------------------------------- #

def exposure(hdr, ev=0.0):
    """Scale the HDR buffer by 2**ev stops (photographic exposure)."""
    return np.asarray(hdr, np.float32) * (2.0 ** ev)


def auto_exposure(hdr, key=0.18, max_gain=8.0):
    """Scale the HDR buffer so its **geometric-mean luminance** lands near `key`
    (the classic middle-grey key value) — a simple global auto-exposure that
    keeps bright scenes from clipping and dim ones from crushing."""
    c = np.asarray(hdr, np.float32)
    lum = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    avg = float(np.exp(np.mean(np.log(np.maximum(lum, 1e-4)))))
    gain = float(np.clip(key / max(avg, 1e-4), 1.0 / max_gain, max_gain))
    return c * gain


# --------------------------------------------------------------------------- #
# lens + film look (operate on the display buffer, after tonemap)             #
# --------------------------------------------------------------------------- #

def chromatic_aberration(frame, amount=0.004):
    """Radial lens dispersion — the red channel is sampled slightly outward and
    the blue channel inward from the centre, growing toward the edges. `amount`
    is the peak shift as a fraction of the frame size."""
    c = np.asarray(frame, np.float32)
    h, w = c.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    dx = (xx - cx) / max(w, 1)
    dy = (yy - cy) / max(h, 1)

    def _samp(chan, s):
        sx = np.clip((xx + dx * s * w).astype(np.int64), 0, w - 1)
        sy = np.clip((yy + dy * s * h).astype(np.int64), 0, h - 1)
        return c[sy, sx, chan]

    out = c.copy()
    out[..., 0] = _samp(0, amount)
    out[..., 2] = _samp(2, -amount)
    return out


def film_grain(frame, amount=0.04, seed=0):
    """Additive film grain — deterministic from `seed`, slightly stronger in the
    shadows (filmic). Operates on a display-range [0,1] buffer."""
    c = np.asarray(frame, np.float32)
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(c.shape[:2]).astype(np.float32)[..., None]
    lum = c.max(axis=2, keepdims=True)
    return np.clip(c + n * amount * (0.5 + 0.5 * (1.0 - lum)), 0.0, 1.0)


def sharpen(frame, amount=0.5, radius=2):
    """Unsharp mask: add back a fraction of (image − blur) to crisp up detail."""
    c = np.asarray(frame, np.float32)
    blur = _box_blur(c, radius)
    return np.clip(c + (c - blur) * amount, 0.0, 1.0)
