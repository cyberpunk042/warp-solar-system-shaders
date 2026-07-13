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


def tonemap(frame, mode="aces", exposure=1.0, gamma=2.2, preserve_hue=False):
    """Tonemap an HDR buffer to display range and gamma-encode.

    ``preserve_hue`` applies the curve to **luminance** and rescales the colour by
    that ratio, so bright saturated emitters (stars, plasma, a hot disk) keep their
    hue instead of desaturating toward yellow/white the way per-channel ACES does.
    A gentle highlight desaturation is still folded in so the very brightest cores
    can reach white."""
    c = np.clip(np.asarray(frame, np.float32) * exposure, 0.0, None)

    def _curve(v):
        if mode == "reinhard":
            return v / (1.0 + v)
        if mode == "agx":
            return _agx(v)
        return aces(v)

    if preserve_hue:
        lum = np.maximum(0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2], 1e-6)
        tl = _curve(lum)
        out = c * (tl / lum)[..., None]
        # let the brightest cores bleach to white (highlight desaturation)
        w = np.clip(tl - 0.85, 0.0, 0.15)[..., None] / 0.15
        out = out * (1.0 - w) + tl[..., None] * w
        c = out
    else:
        c = _curve(c)
    return np.clip(c, 0.0, 1.0) ** (1.0 / gamma)


def downsample(img, k):
    """Average-pool an image by integer factor ``k`` (SSAA box downsample). The
    image dims must be divisible by ``k`` — render at ``k×`` then call this."""
    k = int(k)
    if k <= 1:
        return np.asarray(img, np.float32)
    c = np.asarray(img, np.float32)
    h, w = c.shape[0] // k, c.shape[1] // k
    return c[:h * k, :w * k].reshape(h, k, w, k, -1).mean(axis=(1, 3))


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


def _soft_bright(c, threshold, knee):
    """Extract the bright parts with a smooth **soft knee** around `threshold`
    (Unreal-style) instead of a hard cut — no ringing at the threshold edge."""
    lum = c.max(axis=2, keepdims=True)
    knee = max(knee, 1e-4)
    soft = np.clip(lum - threshold + knee, 0.0, 2.0 * knee)
    soft = soft * soft / (4.0 * knee)
    contrib = np.maximum(soft, lum - threshold) / np.maximum(lum, 1e-4)
    return c * np.clip(contrib, 0.0, 1.0)


def bloom(hdr, threshold=1.0, strength=0.6, radius=6, passes=3, knee=0.5, octaves=4):
    """Soft-knee, **multi-scale** bloom on the HDR buffer.

    The bright parts (soft-knee-thresholded) are blurred at a pyramid of growing
    radii and summed with halving weights, giving a wide, smooth HDR glow with a
    tight bright core and a soft falloff — instead of a single box-blur halo. The
    weighted blend is energy-normalised, so `strength` keeps the same overall
    magnitude as the old single-scale bloom. `knee` softens the threshold;
    `octaves` sets how many doublings of the blur radius contribute."""
    c = np.asarray(hdr, np.float32)
    bright = _soft_bright(c, threshold, knee * max(threshold, 1e-3))
    acc = np.zeros_like(c)
    wsum = 0.0
    b = bright
    r = max(1, int(radius))
    for o in range(max(1, octaves)):
        for _ in range(passes):
            b = _box_blur(b, r)
        w = 0.5 ** o
        acc += b * w
        wsum += w
        r = max(1, r * 2)               # pyramid: each octave doubles the radius
    return c + (acc / max(wsum, 1e-6)) * strength


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


# --------------------------------------------------------------------------- #
# named looks — one-call display-range grades built from the ops above         #
# --------------------------------------------------------------------------- #

LOOKS = {
    "clean": {},
    "cinematic": {"ca": 0.003, "sharpen": 0.30, "vignette": 0.35},
    "film": {"ca": 0.004, "sharpen": 0.25, "vignette": 0.40, "grain": 0.030},
    "dreamy": {"glow": 0.38, "ca": 0.005, "vignette": 0.45, "grain": 0.015},
    "crisp": {"sharpen": 0.6, "vignette": 0.2},
}


def looks():
    """Names of the built-in looks."""
    return list(LOOKS)


def apply_look(frame, look="clean", seed=0):
    """Apply a named look (or a params dict) to a display-range [0,1] frame —
    a one-call grade composed from glow / chromatic_aberration / sharpen /
    vignette / film_grain. See `LOOKS` for the presets."""
    p = LOOKS.get(look, {}) if isinstance(look, str) else dict(look)
    out = np.clip(np.asarray(frame, np.float32), 0.0, 1.0)
    if p.get("glow", 0.0) > 0.0:
        r = max(3, int(min(out.shape[0], out.shape[1]) * 0.02))
        b = _box_blur(out, r)
        g = p["glow"]
        out = np.clip(out * (1.0 - g) + np.maximum(out, b) * g, 0.0, 1.0)
    if p.get("ca", 0.0) > 0.0:
        out = chromatic_aberration(out, p["ca"])
    if p.get("sharpen", 0.0) > 0.0:
        out = sharpen(out, p["sharpen"])
    if p.get("vignette", 0.0) > 0.0:
        out = vignette(out, p["vignette"])
    if p.get("grain", 0.0) > 0.0:
        out = film_grain(out, p["grain"], seed)
    return out
