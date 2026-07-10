"""Host-side post-processing over the HDR framebuffer (NumPy).

Tonemapping (ACES / Reinhard / AgX-ish), threshold bloom (integral-image box blur
≈ Gaussian, no SciPy dependency), and vignette. Runs on the (H,W,3) float HDR
buffer a render kernel produces, returning display-ready [0,1] gamma-encoded RGB.

Sources (docs/research): Narkowicz 2016 (ACES fit); Reinhard et al. 2002;
Troy Sobotka AgX (approximation here).
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


def vignette(frame, amount=0.35):
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx / (w - 1) - 0.5)
    dy = (yy / (h - 1) - 0.5)
    r2 = (dx * dx + dy * dy) * 2.0
    v = (1.0 - amount * r2)[..., None]
    return frame * np.clip(v, 0.0, 1.0)
