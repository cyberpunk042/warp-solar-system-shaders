"""Frame output — LDR (PNG) and true **HDR** containers, plus a `RenderTarget`.

Scenes render to an ``(H, W, 3)`` float32 buffer. For display you tonemap and
quantize to 8-bit PNG, but the linear buffer often carries values well above 1
(stars, suns, bloom sources) — throwing that away loses the data a compositor or
a different tonemap would want. This module keeps it:

- :func:`save_png` — tonemap-free 8-bit write (expects a display-range [0,1]
  frame; clamps).
- :func:`save_npy` — the raw float32 array, lossless (NumPy ``.npy``).
- :func:`save_hdr` / :func:`load_hdr` — Radiance **RGBE** ``.hdr`` (Ward 1991):
  a shared 8-bit exponent per pixel gives ~76 dB of float range in 4 bytes,
  readable by every compositor / DCC tool. Written uncompressed (flat scanlines).
- :class:`RenderTarget` — wraps one HDR frame and ``.save(path)`` dispatches by
  extension (``.png`` / ``.npy`` / ``.hdr``), tonemapping only for PNG.

No new dependencies — RGBE encode/decode is a few NumPy ops.
"""

from __future__ import annotations

import numpy as np


def save_npy(path: str, frame) -> None:
    """Write the raw linear frame as a lossless float32 ``.npy``."""
    np.save(path, np.asarray(frame, np.float32))


def save_png(path: str, frame) -> None:
    """Write a display-range ``[0, 1]`` frame as 8-bit PNG (clamps; no tonemap)."""
    from PIL import Image
    u8 = (np.clip(np.asarray(frame, np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def save_hdr(path: str, frame) -> None:
    """Write an HDR frame as a Radiance RGBE ``.hdr`` (uncompressed scanlines).

    Each pixel stores R,G,B mantissa bytes plus a shared exponent byte, so the
    linear range above 1.0 survives — the point of an HDR container."""
    rgb = np.maximum(np.asarray(frame, np.float32), 0.0)
    h, w = rgb.shape[:2]
    brightest = rgb.max(axis=2)
    mant, expo = np.frexp(brightest)              # brightest = mant * 2**expo, mant∈[0.5,1)
    valid = brightest > 1e-32
    # scale so the brightest channel maps into [128, 256)
    scale = np.where(valid, mant * 256.0 / np.maximum(brightest, 1e-32), 0.0)
    rgbe = np.zeros((h, w, 4), np.uint8)
    rgbe[..., 0] = np.clip(rgb[..., 0] * scale, 0, 255).astype(np.uint8)
    rgbe[..., 1] = np.clip(rgb[..., 1] * scale, 0, 255).astype(np.uint8)
    rgbe[..., 2] = np.clip(rgb[..., 2] * scale, 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(valid, np.clip(expo + 128, 0, 255), 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {h} +X {w}\n".encode("ascii"))
        rgbe.tofile(f)


def load_hdr(path: str) -> np.ndarray:
    """Read a Radiance RGBE ``.hdr`` back to an ``(H, W, 3)`` float32 array
    (uncompressed flat-scanline files, as written by :func:`save_hdr`)."""
    with open(path, "rb") as f:
        assert f.readline().startswith(b"#?"), "not a Radiance HDR file"
        w = h = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("truncated HDR header")
            s = line.strip()
            if s.startswith(b"-Y"):
                parts = s.split()
                h, w = int(parts[1]), int(parts[3])
                break
        rgbe = np.frombuffer(f.read(h * w * 4), np.uint8).reshape(h, w, 4)
    e = rgbe[..., 3].astype(np.int32)
    fexp = np.where(e > 0, np.ldexp(1.0, e - (128 + 8)), 0.0).astype(np.float32)
    rgb = (rgbe[..., :3].astype(np.float32) + 0.5) * fexp[..., None]
    return rgb


class RenderTarget:
    """A rendered HDR frame + one-call save that picks the container by extension.

    ``.png`` tonemaps (ACES by default) then writes 8-bit; ``.npy`` / ``.hdr``
    keep the full linear range. Construct from a scene's ``render()`` output::

        rt = RenderTarget(ws.render("neutron_star", width=1280, height=720))
        rt.save("frame.png")     # display
        rt.save("frame.hdr")     # full HDR for compositing
    """

    def __init__(self, hdr):
        self.hdr = np.asarray(hdr, np.float32)

    @property
    def shape(self):
        return self.hdr.shape

    def tonemapped(self, mode: str = "aces", exposure: float = 1.0) -> np.ndarray:
        """Return a display-range ``[0, 1]`` frame via the post tonemapper."""
        from . import post
        return post.tonemap(self.hdr, mode=mode, exposure=exposure)

    def save(self, path: str, tonemap: str = "aces", exposure: float = 1.0) -> None:
        """Write to `path`; the extension chooses the container.

        ``.hdr`` / ``.npy`` store the raw linear buffer; anything else is treated
        as LDR and gets tonemapped first (unless the buffer already looks like it
        is in display range, in which case it is written as-is)."""
        low = path.lower()
        if low.endswith(".hdr"):
            save_hdr(path, self.hdr)
        elif low.endswith(".npy"):
            save_npy(path, self.hdr)
        else:
            frame = self.hdr
            if float(frame.max()) > 1.0001:       # still linear/HDR -> tonemap for display
                frame = self.tonemapped(tonemap, exposure)
            save_png(path, frame)
