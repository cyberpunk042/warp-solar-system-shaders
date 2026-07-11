"""Portable texture / map sampling over wp.array2d (CPU + CUDA).

Warp 1.15's hardware `wp.Texture` lacks a usable `texture_sample` on CPU, so the
engine samples plain `wp.array2d(dtype=wp.vec3)` maps with a manual bilinear
`@wp.func`. Same results on every device. Includes an equirectangular sampler
(longitude wrap, latitude clamp) for planet maps and a PIL image loader so a real
NASA Blue-Marble JPG drops straight in. See docs/research/01-textures-and-luts.md.
"""

import numpy as np
import warp as wp

_TWO_PI = 6.28318530718
_PI = 3.14159265


@wp.func
def sample2d(tex: wp.array2d(dtype=wp.vec3), u: float, v: float,
             wrap_x: int, wrap_y: int) -> wp.vec3:
    """Bilinear sample of an (H,W) vec3 map at uv in [0,1].
    wrap_* = 1 -> wrap (repeat), 0 -> clamp to edge."""
    h = tex.shape[0]
    w = tex.shape[1]
    x = u * float(w) - 0.5
    y = v * float(h) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)

    if wrap_x == 1:
        x0a = ((x0 % w) + w) % w
        x1a = (((x0 + 1) % w) + w) % w
    else:
        x0a = wp.min(wp.max(x0, 0), w - 1)
        x1a = wp.min(wp.max(x0 + 1, 0), w - 1)
    if wrap_y == 1:
        y0a = ((y0 % h) + h) % h
        y1a = (((y0 + 1) % h) + h) % h
    else:
        y0a = wp.min(wp.max(y0, 0), h - 1)
        y1a = wp.min(wp.max(y0 + 1, 0), h - 1)

    c00 = tex[y0a, x0a]
    c10 = tex[y0a, x1a]
    c01 = tex[y1a, x0a]
    c11 = tex[y1a, x1a]
    a = c00 * (1.0 - fx) + c10 * fx
    b = c01 * (1.0 - fx) + c11 * fx
    return a * (1.0 - fy) + b * fy


@wp.func
def sample3d(vol: wp.array3d(dtype=float), u: float, v: float, w: float,
             wrap: int) -> float:
    """Trilinear sample of a scalar 3D volume at (u,v,w) in [0,1].
    Axis order vol[z, y, x]; wrap=1 repeats, 0 clamps (all axes)."""
    dz = vol.shape[0]
    dy = vol.shape[1]
    dx = vol.shape[2]
    x = u * float(dx) - 0.5
    y = v * float(dy) - 0.5
    z = w * float(dz) - 0.5
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    z0 = int(wp.floor(z))
    fx = x - float(x0)
    fy = y - float(y0)
    fz = z - float(z0)

    if wrap == 1:
        x0a = ((x0 % dx) + dx) % dx
        x1a = (((x0 + 1) % dx) + dx) % dx
        y0a = ((y0 % dy) + dy) % dy
        y1a = (((y0 + 1) % dy) + dy) % dy
        z0a = ((z0 % dz) + dz) % dz
        z1a = (((z0 + 1) % dz) + dz) % dz
    else:
        x0a = wp.min(wp.max(x0, 0), dx - 1)
        x1a = wp.min(wp.max(x0 + 1, 0), dx - 1)
        y0a = wp.min(wp.max(y0, 0), dy - 1)
        y1a = wp.min(wp.max(y0 + 1, 0), dy - 1)
        z0a = wp.min(wp.max(z0, 0), dz - 1)
        z1a = wp.min(wp.max(z0 + 1, 0), dz - 1)

    c000 = vol[z0a, y0a, x0a]
    c100 = vol[z0a, y0a, x1a]
    c010 = vol[z0a, y1a, x0a]
    c110 = vol[z0a, y1a, x1a]
    c001 = vol[z1a, y0a, x0a]
    c101 = vol[z1a, y0a, x1a]
    c011 = vol[z1a, y1a, x0a]
    c111 = vol[z1a, y1a, x1a]
    x00 = c000 * (1.0 - fx) + c100 * fx
    x10 = c010 * (1.0 - fx) + c110 * fx
    x01 = c001 * (1.0 - fx) + c101 * fx
    x11 = c011 * (1.0 - fx) + c111 * fx
    y0i = x00 * (1.0 - fy) + x10 * fy
    y1i = x01 * (1.0 - fy) + x11 * fy
    return y0i * (1.0 - fz) + y1i * fz


@wp.func
def sample_equirect(tex: wp.array2d(dtype=wp.vec3), dir: wp.vec3) -> wp.vec3:
    """Sample an equirectangular map by a unit direction (top row = +Y pole)."""
    lon = wp.atan2(dir[2], dir[0])
    lat = wp.asin(wp.clamp(dir[1], -1.0, 1.0))
    u = lon / _TWO_PI + 0.5
    v = 0.5 - lat / _PI
    return sample2d(tex, u, v, 1, 0)


# ---- host helpers ----------------------------------------------------------

def to_texture(arr, device="cpu"):
    """Upload an (H,W,3) float array to a wp.array2d(dtype=wp.vec3)."""
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    return wp.array2d(a, dtype=wp.vec3, device=device)


def load_equirect(path, device="cpu", srgb_to_linear=True):
    """Load an image (e.g. NASA Blue Marble) as a vec3 map. sRGB -> linear."""
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    if srgb_to_linear:
        im = np.power(im, 2.2)
    return to_texture(im, device)
