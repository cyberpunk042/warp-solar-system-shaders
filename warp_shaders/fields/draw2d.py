"""Host-side 2-D field helpers for the electromagnetism scenes.

Integrate field lines from a vector field and rasterise them (and glowing points) as
additive glow into an HDR numpy image. Screen-space and camera-free — the field-line
scenes (bar magnet, electric dipole, solenoid) share this.
"""

import numpy as np


def integrate_line(field_fn, start, dt, steps, bounds=2.4, stop_r=None, stops=None):
    """Trace one field line from ``start`` by Euler-integrating the normalised field.
    Stops on leaving ``bounds``, exceeding ``steps``, or coming within ``stop_r`` of any
    point in ``stops``. Returns an (N,2) array."""
    p = np.array(start, np.float64)
    pts = [p.copy()]
    for _ in range(steps):
        v = np.array(field_fn(p), np.float64)
        n = np.hypot(v[0], v[1])
        if n < 1e-9:
            break
        p = p + v / n * dt
        pts.append(p.copy())
        if abs(p[0]) > bounds or abs(p[1]) > bounds:
            break
        if stops is not None and stop_r is not None:
            done = False
            for s in stops:
                if np.hypot(p[0] - s[0], p[1] - s[1]) < stop_r:
                    done = True
            if done:
                break
    return np.array(pts, np.float32)


def _to_px(pts, W, H, ax):
    px = (pts[:, 0] / ax * 0.5 + 0.5) * W
    py = (0.5 - pts[:, 1] * 0.5) * H
    return px, py


def draw_polyline(hdr, pts, color, width_px, ax, glow=1.0):
    """Additively rasterise a glowing polyline (world coords) into ``hdr``."""
    H, W, _ = hdr.shape
    if len(pts) < 2:
        return
    px, py = _to_px(pts, W, H, ax)
    col = np.asarray(color, np.float32)
    w = float(width_px)
    for k in range(len(pts) - 1):
        x0, y0, x1, y1 = px[k], py[k], px[k + 1], py[k + 1]
        minx = max(int(min(x0, x1) - w * 3) - 1, 0)
        maxx = min(int(max(x0, x1) + w * 3) + 2, W)
        miny = max(int(min(y0, y1) - w * 3) - 1, 0)
        maxy = min(int(max(y0, y1) + w * 3) + 2, H)
        if maxx <= minx or maxy <= miny:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy + 1e-9
        t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / ll, 0.0, 1.0)
        d2 = (gx - (x0 + t * dx)) ** 2 + (gy - (y0 + t * dy)) ** 2
        g = np.exp(-d2 / (w * w)) * glow
        hdr[miny:maxy, minx:maxx] += col[None, None, :] * g[:, :, None]


def draw_point(hdr, world, color, radius_px, ax):
    """Additively splat a glowing point (world coords)."""
    H, W, _ = hdr.shape
    px = (world[0] / ax * 0.5 + 0.5) * W
    py = (0.5 - world[1] * 0.5) * H
    r = float(radius_px)
    minx = max(int(px - r * 3) - 1, 0); maxx = min(int(px + r * 3) + 2, W)
    miny = max(int(py - r * 3) - 1, 0); maxy = min(int(py + r * 3) + 2, H)
    if maxx <= minx or maxy <= miny:
        return
    gx, gy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
    g = np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (r * r))
    hdr[miny:maxy, minx:maxx] += np.asarray(color, np.float32)[None, None, :] * g[:, :, None]
