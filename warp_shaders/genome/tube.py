"""Turn a real conserved backbone path (from the genome libs) into a solid tube **mesh** — so the DNA can be
ray-traced as a lit, opaque object through the engine's mesh raytracer (``life.render.render_plant``) instead
of drawn as loose points. The tube follows the actual lib positions; nothing is invented — it is the real
matter given a surface.
"""

from __future__ import annotations

import numpy as np

from ..life.mesh import Mesh


def _rmf(path: np.ndarray):
    """Rotation-minimising frame along a polyline (double-reflection method, Wang et al. 2008): a smooth
    (tangent, normal, binormal) per vertex with no twist flips. Returns (T, N, B) each (K,3)."""
    k = path.shape[0]
    t = np.zeros((k, 3), np.float64)
    t[:-1] = path[1:] - path[:-1]
    t[-1] = t[-2]
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(tl, 1e-9)

    n = np.zeros((k, 3), np.float64)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n[0] = np.cross(t[0], ref)
    n[0] /= np.linalg.norm(n[0]) + 1e-9
    for i in range(k - 1):
        v1 = path[i + 1] - path[i]
        c1 = np.dot(v1, v1)
        nl = n[i] - (2.0 / max(c1, 1e-12)) * np.dot(v1, n[i]) * v1
        tl_ = t[i] - (2.0 / max(c1, 1e-12)) * np.dot(v1, t[i]) * v1
        v2 = t[i + 1] - tl_
        c2 = np.dot(v2, v2)
        n[i + 1] = nl - (2.0 / max(c2, 1e-12)) * np.dot(v2, nl) * v2
        n[i + 1] /= np.linalg.norm(n[i + 1]) + 1e-9
    b = np.cross(t, n)
    return t, n, b


def tube_mesh(path: np.ndarray, radius, color, sides: int = 8, cap: bool = True) -> Mesh:
    """Build a closed tube of the given ``radius`` around ``path`` (K,3), ``sides``-gon cross-section.

    ``radius`` may be a scalar or a (K,) per-vertex array (so the strand can taper). ``color`` is a single
    RGB or a (K,3) per-vertex array. Returns a :class:`Mesh` (verts / indices / per-vertex normals+colors).
    """
    path = np.asarray(path, np.float64)
    rad_in = np.full(path.shape[0], float(radius)) if np.isscalar(radius) else np.asarray(radius, np.float64).reshape(-1)
    col_in = np.asarray(color, np.float32)
    if col_in.ndim == 1:
        col_in = np.tile(col_in, (path.shape[0], 1))

    # drop coincident consecutive points — zero-length segments make degenerate rings that break the BVH
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-5])
    path = path[keep]
    rad_in = rad_in[keep]
    col_in = col_in[keep]

    k = path.shape[0]
    _, N, B = _rmf(path)
    rad = rad_in
    col = col_in

    ang = (np.arange(sides) / sides) * 2.0 * np.pi
    ca, sa = np.cos(ang), np.sin(ang)                                     # (S,)
    # ring vertex = center + r*(cos*N + sin*B); normal = the same radial direction
    off = ca[None, :, None] * N[:, None, :] + sa[None, :, None] * B[:, None, :]   # (K,S,3)
    verts = path[:, None, :] + rad[:, None, None] * off
    normals = off
    verts = verts.reshape(-1, 3)
    normals = normals.reshape(-1, 3)
    colors = np.repeat(col, sides, axis=0)

    # side quads between consecutive rings -> two triangles each
    i0 = np.arange(k - 1)[:, None] * sides + np.arange(sides)[None, :]
    i1 = np.arange(k - 1)[:, None] * sides + (np.arange(sides)[None, :] + 1) % sides
    j0 = i0 + sides
    j1 = i1 + sides
    tris = np.stack([i0, j0, i1, i1, j0, j1], axis=-1).reshape(-1, 3)

    faces = [tris]
    if cap:
        c0 = verts.shape[0]
        c1 = c0 + 1
        verts = np.concatenate([verts, path[:1], path[-1:]], axis=0)
        normals = np.concatenate([normals, -_rmf(path)[0][:1], _rmf(path)[0][-1:]], axis=0)
        colors = np.concatenate([colors, col[:1], col[-1:]], axis=0)
        s = np.arange(sides)
        start_fan = np.stack([np.full(sides, c0), (s + 1) % sides, s], axis=-1)
        base = (k - 1) * sides
        end_fan = np.stack([np.full(sides, c1), base + s, base + (s + 1) % sides], axis=-1)
        faces += [start_fan, end_fan]

    indices = np.concatenate(faces, axis=0).astype(np.int32).reshape(-1)
    return Mesh(verts.astype(np.float32), indices, normals.astype(np.float32), colors.astype(np.float32))
