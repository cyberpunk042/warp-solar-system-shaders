"""Geometry → triangle mesh — the bridge from turtle output to the Warp renderer.

Tessellates :class:`~warp_shaders.life.turtle.Geometry` into a single indexed
triangle mesh: each branch **segment** becomes a tapered `sides`-gon **tube**,
each **leaf** a small blade. Emits flat NumPy arrays (verts / indices / vertex
normals / vertex colours) ready to upload as a `wp.Mesh` (which builds a BVH for
per-pixel `wp.mesh_query_ray`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .turtle import Geometry


@dataclass
class Mesh:
    verts: np.ndarray      # (N, 3) float32
    indices: np.ndarray    # (M*3,) int32  (flattened triangles)
    normals: np.ndarray    # (N, 3) float32  (per-vertex)
    colors: np.ndarray     # (N, 3) float32  (per-vertex)

    @property
    def n_tris(self) -> int:
        return len(self.indices) // 3


def merge_meshes(meshes, offsets=None) -> "Mesh":
    """Concatenate several :class:`Mesh` into one (indices re-based).

    `offsets` (optional) is a list of ``(x, y, z)`` translations applied to each
    mesh's vertices — for placing many plants on a ground patch before uploading
    them as a single ``wp.Mesh``.
    """
    meshes = [m for m in meshes if m is not None and m.n_tris > 0]
    if not meshes:
        z = np.zeros((0, 3), np.float32)
        return Mesh(z, np.zeros(0, np.int32), z, z)
    verts, normals, colors, indices = [], [], [], []
    base = 0
    for k, m in enumerate(meshes):
        v = m.verts
        if offsets is not None:
            v = v + np.asarray(offsets[k], np.float32)
        verts.append(v)
        normals.append(m.normals)
        colors.append(m.colors)
        indices.append(m.indices.astype(np.int32) + base)
        base += m.verts.shape[0]
    return Mesh(np.concatenate(verts).astype(np.float32),
               np.concatenate(indices).astype(np.int32),
               np.concatenate(normals).astype(np.float32),
               np.concatenate(colors).astype(np.float32))


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _ortho_frame(d):
    """Two unit vectors spanning the plane perpendicular to unit `d`."""
    a = np.array([0.0, 0.0, 1.0], np.float32) if abs(d[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0], np.float32)
    u = _norm(np.cross(d, a))
    v = np.cross(d, u)
    return u, v


def build_mesh(geo: Geometry, sides: int = 6, min_radius: float = 0.004) -> Mesh:
    """Tessellate `geo` into a single indexed triangle mesh."""
    V, N, C, I = [], [], [], []

    def add_vert(p, n, c):
        V.append(p); N.append(n); C.append(c)
        return len(V) - 1

    # --- branch tubes ---
    ang = [2.0 * math.pi * k / sides for k in range(sides)]
    cos = [math.cos(a) for a in ang]
    sin = [math.sin(a) for a in ang]
    for s in geo.segments:
        d = s.p1 - s.p0
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        d = d / length
        u, v = _ortho_frame(d)
        r0 = max(s.r0, min_radius)
        r1 = max(s.r1, min_radius)
        base = len(V)
        for k in range(sides):
            rad = u * cos[k] + v * sin[k]          # outward radial (unit)
            add_vert((s.p0 + rad * r0).astype(np.float32), rad.astype(np.float32), s.color)
            add_vert((s.p1 + rad * r1).astype(np.float32), rad.astype(np.float32), s.color)
        for k in range(sides):
            k1 = (k + 1) % sides
            a = base + 2 * k        # ring0[k]
            b = base + 2 * k + 1    # ring1[k]
            c = base + 2 * k1 + 1   # ring1[k1]
            e = base + 2 * k1       # ring0[k1]
            I += [a, b, c, a, c, e]  # two tris, outward winding

    # --- leaves (a simple 4-vertex diamond blade) ---
    for lf in geo.leaves:
        h = _norm(lf.h)
        side = _norm(np.cross(h, lf.u)) * (lf.size * 0.32)
        nrm = _norm(np.cross(side, h)).astype(np.float32)
        tip = lf.pos + h * lf.size
        mid = lf.pos + h * (lf.size * 0.42)
        b0 = add_vert(lf.pos.astype(np.float32), nrm, lf.color)
        b1 = add_vert((mid - side).astype(np.float32), nrm, lf.color)
        b2 = add_vert((mid + side).astype(np.float32), nrm, lf.color)
        b3 = add_vert(tip.astype(np.float32), nrm, lf.color)
        I += [b0, b1, b3, b0, b3, b2]

    if not V:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros(0, np.int32),
                    np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    return Mesh(
        np.asarray(V, np.float32),
        np.asarray(I, np.int32),
        np.asarray(N, np.float32),
        np.asarray(C, np.float32),
    )
