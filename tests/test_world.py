"""Tests for life.render.render_world — the multi-sun surface renderer.

Run: `python -m tests.test_world` (or under pytest). Grows a tiny meadow and
renders it under one and two suns, checking the image is finite, in-range, and
non-trivial (something was lit).
"""

import numpy as np
import warp as wp

from warp_shaders.life import plants as _plants
from warp_shaders.life.mesh import merge_meshes
from warp_shaders.life.render import render_world

wp.init()


def _meadow():
    specs = ["grass", "herb", "flower"]
    meshes, offs = [], []
    for k, name in enumerate(specs):
        spec = _plants.get_spec(name)
        mesh, _b = _plants.grow_mesh(spec, 4)
        if mesh.n_tris:
            meshes.append(mesh)
            offs.append((float(k) - 1.0, 0.0, 0.0))
    return merge_meshes(meshes, offs)


def _render(suns):
    field = _meadow()
    assert field.n_tris > 0
    return render_world(field, 96, 72, (0.0, 0.8, 6.0), (0.0, 0.8, 0.0),
                        suns, device="cpu")


def test_one_sun():
    img = _render([((0.4, 0.5, -0.5), (1.0, 0.9, 0.7), 1.0)])
    assert img.shape == (72, 96, 3)
    assert np.all(np.isfinite(img))
    assert img.min() >= 0.0
    assert img.max() > 0.05                       # something is lit


def test_two_suns_brighter():
    # adding a second sun only adds light -> mean brightness should not drop
    one = _render([((0.4, 0.5, -0.5), (1.0, 0.9, 0.7), 1.0)])
    two = _render([((0.4, 0.5, -0.5), (1.0, 0.9, 0.7), 1.0),
                   ((-0.4, 0.5, -0.5), (0.7, 0.8, 1.0), 0.8)])
    assert np.all(np.isfinite(two))
    assert two.mean() >= one.mean() - 1e-3


if __name__ == "__main__":
    test_one_sun()
    print("  render_world one sun (finite, in-range, lit): OK")
    test_two_suns_brighter()
    print("  render_world two suns (adds light): OK")
    print("ALL PASSED")
