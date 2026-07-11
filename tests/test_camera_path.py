"""Tests for engine.camera_path — easing + keyframed camera interpolation.

Run: `python -m tests.test_camera_path` (or under pytest). Pure host maths.
"""

import math

import numpy as np
import warp as wp

from warp_shaders.engine import camera_path as CP

wp.init()


def test_easings():
    for name, fn in CP.EASINGS.items():
        assert abs(fn(0.0)) < 1e-6 and abs(fn(1.0) - 1.0) < 1e-6, name
        # monotone non-decreasing on [0,1]
        xs = np.linspace(0, 1, 21)
        ys = [fn(x) for x in xs]
        assert all(b >= a - 1e-6 for a, b in zip(ys, ys[1:])), name
        assert all(-1e-6 <= y <= 1.0 + 1e-6 for y in ys), name
    # smoothstep is symmetric about 0.5
    assert abs(CP.smoothstep(0.5) - 0.5) < 1e-6
    # clamps outside [0,1]
    assert CP.smoothstep(-1.0) == 0.0 and CP.smoothstep(2.0) == 1.0


def test_dolly_endpoints():
    p = CP.dolly((0.0, 0.0, 5.0), (0.0, 0.0, 2.0), target=(0.0, 0.0, 0.0),
                 fov0=50.0, fov1=30.0)
    e0, t0, f0 = p.sample(0.0)
    e1, t1, f1 = p.sample(1.0)
    assert np.allclose(e0, [0, 0, 5]) and np.allclose(e1, [0, 0, 2])
    assert abs(f0 - 50.0) < 1e-4 and abs(f1 - 30.0) < 1e-4
    # midpoint is between the endpoints along z, fov between 30 and 50
    em, _, fm = p.sample(0.5)
    assert 2.0 < em[2] < 5.0 and 30.0 < fm < 50.0


def test_orbit_radius_and_look():
    center = (0.0, 0.0, 0.0)
    r, elev = 6.0, 0.25
    p = CP.orbit(center, radius=r, elevation=elev, turns=1.0, samples=16)
    for t in np.linspace(0, 1, 25):
        eye, target, fov = p.sample(t)
        assert np.allclose(target, center)
        # spline stays close to the orbit radius (Catmull-Rom bows in slightly)
        assert abs(np.linalg.norm(eye) - r) < 0.25 * r
        assert abs(eye[1] - r * math.sin(elev)) < 0.2 * r    # ~constant elevation


def test_camera_builder():
    p = CP.orbit(radius=4.0, samples=8)
    cam = p.camera(0.3, aspect=16.0 / 9.0)
    assert abs(cam.aspect - 16.0 / 9.0) < 1e-4
    for v in (cam.forward, cam.right, cam.up):
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        assert abs(n - 1.0) < 1e-4                            # orthonormal basis


if __name__ == "__main__":
    test_easings()
    print("  easings (endpoints, monotone, clamp): OK")
    test_dolly_endpoints()
    print("  dolly endpoints + fov interpolation: OK")
    test_orbit_radius_and_look()
    print("  orbit radius + look-at + elevation: OK")
    test_camera_builder()
    print("  camera() builder (aspect + orthonormal basis): OK")
    print("ALL PASSED")
