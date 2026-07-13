"""Ball-and-stick molecule renderer — atoms as CPK-coloured spheres, bonds as
grey sticks, sphere-traced as a signed-distance field with PBR-ish studio
lighting (key + fill + rim + ambient occlusion). See
``docs/research/22-chemistry-and-molecules.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.field import sd_capsule
from ..subatomic.render import orbit_camera

_GREY = wp.constant(wp.vec3(0.62, 0.63, 0.66))       # bond stick colour


@wp.func
def _map(p: wp.vec3, apos: wp.array(dtype=wp.vec3), arad: wp.array(dtype=float),
         na: int, ba: wp.array(dtype=int), bb: wp.array(dtype=int), nb: int,
         br: float) -> float:
    d = float(1.0e9)
    for k in range(na):
        d = wp.min(d, wp.length(p - apos[k]) - arad[k])
    for k in range(nb):
        d = wp.min(d, sd_capsule(p, apos[ba[k]], apos[bb[k]], br))
    return d


@wp.func
def _normal(p: wp.vec3, apos: wp.array(dtype=wp.vec3), arad: wp.array(dtype=float),
            na: int, ba: wp.array(dtype=int), bb: wp.array(dtype=int), nb: int,
            br: float) -> wp.vec3:
    e = 0.012
    dx = _map(p + wp.vec3(e, 0.0, 0.0), apos, arad, na, ba, bb, nb, br) \
        - _map(p - wp.vec3(e, 0.0, 0.0), apos, arad, na, ba, bb, nb, br)
    dy = _map(p + wp.vec3(0.0, e, 0.0), apos, arad, na, ba, bb, nb, br) \
        - _map(p - wp.vec3(0.0, e, 0.0), apos, arad, na, ba, bb, nb, br)
    dz = _map(p + wp.vec3(0.0, 0.0, e), apos, arad, na, ba, bb, nb, br) \
        - _map(p - wp.vec3(0.0, 0.0, e), apos, arad, na, ba, bb, nb, br)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, apos: wp.array(dtype=wp.vec3),
        arad: wp.array(dtype=float), na: int, ba: wp.array(dtype=int),
        bb: wp.array(dtype=int), nb: int, br: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for i in range(5):
        h = 0.02 + 0.12 * float(i)
        d = _map(p + n * h, apos, arad, na, ba, bb, nb, br)
        occ = occ + (h - d) * sca
        sca = sca * 0.7
    return wp.clamp(1.0 - 1.5 * occ, 0.0, 1.0)


@wp.func
def _hitcol(p: wp.vec3, apos: wp.array(dtype=wp.vec3), arad: wp.array(dtype=float),
            acol: wp.array(dtype=wp.vec3), na: int, ba: wp.array(dtype=int),
            bb: wp.array(dtype=int), nb: int, br: float) -> wp.vec3:
    abest = float(1.0e9)
    ci = int(0)
    for k in range(na):
        dd = wp.length(p - apos[k]) - arad[k]
        if dd < abest:
            abest = dd
            ci = k
    bbest = float(1.0e9)
    for k in range(nb):
        bbest = wp.min(bbest, sd_capsule(p, apos[ba[k]], apos[bb[k]], br))
    if bbest < abest:
        return _GREY
    return acol[ci]


@wp.kernel
def mol_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
               apos: wp.array(dtype=wp.vec3), arad: wp.array(dtype=float),
               acol: wp.array(dtype=wp.vec3), na: int, ba: wp.array(dtype=int),
               bb: wp.array(dtype=int), nb: int, br: float, bound: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # studio background gradient
    bg = wp.vec3(0.03, 0.035, 0.05) + wp.vec3(0.05, 0.06, 0.08) * (0.5 + 0.5 * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(96):
        p = ro + rd * t
        d = _map(p, apos, arad, na, ba, bb, nb, br)
        if d < 0.001:
            hit = 1
            break
        t = t + d
        if t > bound:
            break

    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _normal(p, apos, arad, na, ba, bb, nb, br)
        base = _hitcol(p, apos, arad, acol, na, ba, bb, nb, br)
        ao = _ao(p, n, apos, arad, na, ba, bb, nb, br)
        key = wp.normalize(wp.vec3(0.6, 0.8, 0.5))
        fill = wp.normalize(wp.vec3(-0.7, 0.2, 0.4))
        ndl = wp.max(wp.dot(n, key), 0.0)
        ndf = wp.max(wp.dot(n, fill), 0.0)
        h = wp.normalize(key - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 40.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        lit = base * (0.18 * ao + 0.9 * ndl + 0.3 * ndf)
        lit = lit + wp.vec3(1.0, 1.0, 1.0) * (spec * 0.6)
        lit = lit + base * (fres * 0.5)
        col = lit

    img[i, j] = col


def render_molecule(width, height, time, mouse, device, atoms, bonds,
                    bond_r=0.12, dist=None, fov=38.0, exposure=1.05, el0=0.42):
    """atoms: list of (pos3, radius, color3); bonds: list of (i, j)."""
    apos = np.array([a[0] for a in atoms], np.float32).reshape(-1, 3)
    arad = np.array([a[1] for a in atoms], np.float32)
    acol = np.array([a[2] for a in atoms], np.float32).reshape(-1, 3)
    if bonds:
        ba = np.array([b[0] for b in bonds], np.int32)
        bb = np.array([b[1] for b in bonds], np.int32)
    else:
        ba = np.zeros(1, np.int32)
        bb = np.zeros(1, np.int32)
    extent = float(np.max(np.linalg.norm(apos, axis=1)) + arad.max()) if len(apos) else 2.0
    if dist is None:
        dist = extent * 3.2 + 2.0
    cam = orbit_camera(width, height, time, mouse, dist=dist, fov=fov, el0=el0)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(mol_kernel, dim=(height, width),
              inputs=[img, cam, wp.array(apos, dtype=wp.vec3, device=device),
                      wp.array(arad, dtype=wp.float32, device=device),
                      wp.array(acol, dtype=wp.vec3, device=device), int(len(atoms)),
                      wp.array(ba, dtype=wp.int32, device=device),
                      wp.array(bb, dtype=wp.int32, device=device), int(len(bonds)),
                      float(bond_r), float(dist + extent + 2.0),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.5, strength=0.25,
                     radius=max(2, int(min(width, height) * 0.008)), passes=2)
    return post.tonemap(hdr, mode="aces", exposure=exposure)
