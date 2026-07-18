"""C2 — fold → cube: collision-agnostic fold-and-squish of the real card into a ~20x smaller cube.

Operator spec (2026-07-14, verbatim): *"just fold it and dont care about the collision, it build the
compression image in the process and not at the result, a bit like docker image ... you have to fold it
right and squish it just right and you need a really 20x smaller cube of the total surface of the whole
item of compression."*

This is the folding compression, applied to the **real RTX board**. The card is sampled to a 3-D
occupancy grid (from `gpu_board.board_map`), then folded in half, and in half again, on its longest
axis each time — but **collisions are not avoided: the two halves MERGE** (logical OR) where they
overlap. That merge is what pushes past the lossless folding limit (a thin sheet folded losslessly only
condenses ~2-3x by surface) down to a compact **cube ~20x smaller by total surface**. The compressed
image is **built in the process** (each fold overlays onto the growing block, Docker-layer style), not
computed at the end.

"Fold it right, squish it just right" = fold the longest axis each step (so the block stays balanced
toward a cube) until the surface-compression target is met. `total surface` = the exposed-face count of
the occupied voxels (Σ 6·occ − 2·shared-faces). Verified in ``tests/test_foldcube.py``.
"""

import numpy as np
import warp as wp

from warp_shaders.scenes.gpu_board import board_map

# board bounding box (board-local), a touch above/below the PCB to catch components + copper
_BB = (-3.7, 3.7, -0.14, 0.30, -1.5, 1.5)


@wp.kernel
def _sample_kernel(occ: wp.array3d(dtype=wp.int32), nx: int, ny: int, nz: int,
                   x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    i, j, k = wp.tid()
    x = x0 + (x1 - x0) * (float(i) + 0.5) / float(nx)
    y = y0 + (y1 - y0) * (float(j) + 0.5) / float(ny)
    z = z0 + (z1 - z0) * (float(k) + 0.5) / float(nz)
    d = board_map(wp.vec3(x, y, z))
    if d < 0.0:
        occ[i, j, k] = 1
    else:
        occ[i, j, k] = 0


def sample_card(nx=140, ny=12, nz=58, device="cpu"):
    """Voxel occupancy of the real board (1 = inside the card's SDF)."""
    wp.init()
    occ = wp.zeros((nx, ny, nz), dtype=wp.int32, device=device)
    x0, x1, y0, y1, z0, z1 = _BB
    wp.launch(_sample_kernel, dim=(nx, ny, nz),
              inputs=[occ, nx, ny, nz, x0, x1, y0, y1, z0, z1], device=device)
    wp.synchronize_device(device)
    return occ.numpy().astype(np.uint8)


def surface(occ):
    """Total exposed surface of the occupied voxels = Σ 6·occ − 2·(shared occupied faces)."""
    n = int(occ.sum())
    shared = 0
    for ax in range(3):
        a = np.take(occ, range(0, occ.shape[ax] - 1), axis=ax)
        b = np.take(occ, range(1, occ.shape[ax]), axis=ax)
        shared += int(np.sum((a > 0) & (b > 0)))
    return 6 * n - 2 * shared


def _fold_axis(occ, axis):
    """Fold in half along ``axis``, MERGING the two halves (collision-agnostic OR overlay)."""
    n = occ.shape[axis]
    if n % 2:                                        # drop the odd middle slice so halves align
        occ = np.take(occ, range(0, n - 1), axis=axis)
        n -= 1
    h = n // 2
    a = np.take(occ, range(0, h), axis=axis)
    b = np.flip(np.take(occ, range(h, n), axis=axis), axis=axis)   # reflect the far half over
    return np.maximum(a, b)                          # OR-merge: occupied if either half is


def fold_to_cube(occ, target=20.0, max_folds=20):
    """Fold-and-squish the longest axis repeatedly (OR-merge) until the surface compression reaches
    ``target``. Returns (cube, ratio, fold_axes) — fold_axes is the process record."""
    s0 = surface(occ)
    cur = occ.copy()
    axes = []
    for _ in range(max_folds):
        ratio = s0 / max(surface(cur), 1)
        if ratio >= target:
            break
        ax = int(np.argmax(cur.shape))               # fold the longest axis -> stays balanced (cubic)
        cur = _fold_axis(cur, ax)
        axes.append(ax)
    return cur, s0 / max(surface(cur), 1), axes


def compress(occ=None, target=20.0):
    """Convenience: sample the card if needed, fold to the target, return a report dict."""
    if occ is None:
        occ = sample_card()
    s0 = surface(occ)
    cube, ratio, axes = fold_to_cube(occ, target=target)
    return {
        "orig_shape": occ.shape,
        "cube_shape": cube.shape,
        "orig_surface": s0,
        "cube_surface": surface(cube),
        "ratio": ratio,
        "folds": axes,
        "cube": cube,
    }
