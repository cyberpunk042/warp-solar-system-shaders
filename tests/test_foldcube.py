"""Tests for C2 fold->cube (warp_compress.foldcube) — the collision-agnostic 20x fold.

Operator hard requirement: fold the real card and squish it just right into a cube ~20x smaller by
total surface, collisions merged (not avoided), the image built in the process.

  1. the fold reaches >= 20x surface compression on the REAL board occupancy;
  2. the result is cube-ish (balanced dimensions), not a thin sliver;
  3. it is built in the process (a recorded sequence of in-half folds), collision-agnostic (OR-merge);
  4. the folded cube is denser than the original (the squish concentrates the material).

    python -m tests.test_foldcube
"""

import numpy as np

from warp_compress import foldcube as fc


def main():
    occ = fc.sample_card()
    assert occ.sum() > 0, "board occupancy empty"
    dens0 = occ.mean()

    r = fc.compress(occ, target=20.0)
    print(f"  board {r['orig_shape']} surface {r['orig_surface']} -> cube {r['cube_shape']} "
          f"surface {r['cube_surface']}")

    # 1. >= 20x surface compression
    assert r["ratio"] >= 20.0, f"surface compression {r['ratio']:.1f}x < 20x"
    print(f"  20x surface compression: OK  ({r['ratio']:.1f}x in {len(r['folds'])} folds)")

    # 2. cube-ish (balanced dims), not a sliver
    mx, mn = max(r["cube_shape"]), min(r["cube_shape"])
    assert mx / mn <= 2.0, f"not cube-ish (max/min dim {mx/mn:.2f})"
    print(f"  cube-ish shape: OK  (max/min dim {mx/mn:.2f})")

    # 3. built in the process: a real sequence of half-folds, each collision-agnostic (OR-merge)
    assert len(r["folds"]) >= 3, "too few folds to be a real squish"
    # verify the OR-merge property directly: fold of a card = OR of its two reflected halves
    a = (np.arange(8) % 3 == 0).astype(np.uint8)[:, None, None] * np.ones((8, 2, 2), np.uint8)
    folded = fc._fold_axis(a, 0)
    manual = np.maximum(a[:4], np.flip(a[4:], axis=0))
    assert np.array_equal(folded, manual), "fold is not a collision-agnostic OR-merge"
    print(f"  built-in-process OR-merge: OK  (folds {r['folds']})")

    # 4. the squish concentrates material: the cube is denser than the flat board
    dens1 = r["cube"].mean()
    assert dens1 > dens0, f"cube not denser than board (board {dens0:.3f}, cube {dens1:.3f})"
    print(f"  squish concentrates: OK  (density {dens0:.3f} -> {dens1:.3f})")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
