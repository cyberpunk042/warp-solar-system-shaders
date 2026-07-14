"""Tests for C1 merge->cube (warp_compress.mergecube) — dedup + location index, lossless.

Operator spec: merge the same thing together, keep digits for the locations of the various same
elements (the growing "cube part" = the index grid).

  1. exact lossless round-trip: decompress(compress(x)) == x;
  2. merging actually happens (unique pieces << total blocks) and it compresses (ratio > 1);
  3. the location index is a 3-D cube of digits that names each unique piece;
  4. on a highly-repetitive card, more repeats -> better ratio (the merge earns its keep).

    python -m tests.test_mergecube
"""

import numpy as np

from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card


def main():
    occ = sample_card()

    # 1. exact lossless round-trip on the real board
    unique, index, meta = mc.compress(occ, block=5)
    back = mc.decompress(unique, index, meta)
    assert np.array_equal(back, occ), "merge->cube round-trip not lossless"
    print(f"  lossless round-trip: OK  (board {occ.shape} exact)")

    # 2. merging happens and it compresses
    n_blocks = int(index.size)
    n_unique = int(len(unique))
    r = mc.ratio(occ, unique, index, meta)
    assert n_unique < n_blocks, f"nothing merged ({n_unique} unique of {n_blocks})"
    assert r > 1.0, f"did not compress (ratio {r:.2f})"
    print(f"  merge + compress: OK  ({n_blocks} blocks -> {n_unique} unique, {r:.1f}x lossless)")

    # 3. the location index is a 3-D cube of digits naming each unique piece
    assert index.ndim == 3, "location index is not a 3-D cube"
    assert index.min() >= 0 and index.max() < n_unique, "index digits out of range of the dictionary"
    print(f"  location-index cube: OK  (shape {index.shape}, digits 0..{index.max()})")

    # 4. more repetition -> better ratio: a synthetic card tiled from one block compresses far more
    tile = (np.random.default_rng(0).random((5, 5, 5)) > 0.5).astype(np.uint8)
    repetitive = np.tile(tile, (8, 2, 6))                       # the SAME element, many locations
    u2, i2, m2 = mc.compress(repetitive, block=5)
    r2 = mc.ratio(repetitive, u2, i2, m2)
    assert np.array_equal(mc.decompress(u2, i2, m2), repetitive), "repetitive round-trip failed"
    assert len(u2) <= 2 and r2 > r, f"repetition not exploited (unique {len(u2)}, ratio {r2:.1f})"
    print(f"  repetition -> better ratio: OK  (1 unique of {i2.size}, {r2:.0f}x)")

    # 5. the warp_scan_merge scene (C1 as a process) renders on the real board and animates
    import warp as wp
    import warp_shaders as ws
    wp.init()
    a = np.asarray(ws.render("warp_scan_merge", width=120, height=96, time=2.0), np.float32)  # scanning
    b = np.asarray(ws.render("warp_scan_merge", width=120, height=96, time=7.5), np.float32)  # merged
    assert np.all(np.isfinite(a)) and a.max() > 0.1 and a.std() > 0.01, "warp_scan_merge: bad frame"
    assert np.abs(a - b).mean() > 1e-3, "warp_scan_merge: scan/merge did not animate"
    print("  scene warp_scan_merge: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
