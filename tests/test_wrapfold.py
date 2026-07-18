"""Tests for the fold-and-merge codec (warp_compress.wrapfold).

The mechanism: wrap the strand onto the period the codec finds, merge cells that match the coil
below, recurse into nested coils. Verifies exact lossless round-trips, that the found period is
the real one, that folding actually compresses periodic data, and that the lossy dial stays within
its error bound.

    python -m tests.test_wrapfold
"""

import numpy as np

from warp_compress import wrapfold as wf


def main():
    # 1. period detection finds the true period of a periodic strand
    sym = np.array(list(b"ACGT" * 40), np.int32)
    p, score = wf.best_period(sym, tol=0)
    assert p in (4, 8, 12), f"expected a multiple-of-4 period, got {p}"
    assert score > 0.9, f"agreement should be near 1 on strict periodic data ({score:.2f})"
    print(f"  period detection: OK  (found {p}, agreement {score:.2f})")

    # 2. fold_levels + unfold_levels is exact, and it merges most of the strand
    core, levels = wf.fold_levels(sym, tol=0)
    back = wf.unfold_levels(core, levels)
    assert np.array_equal(back, sym), "unfold != identity"
    merged = sum(L.merged for L in levels)
    assert merged > 0.8 * len(sym), f"folding merged too little ({merged}/{len(sym)})"
    print(f"  fold/unfold exact: OK  ({len(levels)} levels, {merged}/{len(sym)} merged)")

    # 3. full codec round-trips exactly on many kinds of data (incl. empty/random)
    cases = {
        "periodic": bytes(list(range(8)) * 60),
        "dna": b"ACGTACGT" * 64,
        "text": b"the chromosome wraps " * 24,
        "sine": bytes(int(128 + 100 * np.sin(i * 2 * np.pi / 16)) & 255 for i in range(512)),
        "random": bytes(np.random.default_rng(1).integers(0, 256, 500).tolist()),
        "empty": b"",
        "tiny": b"Q",
    }
    for name, data in cases.items():
        blob = wf.compress(data, mode="lossless")
        assert wf.decompress(blob) == data, f"lossless round-trip failed: {name}"
    print(f"  lossless codec: OK  ({len(cases)} inputs)")

    # 4. periodic data actually compresses
    data = b"ACGTACGT" * 64
    ratio = len(data) / len(wf.compress(data, mode="lossless"))
    assert ratio > 4.0, f"periodic data barely compressed (ratio {ratio:.1f})"
    print(f"  fold compresses: OK  (dna {ratio:.1f}x)")

    # 5. lossy: reconstruction stays within the tolerance, and shrinks the blob
    rng = np.random.default_rng(2)
    noisy = bytes((int(128 + 90 * np.sin(i * 2 * np.pi / 16)) + int(rng.integers(-6, 6))) & 255
                  for i in range(1500))
    b_lossless = wf.compress(noisy, mode="lossless")
    b_lossy = wf.compress(noisy, mode="lossy", tol=12)
    back = wf.decompress(b_lossy)
    assert len(back) == len(noisy)
    max_err = max(abs(a - b) for a, b in zip(noisy, back))
    assert max_err <= 12, f"lossy error {max_err} exceeds tol"
    assert len(b_lossy) <= len(b_lossless), "lossy did not shrink the blob"
    print(f"  lossy dial: OK  (lossless {len(b_lossless)} -> lossy {len(b_lossy)}, max_err {max_err})")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
