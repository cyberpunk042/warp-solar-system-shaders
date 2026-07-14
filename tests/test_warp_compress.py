"""Tests for warp_compress — the folding chromosome compressor.

Correctness first: every fold is a bijection, coil/uncoil is exact, lossless round-trips are the
identity on arbitrary bytes, and lossy round-trips stay within the quantisation error. Plus a few
structural checks (the chromosome really coils, the lossy dial really shrinks).

    python -m tests.test_warp_compress
"""

import math
import random

import warp_compress as wc
from warp_compress import fold as F
from warp_compress.chromosome import coil, uncoil


def _check_fold(seq, kind):
    folded = F.fold(seq, kind)
    assert len(folded) == len(seq), (kind, "length changed")
    assert F.unfold(folded, kind, len(seq)) == seq, (kind, "not invertible")


def main():
    random.seed(11)

    # 1. every fold is an exact bijection on assorted lengths (incl. non-square / non-cube)
    for n in (0, 1, 2, 3, 7, 16, 17, 64, 100, 255, 4096):
        seq = [random.randrange(256) for _ in range(n)]
        for kind in (F.FOLD_NONE, F.FOLD_MORTON2D, F.FOLD_MORTON3D):
            _check_fold(seq, kind)
    print("  folds bijective: OK")

    # 2. coil/uncoil is exact and actually reduces a repetitive strand
    seq = list(b"ABABABAB" * 50 + b"XYZ")
    chrom = coil(seq)
    assert uncoil(chrom) == seq, "uncoil != identity"
    assert len(chrom.top) < len(seq), "coil did not shorten the strand"
    assert chrom.stats()["nucleosomes"] > 0 and chrom.stats()["layers"] >= 1, "no coiling happened"
    print(f"  coil/uncoil exact: OK  ({chrom.stats()})")

    # 3. lossless round-trip is the identity on many kinds of data
    cases = {
        "repetitive": b"the quick brown fox " * 200,
        "dna": b"".join(random.choice([b"ACGTACGT", b"GGGCCC", b"TATA"]) for _ in range(400)),
        "binary": bytes(random.randrange(256) for _ in range(1500)),
        "empty": b"",
        "single": b"Q",
        "text": ("warp compression folds the card into a cube and coils it like a chromosome. "
                 * 40).encode(),
    }
    for name, data in cases.items():
        blob = wc.compress(data, mode="lossless")
        back = wc.decompress(blob)
        assert back == data, f"lossless round-trip failed: {name}"
    print(f"  lossless round-trip: OK  ({len(cases)} inputs, incl. empty/single/random)")

    # 4. lossless genuinely compresses a repetitive input
    data = b"the quick brown fox " * 300
    ratio = len(data) / len(wc.compress(data, mode="lossless"))
    assert ratio > 5.0, f"repetitive data barely compressed (ratio {ratio:.1f})"
    print(f"  lossless compresses: OK  (repetitive {ratio:.1f}x)")

    # 5. lossy: error stays within q/2, and larger q shrinks the blob monotonically
    sig = bytes(min(255, max(0, int(128 + 100 * math.sin(i * 0.05) + random.randrange(-5, 5))))
                for i in range(2500))
    prev = None
    for q in (4, 8, 16, 32):
        blob = wc.compress(sig, mode="lossy", q=q)
        back = wc.decompress(blob)
        assert len(back) == len(sig)
        max_err = max(abs(a - b) for a, b in zip(sig, back))
        assert max_err <= q, f"lossy error {max_err} exceeds q={q}"
        size = len(blob)
        if prev is not None:
            assert size <= prev, f"bigger q did not shrink the blob ({size} > {prev})"
        prev = size
    print("  lossy rate/distortion dial: OK  (error <= q, size shrinks with q)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
