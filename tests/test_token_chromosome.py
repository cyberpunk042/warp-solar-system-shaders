"""The token_chromosome extraction: lossless, exactly invertible, locality-preserving, O(1) navigation."""
import math

import numpy as np

from warp_compress.token_chromosome import Chromosome, compress, hilbert_index, hilbert_point


def _seq():
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 40, size=500)
    return np.repeat(blocks, rng.integers(2, 10, size=blocks.shape))


def test_hilbert_roundtrip_all_indices():
    bits, dim = 4, 3                       # 4096 sites
    for r in range(1 << (bits * dim)):
        assert hilbert_index(hilbert_point(r, bits, dim), bits, dim) == r


def test_hilbert_locality_unit_steps():
    bits, dim = 5, 3
    prev = hilbert_point(0, bits, dim)
    for r in range(1, 1 << (bits * dim)):
        cur = hilbert_point(r, bits, dim)
        assert sum(abs(a - b) for a, b in zip(cur, prev)) == 1    # exactly one axis, by one
        prev = cur


def test_compress_is_lossless():
    seq = _seq()
    ch = compress(seq, dim=3)
    assert np.array_equal(ch.decompress(), seq)


def test_navigation_is_exact_o1():
    seq = _seq()
    ch = compress(seq, dim=3)
    for r in (0, 1, 17, ch.n // 2, ch.n - 1):
        assert ch.invert(ch.at(r)) == r                          # position <-> rank exact
        assert ch.next(r) == (r + 1) % ch.n
        assert np.array_equal(ch.token(r), seq[r])


def test_rate_beats_raw_id_stream():
    seq = _seq()
    ch = compress(seq, dim=3)
    rate = ch.rate_bits()
    raw = ch.n * max(1, math.ceil(math.log2(rate["V"])))
    assert rate["total_bits"] < raw                              # dedup + RLE actually compresses
