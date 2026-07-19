"""The token_chromosome extraction: lossless, exactly invertible, locality-preserving, O(1) navigation."""
import math

import numpy as np

from warp_compress.lm_memory import build_memory
from warp_compress.token_chromosome import (Chromosome, compress, compress_hierarchical,
                                            hilbert_index, hilbert_point)


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


def test_hierarchy_token_ranges_tile_and_drill():
    seq = _seq()
    h = compress_hierarchical(seq, branch=4, dim=3)
    assert h.count(0) == h.n and h.count(h.n_levels - 1) == 1     # a pyramid down to the root
    for level in range(1, h.n_levels):
        covered = 0
        for r in range(h.count(level)):
            lo, hi = h.token_range(level, r)
            assert lo == covered                                 # ranges tile the sequence, in order
            covered = hi
        assert covered == h.n
    level = min(3, h.n_levels - 1)
    r = h.count(level) // 2
    lo, hi = h.token_range(level, r)
    assert np.array_equal(h.drill(level, r), seq[lo:hi])         # drill-down == the exact token run
    assert h.parent(level, h.children(level, r)[0]) == r         # parent/children consistent


def test_lm_memory_lossless_fetch_and_saving():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((3000, 32)).astype(np.float32)
    p = 1.0 / np.arange(1, 3001)
    p /= p.sum()
    ctx = rng.choice(3000, size=20000, p=p)          # repetitive context: used-unique V << N
    mem = build_memory(ctx, table, branch=4, dim=3)
    for r in (0, 1, 999, mem.n - 1):
        assert np.array_equal(mem.fetch(r), table[ctx[r]])   # O(1) lossless reconstruction
    m = mem.memory_bytes()
    fetch_mem = m["embeddings_bytes"] + m["index_bytes"]
    assert fetch_mem < m["naive_bytes"]                      # smaller than the full per-token cache
