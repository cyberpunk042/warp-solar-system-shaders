"""Huffman-class RRR: GPU rank1 stays exact while the class-stream Huffman lifts the floor toward H0."""
import numpy as np

import warp as wp

from warp_compress.gpu_rrr import GPURRR
from warp_compress.gpu_rrr_huffman import GPURRRHuff, _canonical, _huff_lengths

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _check_rank(bits, seed=0):
    rh = GPURRRHuff(bits, device=_DEV)
    cum = np.concatenate([[0], np.cumsum(np.asarray(bits) != 0)])
    q = np.random.default_rng(seed).integers(0, len(bits) + 1, 3000).astype(np.int32)
    assert np.array_equal(rh.rank1(q), cum[q])
    return rh


def test_rank_exact_on_skewed():
    _check_rank((np.random.default_rng(1).random(80000) < 0.03).astype(np.uint8))


def test_rank_exact_on_balanced():
    _check_rank((np.random.default_rng(2).random(80000) < 0.5).astype(np.uint8))


def test_rank_exact_on_all_zero_and_all_one():
    _check_rank(np.zeros(6000, np.uint8))                   # single class symbol
    _check_rank(np.ones(6000, np.uint8))


def test_rank_exact_at_boundaries():
    rng = np.random.default_rng(3)
    bits = (rng.random(4000) < 0.2).astype(np.uint8)
    rh = GPURRRHuff(bits, device=_DEV)
    cum = np.concatenate([[0], np.cumsum(bits)])
    q = np.array([0, 1, 14, 15, 16, len(bits) - 1, len(bits)], np.int32)
    assert np.array_equal(rh.rank1(q), cum[q])


def test_huffman_beats_fixed_class_on_skewed():
    bits = (np.random.default_rng(4).random(200000) < 0.02).astype(np.uint8)
    r4 = GPURRR(bits, device=_DEV)
    rh = GPURRRHuff(bits, device=_DEV)
    assert rh.size_bits() < r4.size_bits()                 # Huffman class stream is smaller on a skewed plane


def test_canonical_codes_are_prefix_free():
    L = _huff_lengths(np.array([50, 10, 3, 1] + [0] * 12))  # a skewed class histogram
    maxlen, first_code, cnt, fidx, syms, code_of = _canonical(L)
    codes = [(code_of[s], L[s]) for s in syms]
    for i, (ci, li) in enumerate(codes):                    # no code is a prefix of another
        for j, (cj, lj) in enumerate(codes):
            if i != j and li <= lj:
                assert (cj >> (lj - li)) != ci
