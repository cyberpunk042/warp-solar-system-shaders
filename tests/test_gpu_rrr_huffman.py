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


def test_huffman_wavelet_access_reconstructs():
    from warp_compress.gpu_rrr_huffman import RRRWaveletGPUHuff
    rng = np.random.default_rng(5)
    # very skewed values (like quantized weights): heavy near a center -> skewed low bitplanes
    seq = np.clip(np.round(rng.standard_normal(40000) * 2), -7, 7).astype(np.int64) + 7
    rh = RRRWaveletGPUHuff(seq, device=_DEV, bits=4)
    pos = rng.integers(0, len(seq), 4000).astype(np.int32)
    assert np.array_equal(rh.access(pos), seq[pos])


def test_huffman_wavelet_beats_4bit_on_skewed_values():
    from warp_compress.gpu_rrr_huffman import RRRWaveletGPUHuff
    from warp_compress.gpu_rrr_wavelet import RRRWaveletGPU
    rng = np.random.default_rng(6)
    seq = np.clip(np.round(rng.standard_normal(60000) * 1.5), -7, 7).astype(np.int64) + 7
    r4 = RRRWaveletGPU(seq, device=_DEV, bits=4)
    rh = RRRWaveletGPUHuff(seq, device=_DEV, bits=4)
    assert np.array_equal(rh.access(np.arange(len(seq))), seq)   # lossless
    assert rh.index_bytes() < r4.index_bytes()                  # class-stream Huffman is smaller here


def test_weight_store_huffman_is_lossless_and_smaller():
    from warp_compress.weight_store import QuantizedWeightStore
    W = (np.random.default_rng(7).standard_normal((512, 256)) / np.sqrt(256)).astype(np.float32)
    base = QuantizedWeightStore(W, bits=4, device=_DEV)
    huff = QuantizedWeightStore(W, bits=4, device=_DEV, huffman=True)
    assert np.array_equal(huff.reconstruct(), base.reconstruct())   # same quantized values, still lossless
    assert huff.bits_per_weight() <= base.bits_per_weight()


def test_huffman_wavelet_two_level_samples_and_serialise():
    from warp_compress.gpu_rrr_huffman import RRRWaveletGPUHuff
    rng = np.random.default_rng(8)
    seq = np.clip(np.round(rng.standard_normal(50000) * 2), -7, 7).astype(np.int64) + 7
    rh = RRRWaveletGPUHuff(seq, device=_DEV, bits=4)
    # all three per-level sample tables (rank / offset / class) are two-level: uint16 deltas + sparser anchors
    for d, a in ((rh.rank_d, rh.rank_a), (rh.off_d, rh.off_a), (rh.cls_d, rh.cls_a)):
        assert d.dtype == wp.uint16 and a.shape[1] < d.shape[1]
    p, arrs = rh.to_host()                                          # survives round-trip with access still exact
    rh2 = RRRWaveletGPUHuff.from_host(p, arrs, _DEV)
    assert np.array_equal(rh2.access(np.arange(len(seq))), seq)


def test_weight_store_huffman_serialises_with_two_level():
    from warp_compress.weight_store import QuantizedWeightStore
    W = (np.random.default_rng(9).standard_normal((256, 256)) / 16).astype(np.float32)
    st = QuantizedWeightStore(W, bits=4, device=_DEV, huffman=True)
    st2 = QuantizedWeightStore.load(st.save(), device=_DEV)         # the huffman=True default container round-trips
    assert np.array_equal(st.reconstruct(), st2.reconstruct())


def test_canonical_codes_are_prefix_free():
    L = _huff_lengths(np.array([50, 10, 3, 1] + [0] * 12))  # a skewed class histogram
    maxlen, first_code, cnt, fidx, syms, code_of = _canonical(L)
    codes = [(code_of[s], L[s]) for s in syms]
    for i, (ci, li) in enumerate(codes):                    # no code is a prefix of another
        for j, (cj, lj) in enumerate(codes):
            if i != j and li <= lj:
                assert (cj >> (lj - li)) != ci
