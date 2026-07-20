"""Block-wise rANS value array: lossless GPU decode/fetch, near-entropy on skewed data, serialise, coder."""
import numpy as np

import warp as wp

from warp_compress.gpu_rans import BlockRANSArray, _M, _normalize

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def test_normalize_sums_to_M_and_keeps_present_symbols():
    freq, cum, slot2sym = _normalize(np.array([500, 3, 1, 0, 40]))
    assert int(freq.sum()) == _M
    assert (freq[[0, 1, 2, 4]] >= 1).all() and freq[3] == 0    # present symbols keep freq>=1


def test_decode_reconstructs_values():
    rng = np.random.default_rng(1)
    for vals in (np.clip(np.round(rng.standard_normal(40000) * 2), -7, 7).astype(np.int64) + 7,
                 np.where(rng.random(40000) < 0.05, rng.integers(0, 15, 40000), 7).astype(np.int64)):
        assert np.array_equal(BlockRANSArray(vals, block=64, device=_DEV).decode(), vals)


def test_fetch_matches_random_positions():
    rng = np.random.default_rng(2)
    vals = np.where(rng.random(30000) < 0.1, rng.integers(0, 15, 30000), 7).astype(np.int64)
    ra = BlockRANSArray(vals, block=128, device=_DEV)
    idx = rng.integers(0, len(vals), 1500).astype(np.int32)
    assert np.array_equal(ra.fetch(idx), vals[idx])


def test_rans_beats_huffman_on_low_entropy_large_block():
    from warp_compress.gpu_block_huffman import BlockHuffmanArray
    rng = np.random.default_rng(3)
    vals = np.where(rng.random(200000) < 0.03, rng.integers(0, 15, 200000), 5).astype(np.int64)   # H0 < 1
    ra = BlockRANSArray(vals, block=1024, device=_DEV)
    bh = BlockHuffmanArray(vals, block=1024, device=_DEV)
    assert np.array_equal(ra.decode(), vals)
    assert ra.size_bits() < bh.size_bits()                     # rANS breaks Huffman's 1-bit-per-symbol floor


def test_to_host_from_host_roundtrip():
    rng = np.random.default_rng(4)
    vals = np.clip(np.round(rng.standard_normal(20000)), -7, 7).astype(np.int64) + 7
    ra = BlockRANSArray(vals, block=256, device=_DEV)
    p, a = ra.to_host()
    assert np.array_equal(BlockRANSArray.from_host(p, a, _DEV).decode(), vals)


def test_single_symbol_and_small_inputs():
    assert np.array_equal(BlockRANSArray(np.full(500, 7, np.int64), device=_DEV).decode(), np.full(500, 7))
    v = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], np.int64)
    assert np.array_equal(BlockRANSArray(v, block=4, device=_DEV).decode(), v)


def test_weight_store_rans_coder_lossless_and_saves():
    from warp_compress.weight_store import QuantizedWeightStore
    W = (np.random.default_rng(5).standard_normal((256, 256)) * 0.02).astype(np.float32)
    rr = QuantizedWeightStore(W, bits=4, coder="rrr", huffman=True, device=_DEV)
    ra = QuantizedWeightStore(W, bits=4, coder="rans", block=256, device=_DEV)
    assert np.array_equal(rr.reconstruct(), ra.reconstruct())  # both dequant the same int4 values
    ra2 = QuantizedWeightStore.load(ra.save(), device=_DEV)
    assert np.array_equal(ra.reconstruct(), ra2.reconstruct())
