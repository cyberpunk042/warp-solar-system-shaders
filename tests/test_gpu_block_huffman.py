"""Block-wise LUT-Huffman value array (DFloat11-style fast decode): lossless decode/fetch, serialise, coder."""
import numpy as np

import warp as wp

from warp_compress.gpu_block_huffman import BlockHuffmanArray, _canonical, _huff_lengths

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _peaky(n=50000, seed=0):
    return np.clip(np.round(np.random.default_rng(seed).standard_normal(n) * 2), -7, 7).astype(np.int64) + 7


def test_decode_reconstructs_the_values():
    for vals in (_peaky(seed=1), np.random.default_rng(2).integers(0, 16, 40000)):
        bh = BlockHuffmanArray(vals, block=16, device=_DEV)
        assert np.array_equal(bh.decode(), vals)


def test_fetch_matches_random_positions():
    vals = _peaky(seed=3)
    bh = BlockHuffmanArray(vals, block=32, device=_DEV)
    idx = np.random.default_rng(4).integers(0, len(vals), 2000).astype(np.int32)
    assert np.array_equal(bh.fetch(idx), vals[idx])


def test_block_size_is_lossless_across_dials():
    vals = _peaky(seed=5)
    for blk in (8, 16, 64, 256):
        assert np.array_equal(BlockHuffmanArray(vals, block=blk, device=_DEV).decode(), vals)


def test_single_symbol_and_small_inputs():
    assert np.array_equal(BlockHuffmanArray(np.full(1000, 7, np.int64), device=_DEV).decode(), np.full(1000, 7))
    v = np.array([3, 1, 4, 1, 5, 9, 2, 6], np.int64)
    assert np.array_equal(BlockHuffmanArray(v, block=4, device=_DEV).decode(), v)


def test_to_host_from_host_roundtrip():
    vals = _peaky(seed=6)
    bh = BlockHuffmanArray(vals, block=64, device=_DEV)
    p, a = bh.to_host()
    bh2 = BlockHuffmanArray.from_host(p, a, _DEV)
    assert np.array_equal(bh2.decode(), vals)


def test_canonical_codes_prefix_free():
    L = _huff_lengths(np.array([100, 30, 10, 3, 1] + [0] * 11))
    maxlen, code_of = _canonical(L)
    codes = [(code_of[s], L[s]) for s in range(len(L)) if L[s] > 0]
    for i, (ci, li) in enumerate(codes):
        for j, (cj, lj) in enumerate(codes):
            if i != j and li <= lj:
                assert (cj >> (lj - li)) != ci


def test_weight_store_block_coder_matches_and_saves():
    from warp_compress.weight_store import QuantizedWeightStore
    W = (np.random.default_rng(7).standard_normal((256, 256)) / 16).astype(np.float32)
    rr = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    bl = QuantizedWeightStore(W, bits=4, device=_DEV, coder="block", block=64)
    assert np.array_equal(rr.reconstruct(), bl.reconstruct())    # both dequant the same int4 values (lossless)
    bl2 = QuantizedWeightStore.load(bl.save(), device=_DEV)
    assert np.array_equal(bl.reconstruct(), bl2.reconstruct())    # block coder serialises + round-trips
    idx = np.array([0, 7, 5000])
    assert np.allclose(bl.fetch(idx), bl.reconstruct().ravel()[idx])
