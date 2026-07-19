"""GPU (Warp) wavelet: batched access/rank match the sequence and the CPU wavelet, over the succinct index."""
import numpy as np

from warp_compress.gpu_wavelet import GPUWavelet
from warp_compress.wavelet import WaveletMatrix

import warp as wp

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"     # kernels run on either; test both paths


def _seq(n=20000, V=200, seed=3):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    return rng.choice(V, size=n, p=p).astype(np.int64), V


def test_gpu_access_reconstructs_sequence():
    seq, _ = _seq()
    gw = GPUWavelet(seq, device=_DEV)
    rng = np.random.default_rng(1)
    pos = rng.integers(0, len(seq), 5000).astype(np.int32)
    assert np.array_equal(gw.access(pos), seq[pos])


def test_gpu_rank_matches_naive_and_cpu_wavelet():
    seq, V = _seq()
    gw = GPUWavelet(seq, device=_DEV)
    wm = WaveletMatrix(seq)
    rng = np.random.default_rng(2)
    c = rng.integers(0, V, 400).astype(np.int32)
    i = rng.integers(0, len(seq) + 1, 400).astype(np.int32)
    got = gw.rank(c, i)
    for j in range(0, 400, 37):
        naive = int(np.count_nonzero(seq[: i[j]] == c[j]))
        assert got[j] == naive
        assert got[j] == wm.rank(int(c[j]), int(i[j]))          # agrees with the CPU wavelet too


def test_gpu_index_is_succinct():
    seq, _ = _seq(n=50000, V=256)
    gw = GPUWavelet(seq, device=_DEV)
    # packed bitplanes are ~n*bits bits; must be far below the CPU prefix-array form (n*bits*8 bytes)
    assert gw.index_bytes() < len(seq) * gw.bits                # < 1 byte per bit-level per token (i.e. packed)


def test_gpu_access_handles_boundary_positions():
    seq, _ = _seq(n=8000, V=64)
    gw = GPUWavelet(seq, device=_DEV)
    pos = np.array([0, 1, len(seq) // 2, len(seq) - 2, len(seq) - 1], np.int32)
    assert np.array_equal(gw.access(pos), seq[pos])
