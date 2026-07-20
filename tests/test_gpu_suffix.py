"""GPU-built suffix array / BWT: bit-identical to the CPU builder, and the FM-index can build fully on-GPU."""
import numpy as np

import warp as wp

from warp_compress.fm_index import suffix_array
from warp_compress.gpu_rrr_wavelet import GPURRRFMIndex
from warp_compress.gpu_suffix import gpu_bwt, gpu_suffix_array

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _sentinel(seq):
    return np.concatenate([np.asarray(seq, np.int64) + 1, [0]])


def test_matches_cpu_suffix_array_across_distributions():
    rng = np.random.default_rng(0)
    for seq in (rng.integers(0, 64, 20000), rng.integers(0, 4, 30000),
                np.cumsum(rng.integers(0, 3, 15000)) % 16, np.tile(rng.integers(0, 8, 300), 60)):
        s = _sentinel(seq)
        assert np.array_equal(gpu_suffix_array(s, device=_DEV), suffix_array(s))


def test_handles_tiny_and_uniform_inputs():
    assert np.array_equal(gpu_suffix_array(np.array([0], np.int64), device=_DEV), suffix_array(np.array([0])))
    s = _sentinel(np.full(500, 3))                          # all-equal symbols + sentinel
    assert np.array_equal(gpu_suffix_array(s, device=_DEV), suffix_array(s))


def test_gpu_bwt_is_invertible_shape_and_matches_cpu():
    rng = np.random.default_rng(1)
    seq = rng.integers(0, 32, 8000)
    bwt, sa = gpu_bwt(seq, device=_DEV)
    s = _sentinel(seq)
    assert np.array_equal(sa, suffix_array(s)) and np.array_equal(bwt, s[(sa - 1) % s.shape[0]])


def test_fm_index_built_on_gpu_matches_cpu_built():
    rng = np.random.default_rng(2)
    seq = rng.integers(0, 24, 12000)
    gpu_built = GPURRRFMIndex(seq, device=_DEV, build="gpu")
    cpu_built = GPURRRFMIndex(seq, device=_DEV, build="cpu")
    pats = [[int(x) for x in seq[a:a + 3]] for a in rng.integers(0, len(seq) - 3, 20)]
    assert np.array_equal(gpu_built.count(pats), cpu_built.count(pats))   # same index, same search results
