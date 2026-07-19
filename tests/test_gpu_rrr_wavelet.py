"""RRR-backed GPU wavelet + FM-index: entropy-sized index with GPU access/rank and count/predict_next."""
import numpy as np

import warp as wp

from warp_compress.fm_index import FMIndex, suffix_array
from warp_compress.gpu_rrr_wavelet import GPURRRFMIndex, RRRWaveletGPU
from warp_compress.gpu_wavelet import GPUWavelet

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _markov(n=40000, V=32, seed=0):
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.ones(V) * 0.3, size=V)
    seq = np.empty(n, np.int64)
    seq[0] = 0
    for i in range(1, n):
        seq[i] = rng.choice(V, p=trans[seq[i - 1]])
    return seq, V


def test_rrr_wavelet_access_reconstructs():
    seq, _ = _markov()
    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]]
    rrw = RRRWaveletGPU(bwt, device=_DEV)
    rng = np.random.default_rng(1)
    pos = rng.integers(0, bwt.shape[0], 4000).astype(np.int32)
    assert np.array_equal(rrw.access(pos), bwt[pos])


def test_rrr_wavelet_rank_matches_naive():
    seq, V = _markov()
    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]]
    rrw = RRRWaveletGPU(bwt, device=_DEV)
    rng = np.random.default_rng(2)
    c = rng.integers(0, V + 1, 300).astype(np.int32)
    i = rng.integers(0, bwt.shape[0] + 1, 300).astype(np.int32)
    got = rrw.rank(c, i)
    for j in range(0, 300, 29):
        assert got[j] == int(np.count_nonzero(bwt[: i[j]] == c[j]))


def test_rrr_wavelet_is_smaller_than_packed_on_bwt():
    seq, _ = _markov()
    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]]
    rrw = RRRWaveletGPU(bwt, device=_DEV)
    pk = GPUWavelet(bwt, device=_DEV)
    assert rrw.index_bytes() < pk.index_bytes()          # BWT planes are skewed -> RRR beats packed


def test_rrr_fm_count_and_predict_match_cpu():
    seq, V = _markov(n=30000, V=24, seed=3)
    gfm = GPURRRFMIndex(seq, device=_DEV)
    cfm = FMIndex(seq)
    rng = np.random.default_rng(4)
    pats = [[int(x) for x in seq[a:a + 3]] for a in rng.integers(0, len(seq) - 3, 30)]
    got = gfm.count(pats)
    for j in range(0, 30, 7):
        assert got[j] == cfm.count(pats[j])
    for _ in range(6):
        i = int(rng.integers(50, len(seq)))
        ctx = [int(seq[i - 2]), int(seq[i - 1])]
        cdist = cfm.predict_next(ctx)
        if cdist:
            assert int(np.argmax(gfm.predict_next(ctx, vocab=V))) == max(cdist, key=cdist.get)


def test_rrr_fm_absent_pattern_is_zero():
    seq, V = _markov(n=15000, V=16, seed=5)
    gfm = GPURRRFMIndex(seq, device=_DEV)
    assert gfm.count([[V + 2, V + 3]])[0] == 0
