"""GPU (Warp) FM-index backward search: batched count and predict_next match the CPU FM-index and naive."""
import numpy as np

import warp as wp

from warp_compress.fm_index import FMIndex
from warp_compress.gpu_fm_index import GPUFMIndex

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _seq(n=30000, V=48, seed=2):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    return rng.choice(V, size=n, p=p).astype(np.int64), V


def test_gpu_count_matches_naive_and_cpu():
    seq, _ = _seq()
    gfm = GPUFMIndex(seq, device=_DEV)
    cfm = FMIndex(seq)
    rng = np.random.default_rng(4)
    pats = [[int(x) for x in seq[a:a + 3]] for a in rng.integers(0, len(seq) - 3, 40)]
    got = gfm.count(pats)
    for j in range(0, 40, 5):
        naive = sum(1 for i in range(len(seq) - 3) if list(seq[i:i + 3]) == pats[j])
        assert got[j] == naive
        assert got[j] == cfm.count(pats[j])                 # agrees with the CPU FM-index


def test_gpu_count_of_absent_pattern_is_zero():
    seq, V = _seq()
    gfm = GPUFMIndex(seq, device=_DEV)
    assert gfm.count([[V + 3, V + 4]])[0] == 0              # symbols outside the alphabet


def test_gpu_predict_next_top1_matches_cpu_on_markov():
    rng = np.random.default_rng(1)
    V = 12
    trans = rng.dirichlet(np.ones(V) * 0.25, size=(V, V))
    seq = np.empty(30000, np.int64)
    seq[:2] = [0, 1]
    for i in range(2, len(seq)):
        seq[i] = rng.choice(V, p=trans[seq[i - 2], seq[i - 1]])
    gfm = GPUFMIndex(seq, device=_DEV)
    cfm = FMIndex(seq)
    for _ in range(8):
        i = int(rng.integers(100, len(seq)))
        ctx = [int(seq[i - 2]), int(seq[i - 1])]
        gdist = gfm.predict_next(ctx, vocab=V)
        cdist = cfm.predict_next(ctx)
        if cdist:                                           # both should peak on the same continuation
            assert int(np.argmax(gdist)) == max(cdist, key=cdist.get)


def test_gpu_locate_matches_naive_and_cpu():
    seq, _ = _seq(n=12000, V=32)
    gfm = GPUFMIndex(seq, device=_DEV, sa_sample=16)
    cfm = FMIndex(seq, sa_sample=16)
    rng = np.random.default_rng(7)
    for _ in range(15):
        at = int(rng.integers(0, len(seq) - 3))
        pat = [int(x) for x in seq[at:at + 3]]
        naive = sorted(i for i in range(len(seq) - 2) if list(seq[i:i + 3]) == pat)
        got = list(gfm.locate([pat])[0])
        assert got == naive
        assert got == cfm.locate(pat)                     # agrees with the CPU FM-index locate


def test_gpu_locate_batched_and_absent():
    seq, V = _seq(n=6000, V=24)
    gfm = GPUFMIndex(seq, device=_DEV)
    a = [int(x) for x in seq[500:503]]
    b = [V + 5, V + 6]                                    # absent
    res = gfm.locate([a, b])
    assert list(res[0]) == sorted(i for i in range(len(seq) - 2) if list(seq[i:i + 3]) == a)
    assert res[1].shape[0] == 0


def test_gpu_locate_respects_sa_sample():
    seq, _ = _seq(n=5000, V=16)
    for s in (8, 32, 64):                                 # different sampling rates -> same answer
        gfm = GPUFMIndex(seq, device=_DEV, sa_sample=s)
        pat = [int(x) for x in seq[1000:1003]]
        naive = sorted(i for i in range(len(seq) - 2) if list(seq[i:i + 3]) == pat)
        assert list(gfm.locate([pat])[0]) == naive


def test_gpu_predict_next_is_a_distribution():
    seq, V = _seq(n=20000, V=24)
    gfm = GPUFMIndex(seq, device=_DEV)
    d = gfm.predict_next([int(seq[500]), int(seq[501])], vocab=V)
    assert abs(d.sum() - 1.0) < 1e-6 and (d >= 0).all()
