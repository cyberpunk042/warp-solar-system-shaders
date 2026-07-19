"""GPU (Warp) reference/delta cluster: batched fetch + whole-leaf decode reconstruct the originals in VRAM."""
import numpy as np

import warp as wp

from warp_compress.gpu_delta import GPUDeltaCluster

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _cluster(k=16, n=500, mut=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, size=n)
    out = []
    for _ in range(k):
        s = base.copy()
        f = rng.integers(0, n, size=mut)
        s[f] = rng.integers(0, 4, size=mut)
        out.append(s)
    return out


def test_batched_fetch_matches_originals():
    seqs = _cluster(seed=1)
    gd = GPUDeltaCluster(seqs, device=_DEV)
    rng = np.random.default_rng(2)
    Q = 5000
    lv = rng.integers(0, len(seqs), Q).astype(np.int32)
    ps = np.array([rng.integers(0, gd.lengths[l]) for l in lv], np.int32)
    got = gd.fetch(lv, ps)
    truth = np.array([seqs[lv[i]][ps[i]] for i in range(Q)])
    assert np.array_equal(got, truth)


def test_whole_leaf_decode_roundtrips():
    seqs = _cluster(k=13, n=400, mut=6, seed=3)     # odd count exercises the carry-up path
    gd = GPUDeltaCluster(seqs, device=_DEV)
    for k in range(len(seqs)):
        assert np.array_equal(gd.decode_leaf(k), seqs[k])


def test_base_leaf_has_no_deltas():
    seqs = _cluster(k=8, n=300, mut=5, seed=4)
    gd = GPUDeltaCluster(seqs, device=_DEV)
    assert np.array_equal(gd.decode_leaf(0), seqs[0])   # leftmost leaf == base, empty path


def test_compresses_related_cluster():
    seqs = _cluster(k=32, n=2000, mut=20, seed=5)       # ~1% divergence
    gd = GPUDeltaCluster(seqs, device=_DEV)
    raw = sum(len(s) for s in seqs) * 4
    assert gd.size_bytes() < raw                        # base + sparse deltas beats storing every member


def test_large_alphabet_delta_cluster():
    rng = np.random.default_rng(6)
    base = rng.integers(0, 5000, size=800)              # large alphabet: delta is alphabet-agnostic
    seqs = []
    for _ in range(10):
        s = base.copy(); f = rng.integers(0, 800, size=8); s[f] = rng.integers(0, 5000, size=8); seqs.append(s)
    gd = GPUDeltaCluster(seqs, device=_DEV)
    for k in range(len(seqs)):
        assert np.array_equal(gd.decode_leaf(k), seqs[k])
