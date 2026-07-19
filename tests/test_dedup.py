"""Content-aware dataset dedup: exact-dup refs + near-dup deltas + unique bases, GPU-addressable, lossless."""
import numpy as np

import warp as wp

from warp_compress.dedup import DedupStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _dataset(L=128, n_uniq=40, seed=0):
    rng = np.random.default_rng(seed)
    uniq = [rng.integers(0, 5000, L).astype(np.int64) for _ in range(n_uniq)]
    docs, kinds = [], []
    for d in uniq:
        docs.append(d); kinds.append("u")
        if rng.random() < 0.5:
            docs.append(d.copy()); kinds.append("exact")
        if rng.random() < 0.5:
            e = d.copy(); e[rng.integers(0, L, 4)] = rng.integers(0, 5000, 4); docs.append(e); kinds.append("near")
    order = rng.permutation(len(docs))
    return [docs[i] for i in order], [kinds[i] for i in order]


def test_decode_reconstructs_every_document():
    docs, _ = _dataset(seed=1)
    store = DedupStore(docs, device=_DEV)
    for k in range(len(docs)):
        assert np.array_equal(store.decode(k), docs[k])


def test_detects_exact_and_near_dups():
    docs, kinds = _dataset(n_uniq=50, seed=2)
    store = DedupStore(docs, device=_DEV)
    assert store.n_bases == kinds.count("u")             # one base per unique document
    assert store.n_exact == kinds.count("exact")
    assert store.n_near == kinds.count("near")


def test_beats_raw_but_is_lossless():
    docs, _ = _dataset(n_uniq=60, seed=3)
    store = DedupStore(docs, device=_DEV)
    assert store.size_bytes() < store.raw_bytes()        # dedup + delta beats storing every doc raw
    assert np.array_equal(store.decode(len(docs) // 2), docs[len(docs) // 2])


def test_batched_fetch_matches():
    docs, _ = _dataset(seed=4)
    store = DedupStore(docs, device=_DEV)
    rng = np.random.default_rng(5)
    dq = rng.integers(0, len(docs), 2000).astype(np.int32)
    pq = np.array([rng.integers(0, store.lengths[d]) for d in dq], np.int32)
    got = store.fetch(dq, pq)
    assert np.array_equal(got, np.array([docs[d][p] for d, p in zip(dq, pq)]))


def test_all_unique_still_correct_and_no_expansion_on_dups():
    rng = np.random.default_rng(6)
    docs = [rng.integers(0, 5000, 100).astype(np.int64) for _ in range(20)]   # all unique
    store = DedupStore(docs, device=_DEV)
    assert store.n_exact == 0 and store.n_near == 0
    assert all(np.array_equal(store.decode(k), docs[k]) for k in range(20))
