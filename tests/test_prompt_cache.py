"""Shared-prefix prompt-cache token store: GPU span recovery + the storage savings from prefix sharing."""
import numpy as np

import warp as wp

from warp_compress.prompt_cache import SharedPrefixStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _store(K=32, plen=200, slen=16, seed=0):
    rng = np.random.default_rng(seed)
    prefix = rng.integers(0, 50257, plen).astype(np.int64)
    sufs = [rng.integers(0, 50257, slen + int(rng.integers(0, 8))).astype(np.int64) for _ in range(K)]
    return SharedPrefixStore(prefix, sufs, device=_DEV), prefix, sufs


def test_recover_request_is_prefix_plus_suffix():
    store, prefix, sufs = _store()
    for r in (0, 7, 31):
        assert np.array_equal(store.recover_request(r), np.concatenate([prefix, sufs[r]]))


def test_batched_span_recovery_matches():
    store, prefix, sufs = _store(seed=1)
    rng = np.random.default_rng(2)
    reqs = rng.integers(0, store.K, 3000).astype(np.int32)
    pos = np.array([rng.integers(0, store.req_len(r)) for r in reqs], np.int32)
    got = store.recover(reqs, pos)
    truth = np.array([(prefix[p] if p < len(prefix) else sufs[r][p - len(prefix)])
                      for r, p in zip(reqs, pos)])
    assert np.array_equal(got, truth)


def test_shared_prefix_beats_duplicated_storage():
    store, _, _ = _store(K=64, plen=800, slen=16, seed=3)
    assert store.size_bytes() < store.raw_duplicated_bytes()
    # with a big shared prefix and many requests, the saving is close to K×
    assert store.raw_duplicated_bytes() / store.size_bytes() > 10


def test_single_request_degenerates():
    store, prefix, sufs = _store(K=1, plen=50, slen=10, seed=4)
    assert np.array_equal(store.recover_request(0), np.concatenate([prefix, sufs[0]]))
