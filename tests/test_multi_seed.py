"""N typed seed chromosomes: cluster a mixed batch to prefix anchors, share each once, recover on the GPU."""
import numpy as np

import warp as wp

from warp_compress.multi_seed import MultiSeedStore, _lcp

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _mixed(n_prompts=4, K=120, seed=0):
    rng = np.random.default_rng(seed)
    prompts = [rng.integers(0, 50257, int(rng.integers(200, 400))).astype(np.int64) for _ in range(n_prompts)]
    seqs = []
    for _ in range(K):
        j = int(rng.integers(0, n_prompts))
        seqs.append(np.concatenate([prompts[j], rng.integers(0, 50257, int(rng.integers(8, 30))).astype(np.int64)]))
    return seqs, n_prompts


def test_lcp_is_correct():
    assert _lcp([np.array([1, 2, 3, 9]), np.array([1, 2, 3, 4, 5]), np.array([1, 2, 7])]) == 2
    assert _lcp([np.array([5, 6]), np.array([9, 6])]) == 0


def test_recovers_every_request_exactly():
    seqs, _ = _mixed(seed=1)
    store = MultiSeedStore(seqs, device=_DEV)
    for r in (0, 40, 119):
        assert np.array_equal(store.recover_request(r), seqs[r])


def test_discovers_the_prompt_anchors():
    seqs, n_prompts = _mixed(n_prompts=5, K=200, seed=2)
    store = MultiSeedStore(seqs, device=_DEV)
    assert store.n_seeds == n_prompts                     # one seed per distinct system prompt
    assert sum(store.sizes) == 200                        # every request assigned to exactly one seed


def test_multi_seed_beats_single_global_on_mixed_batch():
    seqs, _ = _mixed(n_prompts=5, K=200, seed=3)
    multi = MultiSeedStore(seqs, device=_DEV)
    single = MultiSeedStore(seqs, n_seeds=1, device=_DEV)
    dup = multi.raw_duplicated_bytes()
    assert dup / multi.size_bytes() > 5                   # multi-seed shares each prompt once -> big win
    assert dup / single.size_bytes() < 2                  # one global prefix finds no common head -> ~no win
    assert multi.size_bytes() < single.size_bytes()


def test_batched_recovery_matches():
    seqs, _ = _mixed(seed=4)
    store = MultiSeedStore(seqs, device=_DEV)
    rng = np.random.default_rng(5)
    rq = rng.integers(0, len(seqs), 2000).astype(np.int32)
    pp = np.array([rng.integers(0, store.req_len(r)) for r in rq], np.int32)
    got = store.recover(rq, pp)
    truth = np.array([seqs[r][p] for r, p in zip(rq, pp)])
    assert np.array_equal(got, truth)
