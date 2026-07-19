"""multi_seed — N typed seed chromosomes: cluster a mixed batch to its prefix-anchors, then share each once.

X and Y were the first two seed chromosomes; this is the general case. A real serving mix isn't one shared
prefix — it's several distinct system prompts / near-duplicate families. ChromoFold discovers the prefix
**anchors** (seeds), assigns every sequence to its best-matching seed, and stores each seed's shared prefix
ONCE + per-member tails. Seeds are ranked by cluster size = **importance** (the most common prefix is seed A,
then B, …), and the number of seeds is a tunable (`n_seeds`): 1 collapses to a single global prefix, ∞ is one
seed per distinct prefix.

This is the honest generalization of the shared-prefix prompt cache to a mixed batch: a single global prefix
finds no common head across different system prompts and compresses nothing; multi-seed recovers the win per
cluster. Recovery of any request's token span runs on the GPU in O(1). Run: python -m warp_compress.multi_seed
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _recover_k(seeds: wp.array(dtype=wp.int32), seed_start: wp.array(dtype=wp.int32),
               seed_plen: wp.array(dtype=wp.int32), tails: wp.array(dtype=wp.int32),
               tail_start: wp.array(dtype=wp.int32), seed_of: wp.array(dtype=wp.int32),
               req_in: wp.array(dtype=wp.int32), pos_in: wp.array(dtype=wp.int32),
               out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    r = req_in[t]
    p = pos_in[t]
    sid = seed_of[r]
    pl = seed_plen[sid]
    if p < pl:
        out[t] = seeds[seed_start[sid] + p]            # in this request's seed (shared) prefix
    else:
        out[t] = tails[tail_start[r] + (p - pl)]       # in this request's private tail


def _lcp(members) -> int:
    """Longest common prefix length across a group of token arrays."""
    L = int(min(len(m) for m in members))
    m0 = members[0]
    for s in members[1:]:
        k = 0
        lim = min(L, len(s))
        while k < lim and int(s[k]) == int(m0[k]):
            k += 1
        L = k
        if L == 0:
            break
    return L


def _cluster(seqs, n_seeds, sig_len):
    """Group by a prefix signature (first sig_len tokens); the largest groups become the seeds (importance
    order). With a cap `n_seeds`, groups beyond the cap become singleton seeds (stored whole, no sharing)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[tuple(int(x) for x in s[:sig_len])].append(i)
    ranked = sorted(groups.values(), key=len, reverse=True)
    if n_seeds is not None and len(ranked) > n_seeds:
        kept = ranked[:n_seeds]
        singles = [i for g in ranked[n_seeds:] for i in g]
        ranked = kept + [[i] for i in singles]         # the tail-of-distribution each stands alone
    return ranked


class MultiSeedStore:
    """A mixed batch stored as N seed prefixes (shared once each) + per-request tails, resident on the GPU."""

    def __init__(self, sequences, n_seeds: int | None = None, sig_len: int = 32, device: str = "cuda:0"):
        self.device = device
        seqs = [np.asarray(s, np.int64) for s in sequences]
        self.K = len(seqs)
        clusters = _cluster(seqs, n_seeds, sig_len)     # importance-ranked (largest first)

        seed_arrays, seed_plen, seed_of = [], [], np.zeros(self.K, np.int64)
        tails = [None] * self.K
        self.sizes = []
        for sid, members in enumerate(clusters):
            lcp = _lcp([seqs[i] for i in members]) if len(members) > 1 else len(seqs[members[0]])
            seed_arrays.append(seqs[members[0]][:lcp])
            seed_plen.append(lcp)
            self.sizes.append(len(members))
            for i in members:
                seed_of[i] = sid
                tails[i] = seqs[i][lcp:]

        self.n_seeds = len(clusters)
        seeds_flat = np.concatenate(seed_arrays) if seed_arrays else np.zeros(1, np.int64)
        seed_start = np.concatenate([[0], np.cumsum([len(a) for a in seed_arrays])])[:-1]
        tails_flat = np.concatenate([tails[i] for i in range(self.K)]) if self.K else np.zeros(1, np.int64)
        tail_start = np.concatenate([[0], np.cumsum([len(tails[i]) for i in range(self.K)])])[:-1]

        self._seqs = seqs
        self._seed_plen_np = np.asarray(seed_plen, np.int64)
        self._seed_of_np = seed_of
        self._seeds_bytes = int(seeds_flat.shape[0])
        self._tails_bytes = int(tails_flat.shape[0])
        self.seeds = wp.array(seeds_flat.astype(np.int32), dtype=wp.int32, device=device)
        self.seed_start = wp.array(seed_start.astype(np.int32), dtype=wp.int32, device=device)
        self.seed_plen = wp.array(self._seed_plen_np.astype(np.int32), dtype=wp.int32, device=device)
        self.tails = wp.array(tails_flat.astype(np.int32), dtype=wp.int32, device=device)
        self.tail_start = wp.array(tail_start.astype(np.int32), dtype=wp.int32, device=device)
        self.seed_of = wp.array(seed_of.astype(np.int32), dtype=wp.int32, device=device)

    def req_len(self, r: int) -> int:
        return int(self._seqs[r].shape[0])

    def size_bytes(self) -> int:
        return (self._seeds_bytes + self._tails_bytes) * 2       # seeds once + tails (uint16)

    def raw_duplicated_bytes(self) -> int:
        return int(sum(s.shape[0] for s in self._seqs)) * 2

    def recover(self, reqs, positions) -> np.ndarray:
        r = wp.array(np.asarray(reqs, np.int32), dtype=wp.int32, device=self.device)
        p = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(r.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_recover_k, dim=r.shape[0],
                  inputs=[self.seeds, self.seed_start, self.seed_plen, self.tails, self.tail_start,
                          self.seed_of, r, p, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def recover_request(self, r: int) -> np.ndarray:
        L = self.req_len(r)
        return self.recover(np.full(L, r, np.int32), np.arange(L, dtype=np.int32))


def _demo():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)

    # a realistic MIXED batch: 5 distinct system prompts (different lengths) + K requests split among them
    n_prompts = 5
    prompts = [rng.integers(0, 50257, int(rng.integers(500, 900))).astype(np.int64) for _ in range(n_prompts)]
    K = 300
    seqs, truth_seed = [], []
    for _ in range(K):
        j = int(rng.integers(0, n_prompts))                     # which system prompt this request uses
        suffix = rng.integers(0, 50257, int(rng.integers(12, 40))).astype(np.int64)
        seqs.append(np.concatenate([prompts[j], suffix]))
        truth_seed.append(j)

    store = MultiSeedStore(seqs, sig_len=32, device=dev)

    ok = all(np.array_equal(store.recover_request(r), seqs[r]) for r in (0, 150, 299))
    Q = 1 << 18
    rq = rng.integers(0, K, Q).astype(np.int32)
    pp = np.array([rng.integers(0, store.req_len(r)) for r in rq], np.int32)
    for _ in range(3):
        store.recover(rq, pp)
    t0 = time.perf_counter()
    for _ in range(20):
        store.recover(rq, pp)
    ns = (time.perf_counter() - t0) / 20 / Q * 1e9

    # a SINGLE global prefix (n_seeds=1) can't find a common head across different prompts -> ~no compression
    single = MultiSeedStore(seqs, n_seeds=1, sig_len=32, device=dev)

    dup = store.raw_duplicated_bytes()
    print(f"device={dev}   mixed batch: {K} requests over {n_prompts} distinct system prompts")
    print(f"[cluster] discovered {store.n_seeds} seeds (importance-ranked sizes): {sorted(store.sizes, reverse=True)}")
    print(f"[correct] GPU span recovery == original request ✓" if ok else "[correct] FAIL")
    print(f"[store]  raw-duplicated {dup/1e3:8.1f} KB")
    print(f"         single global prefix (n_seeds=1) {single.size_bytes()/1e3:8.1f} KB  "
          f"=> {dup/single.size_bytes():.2f}×  (fails: no common head across prompts)")
    print(f"         MULTI-SEED ({store.n_seeds} anchors)        {store.size_bytes()/1e3:8.1f} KB  "
          f"=> {dup/store.size_bytes():.1f}×  (each prompt shared once)")
    print(f"[recover] {Q:,} random span tokens in {(Q*ns/1e9)*1e3:.2f} ms  ({ns:.1f} ns/token, O(1) on the GPU)")
    print("\n=> N typed seed chromosomes generalize X/Y: cluster the mixed batch to its prefix anchors (ranked "
          "by importance), share each once. A single global prefix compresses a mixed batch by ~1×; multi-seed\n"
          "   recovers the per-cluster win. `n_seeds` is the tunable — 1 = global, ∞ = one per distinct prefix. "
          "This is the realistic serving mix (many system prompts), not one shared prompt.")


if __name__ == "__main__":
    _demo()
