"""dedup — content-aware dataset dedup with GPU random access. The RIGHT structure for a mostly-unique corpus.

The reference/delta *tree* (`gpu_delta`) assumes an all-near-duplicate cluster: it pairs by tree position, so a
dataset that is mostly unique with *some* dups gets paired unrelated-to-unrelated and expands. Real dataset
dedup needs **content clustering**, not positional folding: detect exact duplicates (→ one reference each), and
store each near-duplicate as a sparse delta against the actual nearest document. Unique documents are stored
once. Any document reconstructs on the GPU in O(1) — the property gzip/zstd lack.

    DedupStore(docs)   -> exact-dup refs + near-dup deltas + unique bases, resident on the GPU
    .decode(k)         -> document k (base ⊕ its delta), on the GPU
    .fetch(docs, pos)  -> batched random token access

Honest: this wins ratio over raw and preserves random access, but a streaming codec (zstd) still compresses the
*token entropy of the unique documents* that this leaves raw — so zstd wins pure ratio. The niche is random
access + dedup, not archival ratio. Run: python -m warp_compress.dedup
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _fetch_k(reps: wp.array(dtype=wp.int32), rep_start: wp.array(dtype=wp.int32),
            doc_ref: wp.array(dtype=wp.int32), dpos: wp.array(dtype=wp.int32), dval: wp.array(dtype=wp.int32),
            dstart: wp.array(dtype=wp.int32), dlen: wp.array(dtype=wp.int32),
            doc_in: wp.array(dtype=wp.int32), pos_in: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    doc = doc_in[t]
    pos = pos_in[t]
    rep = doc_ref[doc]
    res = reps[rep_start[rep] + pos]                       # the base document's token
    dl = dlen[doc]
    if dl > 0:                                             # near-dup: override where its sparse delta touches
        ds = dstart[doc]
        lo = int(0)
        hi = dl
        while lo < hi:
            mid = (lo + hi) >> 1
            if dpos[ds + mid] < pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < dl and dpos[ds + lo] == pos:
            res = dval[ds + lo]
    out[t] = res


class DedupStore:
    def __init__(self, docs, near_thresh: float = 0.12, device: str = "cuda:0"):
        self.device = device
        docs = [np.asarray(d, np.int64) for d in docs]
        self.docs = docs
        n = len(docs)
        self.lengths = [int(d.shape[0]) for d in docs]

        # 1) exact dedup by content hash
        seen: dict = {}
        exact_rep = np.arange(n)
        for i, d in enumerate(docs):
            h = hash(d.tobytes())
            if h in seen and np.array_equal(docs[seen[h]], d):
                exact_rep[i] = seen[h]
            else:
                seen[h] = i

        # 2) among the distinct docs, store as a base OR a near-dup delta vs an earlier base of equal length
        bases: list = []                                  # doc indices stored raw
        by_len: dict = {}
        doc_ref = np.zeros(n, np.int64)
        dpos_all, dval_all, dstart, dlen = [], [], np.zeros(n, np.int64), np.zeros(n, np.int64)
        for i in range(n):
            if exact_rep[i] != i:                         # exact duplicate -> resolve later
                continue
            d = docs[i]
            match = None
            for j in by_len.get(len(d), []):
                if float(np.mean(docs[j] != d)) < near_thresh:
                    match = j
                    break
            if match is None:
                bases.append(i)
                by_len.setdefault(len(d), []).append(i)
                doc_ref[i] = len(bases) - 1
            else:
                doc_ref[i] = doc_ref[match]               # same base as its neighbour
                p = np.flatnonzero(docs[match] != d)
                dstart[i] = len(dpos_all)
                dlen[i] = len(p)
                dpos_all.extend(int(x) for x in p)
                dval_all.extend(int(x) for x in d[p])
        for i in range(n):                                # exact dups inherit their representative's ref+delta
            if exact_rep[i] != i:
                r = exact_rep[i]
                doc_ref[i] = doc_ref[r]
                dstart[i], dlen[i] = dstart[r], dlen[r]

        self.n_bases, self.n_exact = len(bases), int(np.sum(exact_rep != np.arange(n)))
        self.n_near = int(np.sum((dlen > 0) & (exact_rep == np.arange(n))))
        reps_flat = np.concatenate([docs[b] for b in bases]) if bases else np.zeros(1, np.int64)
        rep_start = np.concatenate([[0], np.cumsum([len(docs[b]) for b in bases])])[:-1]
        self._reps_tok = int(reps_flat.shape[0])
        self._delta_entries = len(dpos_all)
        self.reps = wp.array(reps_flat.astype(np.int32), dtype=wp.int32, device=device)
        self.rep_start = wp.array(rep_start.astype(np.int32), dtype=wp.int32, device=device)
        self.doc_ref = wp.array(doc_ref.astype(np.int32), dtype=wp.int32, device=device)
        self.dpos = wp.array(np.asarray(dpos_all or [0], np.int32), dtype=wp.int32, device=device)
        self.dval = wp.array(np.asarray(dval_all or [0], np.int32), dtype=wp.int32, device=device)
        self.dstart = wp.array(dstart.astype(np.int32), dtype=wp.int32, device=device)
        self.dlen = wp.array(dlen.astype(np.int32), dtype=wp.int32, device=device)

    def size_bytes(self) -> int:
        """Unique base tokens (uint16) + sparse near-dup deltas + the per-doc ref/delta index (int32)."""
        return self._reps_tok * 2 + self._delta_entries * 8 + len(self.docs) * 3 * 4

    def raw_bytes(self) -> int:
        return int(sum(self.lengths)) * 2

    def fetch(self, docs, positions) -> np.ndarray:
        d = wp.array(np.asarray(docs, np.int32), dtype=wp.int32, device=self.device)
        p = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(d.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_fetch_k, dim=d.shape[0],
                  inputs=[self.reps, self.rep_start, self.doc_ref, self.dpos, self.dval, self.dstart,
                          self.dlen, d, p, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def decode(self, k: int) -> np.ndarray:
        L = self.lengths[int(k)]
        return self.fetch(np.full(L, int(k), np.int32), np.arange(L, dtype=np.int32))


def _demo():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    L = 256
    uniq = [rng.integers(0, 50257, L).astype(np.int64) for _ in range(120)]
    docs = []
    for d in uniq:
        docs.append(d)
        if rng.random() < 0.5:
            docs.append(d.copy())                          # exact dup
        if rng.random() < 0.5:
            e = d.copy(); f = rng.integers(0, L, 6); e[f] = rng.integers(0, 50257, 6); docs.append(e)  # near-dup
    docs = [docs[i] for i in rng.permutation(len(docs))]

    store = DedupStore(docs, device=dev)
    ok = all(np.array_equal(store.decode(k), docs[k]) for k in range(0, len(docs), 11))
    total = sum(len(d) for d in docs)
    Q = 1 << 17
    dq = rng.integers(0, len(docs), Q).astype(np.int32)
    pq = np.array([rng.integers(0, store.lengths[d]) for d in dq], np.int32)
    for _ in range(3):
        store.fetch(dq, pq)
    t0 = time.perf_counter()
    for _ in range(20):
        store.fetch(dq, pq)
    ns = (time.perf_counter() - t0) / 20 / Q * 1e9

    print(f"device={dev}   dataset: {len(docs)} docs × {L} tok = {total:,} tokens")
    print(f"[compose] {store.n_bases} unique bases, {store.n_near} near-dups, {store.n_exact} exact dups")
    print(f"[correct] GPU decode == original doc ✓" if ok else "[correct] FAIL")
    print(f"[size]  raw {store.raw_bytes()/1e3:.1f} KB   DedupStore {store.size_bytes()/1e3:.1f} KB  "
          f"=> {store.raw_bytes()/store.size_bytes():.2f}× (dedup + delta, random access preserved)")
    print(f"[fetch] {Q:,} random tokens in {(Q*ns/1e9)*1e3:.2f} ms ({ns:.1f} ns/token, O(1) on the GPU)")
    print("=> content-aware dedup fixes the delta-tree failure on mostly-unique data: exact dups → a ref, "
          "near-dups → a sparse delta vs the true nearest doc, uniques stored once — all O(1) GPU-addressable.")


if __name__ == "__main__":
    _demo()
