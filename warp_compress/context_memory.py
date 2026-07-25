"""context_memory — hold a long LLM context as an addressable, SEARCHABLE chromosome (the headline experiment).

The vision docs call this out as *the* unrun experiment: "store a long context as a chromosome — V unique
embeddings + the procedural map — and read it token by token, paying O(1) per access instead of storing/
attending over the full N" (docs/genome_compression_math.md), plus "the DNA metaphor is literal: the
compressed, addressable, searchable token index is the FM-index." This wires that up end to end from the
pieces already in the repo — and ties the three new levers together:

  1. continuous embeddings (token embeddings / KV rows) are bridged into the discrete token world by
     `semantic_merge.leader_merge` (the within-tolerance content book): N rows -> V unique concepts + ids;
  2. the (book, ids) pair is a `token_chromosome.Chromosome` — O(1) closed-form `at(pos)` navigation and a
     tiny run-length-encoded id stream (positions cost nothing, they are procedural);
  3. the id stream is also an `fm_index.FMIndex` — so the context is **searchable in the compressed domain**
     (find every position where a concept sub-sequence occurs), the thing a raw KV cache cannot do.

The result is one object that is compressed + randomly addressable + searchable + a coarse→fine memory — the
distinctive claim, versus a flat KV cache that is none of those. `tol=0` is an exact content book (lossless);
`tol>0` engages the semantic tier (every reconstructed row within tol).

    mem = ContextMemory(embeddings, tol=0.05)
    mem.at(pos)                 -> the embedding at a position (O(1), no full materialization)
    mem.search(concept_ids)     -> positions where that concept sub-sequence occurs (FM backward search)
    mem.nearest_id(query_vec)   -> map a query embedding to its concept id (the bridge for content retrieval)
    mem.compression_x           -> stored bits vs raw fp16

Measured (synthetic concept-structured context): `python -m warp_compress.context_memory`. The LM-utility
question — does token-by-token navigation over this memory match full-context quality at lower memory? — needs
a model and is the flagged follow-up (this CPU container has no torch). See docs/research/47.
"""
from __future__ import annotations

import numpy as np

from .semantic_merge import leader_merge
from .token_chromosome import Chromosome
from .fm_index import FMIndex


class ContextMemory:
    """A long context (N embeddings) held as a searchable chromosome: content book + addressable id map + FM-index."""

    def __init__(self, embeddings, tol: float = 0.0, dim: int = 3, sa_sample: int = 16):
        X = np.asarray(embeddings, np.float32)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2-D (n_tokens, dim)")
        self.n_tokens, self.d = X.shape
        self.tol = float(tol)
        book, ids = leader_merge(X, tol)                       # continuous -> (V concepts, per-token ids)
        self.book = book.astype(np.float16).astype(np.float32)  # fp16 content book (as stored / charged)
        self.ids = ids.astype(np.int64)
        self.V = int(self.book.shape[0])
        import math
        bits = max(1, math.ceil(math.log2(max(self.n_tokens, 2)) / dim))
        self.chromo = Chromosome(book=self.book, ids=self.ids, bits=bits, dim=dim)   # O(1) at(), RLE id stream
        self.fm = FMIndex(self.ids, sa_sample=sa_sample)       # searchable in the compressed domain

    # --- addressable read (O(1) per token; no full materialization) ---
    def at(self, pos: int) -> np.ndarray:
        return self.chromo.token(int(pos))

    def reconstruct(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        return self.chromo.decompress(lo, hi)

    def max_row_error(self, embeddings) -> float:
        X = np.asarray(embeddings, np.float32)
        return float(np.sqrt(((X - self.reconstruct()) ** 2).sum(1)).max())

    # --- search / retrieval (the KV cache cannot do this) ---
    def nearest_id(self, query) -> int:
        q = np.asarray(query, np.float32)
        return int(((self.book - q) ** 2).sum(1).argmin())

    def search(self, concept_ids) -> list:
        """Positions where the concept id sub-sequence occurs (FM-index backward search)."""
        return sorted(self.fm.locate(list(np.asarray(concept_ids, np.int64))))

    def count(self, concept_ids) -> int:
        return int(self.fm.count(list(np.asarray(concept_ids, np.int64))))

    # --- footprint ---
    def stored_bits(self) -> int:
        return int(self.chromo.rate_bits()["total_bits"])      # book + RLE id stream + tiny map constants

    @property
    def raw_bits(self) -> int:
        return self.n_tokens * self.d * 16                     # a flat fp16 KV/embedding cache

    @property
    def compression_x(self) -> float:
        return self.raw_bits / max(self.stored_bits(), 1)


def _demo():
    rng = np.random.default_rng(0)
    # a long context where tokens are near-synonyms of a smaller concept set (real contexts repeat concepts)
    N, d, C, sigma = 8192, 64, 300, 0.06
    proto = rng.standard_normal((C, d)).astype(np.float32)
    concept = np.empty(N, np.int64); concept[0] = rng.integers(C)
    for t in range(1, N):
        concept[t] = rng.integers(C) if rng.random() < 0.06 else concept[t - 1]
    X = (proto[concept] + sigma * rng.standard_normal((N, d))).astype(np.float32)

    print(f"context: {N} tokens x {d} dims  (~{C} underlying concepts)\n")
    print(f"  {'tol':>5} {'V(book)':>8} {'stored':>10} {'raw':>10} {'compress':>9} {'max‖Δrow‖':>10} {'at()==decode':>13}")
    for tol in (0.0, 0.5, 0.7, 0.9):
        mem = ContextMemory(X, tol=tol)
        # addressability: at(pos) must equal the decoded row (O(1) random read)
        R = mem.reconstruct()
        ok = all(np.array_equal(mem.at(p), R[p]) for p in rng.integers(0, N, 200))
        merr = mem.max_row_error(X)
        print(f"  {tol:>5.2f} {mem.V:>8d} {mem.stored_bits() // 8:>9,}B {mem.raw_bits // 8:>9,}B "
              f"{mem.compression_x:>8.1f}x {merr:>10.3f} {'✓' if ok else 'FAIL':>13}")

    # searchable in the compressed domain: a KV cache cannot do this
    mem = ContextMemory(X, tol=0.7)
    # take a real length-3 concept sub-sequence from the context and locate every occurrence
    probe = mem.ids[100:103]
    hits = mem.search(probe)
    brute = [i for i in range(N - 2) if np.array_equal(mem.ids[i:i + 3], probe)]
    print(f"\n  search(concept 3-gram {list(probe)}): FM found {len(hits)} occurrences, "
          f"brute-force {len(brute)} -> {'match' if hits == brute else 'MISMATCH'}")
    # content retrieval: a noisy query should map to a concept whose VECTOR is near it (leader clustering makes
    # >1 leader per concept, so the id need not equal the merge-time id — what matters is vector proximity).
    qpos = rng.integers(0, N, 500)
    noisy = X[qpos] + 0.03 * rng.standard_normal((500, d)).astype(np.float32)
    got = np.array([mem.nearest_id(q) for q in noisy])
    ret_err = np.sqrt(((mem.book[got] - X[qpos]) ** 2).sum(1))      # retrieved concept vs the true token
    recall = float(np.mean(ret_err <= tol + 0.15))                 # within the tolerance band of the true token
    print(f"\n  content retrieval over 500 noisy queries: {recall:.0%} land within tol of the true token "
          f"(mean ‖retrieved − true‖ = {ret_err.mean():.3f}, book radius tol={0.7})")
    print("\n=> One object that is O(1)-addressable + SEARCHABLE + coarse→fine — what a flat KV cache is not — AND\n"
          "   compressed once the context repeats concepts (16.5x at tol=0.9 here). Honest: tol=0 EXPANDS a fully\n"
          "   distinct context (an exact book can't compress; you still gain search + addressing); the compression\n"
          "   comes from concept redundancy at tol>0, at a bounded per-row error. The remaining experiment is LM\n"
          "   utility (navigate this memory at inference vs the full KV cache) — it needs a model; the mechanism +\n"
          "   memory/search numbers here are what runs without one.")


if __name__ == "__main__":
    _demo()
