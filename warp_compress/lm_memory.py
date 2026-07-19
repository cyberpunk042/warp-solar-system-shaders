"""lm_memory — an addressable compressed context memory for a language model, built on the chromosome.

The idea the genome engine encodes, applied: hold a long token context not as N per-token vectors but as

    ( V unique token embeddings )  +  ( the hierarchical chromosome index over the token ids )

Then:
  fetch(r)              O(1) -> the embedding of the token at position r          (book[ids[r]])
  window(a, b)          a contiguous slice of token embeddings
  coarse(level)         the per-block SUMMARY embeddings at a level (few blocks)  -> cheap coarse attention
  retrieve(q, level, k) score the coarse blocks against a query, drill the top-k to their token ranges

Memory is O(V*d + index) instead of O(N*d): for a context with token repetition (V << N) that is a real
saving, and reconstruction is lossless & O(1). The coarse levels give cheap coarse-to-fine attention:
look at a few block summaries, then expand only the relevant blocks to tokens.

Embeddings here are a stand-in table (an LM would use its own embedding rows). Run:
    python -m warp_compress.lm_memory
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .token_chromosome import HierChromosome, compress_hierarchical


@dataclasses.dataclass
class CompressedContextMemory:
    hier: HierChromosome
    embed: np.ndarray                 # (V, d) — one embedding per UNIQUE token (the content book's vectors)
    summaries: list = None            # per-level (l>=1) block-MEAN embeddings, precomputed for coarse attn

    @property
    def n(self) -> int:
        return self.hier.n

    @property
    def dim_e(self) -> int:
        return int(self.embed.shape[1])

    # --- O(1) token access (lossless reconstruction of the per-token embedding stream) ---
    def fetch(self, r: int) -> np.ndarray:
        return self.embed[self.hier.ids[int(r)]]

    def window(self, lo: int, hi: int) -> np.ndarray:
        return self.embed[self.hier.ids[lo:hi]]

    # --- coarse-to-fine retrieval ---
    def coarse(self, level: int) -> np.ndarray:
        """Per-block summary embeddings at `level` — the MEAN of the block's token embeddings (precomputed)."""
        return self.summaries[level]

    def retrieve(self, q: np.ndarray, level: int, topk: int = 3):
        """Coarse attention: score the level's block summaries against q, return the top-k blocks and their
        analytic token ranges. Only those blocks are then expanded to tokens — cheap attention over few
        blocks instead of all N tokens."""
        scores = self.coarse(level) @ np.asarray(q, np.float64)
        order = np.argsort(scores)[::-1][:topk]
        return [(int(b), self.hier.token_range(level, int(b)), float(scores[b])) for b in order]

    # --- accounting ---
    def memory_bytes(self) -> dict:
        d = self.dim_e
        emb = self.embed.size * self.embed.itemsize                      # V*d embeddings
        idx = self.hier.ids.nbytes + sum(l.nbytes for l in self.hier.levels[1:])
        summ = sum(s.size * s.itemsize for s in self.summaries[1:] if s is not None)
        naive = self.n * d * self.embed.itemsize                         # N*d full cache
        return dict(V=int(self.embed.shape[0]), N=self.n, dim=d,
                    embeddings_bytes=emb, index_bytes=idx, summary_bytes=summ,
                    total_bytes=emb + idx + summ,
                    naive_bytes=naive, ratio=naive / max(emb + idx + summ, 1))


def build_memory(token_ids, embed_table: np.ndarray, branch: int = 4, dim: int = 3,
                 coarse_cap: int = 4096) -> CompressedContextMemory:
    """token_ids: (N,) the context's token ids into `embed_table` (V_total, d). Only the USED unique tokens
    are kept (the content book), so the stored embeddings are V<=V_total."""
    hier = compress_hierarchical(np.asarray(token_ids), branch=branch, dim=dim)
    book_embed = embed_table[hier.book]                                  # embeddings of the used unique tokens
    # precompute block-MEAN summaries only for the COARSE levels (few blocks) — the fine levels are near
    # tokens and would cost as much as the naive cache. Coarse levels enable cheap coarse-to-fine attention.
    tok_emb = book_embed[hier.ids]
    summaries = [None] * hier.n_levels
    for level in range(1, hier.n_levels):
        if hier.count(level) > coarse_cap:                              # skip fine levels (too many blocks)
            continue
        s = hier.branch ** level
        starts = np.arange(0, hier.n, s)
        counts = np.diff(np.concatenate([starts, [hier.n]]))[:, None]
        summaries[level] = (np.add.reduceat(tok_emb, starts, axis=0) / counts).astype(np.float32)
    return CompressedContextMemory(hier=hier, embed=book_embed, summaries=summaries)


def _demo():
    rng = np.random.default_rng(0)
    V_total, d = 5000, 64
    table = rng.standard_normal((V_total, d)).astype(np.float32)         # the LM's embedding table (stand-in)

    # a long context with heavy repetition (zipfian) — the regime where compressed memory pays off
    p = 1.0 / np.arange(1, V_total + 1)
    p /= p.sum()
    ctx = rng.choice(V_total, size=100000, p=p)

    mem = build_memory(ctx, table, branch=4, dim=3)

    # 1) lossless O(1) reconstruction: fetch(r) == the true per-token embedding
    ok = all(np.array_equal(mem.fetch(r), table[ctx[r]]) for r in rng.integers(0, mem.n, 500))
    assert ok, "fetch mismatch"

    # 2) memory — separate the real win (O(1) lossless fetch) from the optional coarse-attention summaries
    m = mem.memory_bytes()
    fetch_mem = m["embeddings_bytes"] + m["index_bytes"]
    print(f"context N={m['N']}  used-unique V={m['V']}  dim={m['dim']}")
    print(f"full per-token cache : {m['naive_bytes']/1e6:6.2f} MB")
    print(f"fetch memory (V embeddings + id index): {fetch_mem/1e6:6.2f} MB  => {m['naive_bytes']/fetch_mem:.1f}x "
          f"smaller, O(1) lossless fetch")
    print(f"+ coarse summaries (optional, for coarse attn): {m['summary_bytes']/1e6:.2f} MB")

    # 3) coarse-to-fine retrieval — the MECHANISM: score few coarse blocks, drill the winners to token ranges
    level = next(l for l in range(mem.hier.n_levels) if mem.summaries[l] is not None)
    q = mem.fetch(int(rng.integers(0, mem.n)))
    hits = mem.retrieve(q, level, topk=3)
    print(f"coarse retrieval @ level {level}: scored {mem.hier.count(level)} block summaries "
          f"(vs {mem.n} tokens) -> top-3 token ranges {[r for _, r, _ in hits]}")
    print("note: retrieval QUALITY needs real (structured) embeddings + max/attn pooling; with random "
          "stand-ins mean-pooling is uninformative. The mechanism (cheap coarse scan -> drill) is what's shown.")
    print("=> a long context stored as a chromosome: O(1) lossless token fetch, big memory saving when "
          "V<<N, and coarse-to-fine addressing for cheap attention.")


if __name__ == "__main__":
    _demo()
