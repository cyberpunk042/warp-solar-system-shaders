"""gpu_delta — reference/delta cluster decode on the GPU, in Warp. The LoRA/adapter-library path, in VRAM.

`super_chromosome.build_delta` folds a cluster of near-duplicate sequences into one base + a tree of sparse
deltas (each leaf = base ⊕ the deltas on the right-turns of its root→leaf path). That's the format behind the
`delta` / `conversation` / `lora-library` presets — near-duplicate contexts, appended turns, a library of
adapters stored as diffs off a base. This puts its **decode** on the GPU: a batch of (leaf, position) queries,
each thread reconstructing one token by walking its leaf's path deltas deepest-first (deeper overwrites
shallower) — partial unfold of a compressed cluster, never leaving the GPU.

    fetch(leaves, positions)   one thread per query -> the reconstructed token, batched
    decode_leaf(leaf)          reconstruct a whole sequence (batched fetch over its positions)

Run: python -m warp_compress.gpu_delta
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .super_chromosome import GAP, build_delta

wp.init()


@wp.kernel
def _fetch_k(base: wp.array(dtype=wp.int32), nbase: int,
             dpos: wp.array(dtype=wp.int32), dval: wp.array(dtype=wp.int32),
             dstart: wp.array(dtype=wp.int32), dlen: wp.array(dtype=wp.int32),
             lp: wp.array(dtype=wp.int32), lp_start: wp.array(dtype=wp.int32),
             lp_len: wp.array(dtype=wp.int32),
             leaf_in: wp.array(dtype=wp.int32), pos_in: wp.array(dtype=wp.int32),
             out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    leaf = leaf_in[t]
    pos = pos_in[t]
    res = int(-1)
    if pos < nbase:
        res = base[pos]                                     # default: the base value at this position
    found = int(0)
    lst = lp_start[leaf]
    llen = lp_len[leaf]
    for kk in range(llen):
        k = llen - 1 - kk                                   # deepest delta first (it overwrites shallower)
        if found == 0:
            ni = lp[lst + k]
            ds = dstart[ni]
            dl = dlen[ni]
            lo = int(0)                                     # binary search this node's sorted delta positions
            hi = dl
            while lo < hi:
                mid = (lo + hi) >> 1
                if dpos[ds + mid] < pos:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < dl and dpos[ds + lo] == pos:
                res = dval[ds + lo]
                found = 1
    out[t] = res


class GPUDeltaCluster:
    """A reference/delta-compressed cluster, resident on a Warp device; batched fetch reconstructs on the GPU."""

    def __init__(self, sequences, device: str = "cuda:0"):
        self.device = device
        self.sc = build_delta(sequences)
        self.lengths = list(self.sc.lengths)

        base = np.asarray(self.sc.base.decompress(), np.int64)
        self._nbase = int(base.shape[0])

        # index every internal node's delta; record, per leaf, the ordered node indices applied on right-turns
        node_pos, node_val, dstart, dlen = [], [], [], []
        node_idx: dict = {}

        def reg(node) -> int:
            key = id(node)
            if key not in node_idx:
                node_idx[key] = len(dstart)
                dstart.append(sum(dlen))
                dlen.append(int(node.pos.shape[0]))
                node_pos.append(np.asarray(node.pos, np.int64))
                node_val.append(np.asarray(node.val, np.int64))
            return node_idx[key]

        leaf_apply: dict = {}

        def rec(node, applied):
            if node.pos is None:                            # leaf
                leaf_apply[node.leaf_id] = list(applied)
                return
            idx = reg(node)
            rec(node.left, applied)                         # left turn: this node's delta does NOT apply
            rec(node.right, applied + [idx])                # right turn: it does (deeper => later => wins)

        rec(self.sc.root, [])

        dpos = np.concatenate(node_pos) if node_pos else np.zeros(1, np.int64)
        dvals = np.concatenate(node_val) if node_val else np.zeros(1, np.int64)
        lp, lp_start, lp_len = [], [], []
        for leaf in range(self.sc.n_leaves):
            path = leaf_apply.get(leaf, [])
            lp_start.append(len(lp))
            lp_len.append(len(path))
            lp.extend(path)

        self._delta_entries = int(dpos.shape[0]) if node_pos else 0
        self.base = wp.array(base.astype(np.int32), dtype=wp.int32, device=device)
        self.dpos = wp.array(dpos.astype(np.int32), dtype=wp.int32, device=device)
        self.dval = wp.array(dvals.astype(np.int32), dtype=wp.int32, device=device)
        self.dstart = wp.array(np.asarray(dstart or [0], np.int32), dtype=wp.int32, device=device)
        self.dlen = wp.array(np.asarray(dlen or [0], np.int32), dtype=wp.int32, device=device)
        self.lp = wp.array(np.asarray(lp or [0], np.int32), dtype=wp.int32, device=device)
        self.lp_start = wp.array(np.asarray(lp_start, np.int32), dtype=wp.int32, device=device)
        self.lp_len = wp.array(np.asarray(lp_len, np.int32), dtype=wp.int32, device=device)

    @property
    def n_leaves(self) -> int:
        return self.sc.n_leaves

    def size_bytes(self) -> int:
        """Compressed footprint: the base sequence + the sparse deltas (positions + values)."""
        return self._nbase * 4 + self._delta_entries * 8

    def fetch(self, leaves, positions) -> np.ndarray:
        lv = wp.array(np.asarray(leaves, np.int32), dtype=wp.int32, device=self.device)
        ps = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(lv.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_fetch_k, dim=lv.shape[0],
                  inputs=[self.base, self._nbase, self.dpos, self.dval, self.dstart, self.dlen,
                          self.lp, self.lp_start, self.lp_len, lv, ps, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def decode_leaf(self, leaf: int) -> np.ndarray:
        """Reconstruct a whole sequence on the GPU (batched fetch over every position)."""
        L = self.lengths[int(leaf)]
        return self.fetch(np.full(L, int(leaf), np.int32), np.arange(L, dtype=np.int32))


def _demo():
    import time

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # a LoRA-library / near-duplicate-context flavour: many related strands over a base (here ACGT, but any V)
    base = rng.integers(0, 4, size=4000)
    seqs = []
    for _ in range(64):
        s = base.copy()
        f = rng.integers(0, base.shape[0], size=40)        # ~1% divergence per member
        s[f] = rng.integers(0, 4, size=40)
        seqs.append(s)

    gd = GPUDeltaCluster(seqs, device=dev)

    # 1) correctness: batched random fetch == the originals; a full leaf decode round-trips
    Q = 200000
    ql = rng.integers(0, len(seqs), Q).astype(np.int32)
    qp = np.array([rng.integers(0, gd.lengths[l]) for l in ql], np.int32)
    got = gd.fetch(ql, qp)
    truth = np.array([seqs[ql[i]][qp[i]] for i in range(Q)])
    assert np.array_equal(got, truth), "GPU delta fetch mismatch"
    assert all(np.array_equal(gd.decode_leaf(k), seqs[k]) for k in (0, 7, 33, 63)), "leaf decode mismatch"

    # 2) size + throughput
    total = sum(len(s) for s in seqs)
    gd.fetch(ql[:1000], qp[:1000])                          # warm up
    t0 = time.perf_counter(); gd.fetch(ql, qp); dt = time.perf_counter() - t0
    print(f"device={dev}   cluster {len(seqs)} × {base.shape[0]} tokens = {total:,}   compressed {gd.size_bytes()/1e3:.1f} KB "
          f"({total*4/gd.size_bytes():.2f}× vs raw int32)")
    print(f"[correct] batched GPU fetch == originals ✓   whole-leaf decode round-trips ✓")
    print(f"[speed]   {Q:,} tokens reconstructed in {dt*1e3:.1f} ms = {Q/dt/1e6:.0f} M tok/s (batched, in VRAM)")
    print("=> the reference/delta cluster (LoRA library, conversation history, near-dup contexts) decodes on "
          "the GPU: each token is base[pos] overridden by the deepest path delta that touches it. Partial\n"
          "   unfold of a compressed cluster, never leaving the GPU — the delta/lora-library presets, running.")


if __name__ == "__main__":
    _demo()
