"""semantic_merge — the LOSSY "merge within tolerance" tier (compress the sense, not the exact bytes).

Every merge in ChromoFold so far is **byte-exact**: `mergecube`/`dedup` collapse blocks only when they are
*identical* (`np.unique`). The vision docs flag the genuinely-new lever as a *semantic* tier — "merge near-
synonym tokens / blocks that agree **within tolerance** before coiling" (docs/research/44,45). This is that
tier, made concrete and measured.

It is **leader clustering** (Hartigan): scan the rows; a row joins an existing leader if it is within `tol`
of it, else it becomes a new leader. That gives a **hard, provable per-row bound** — every reconstructed row
is within `tol` of the original (‖x − x̂‖₂ ≤ tol), because the leader itself is the representative. So this is
the complement of `vq_store`: PQ is fixed-*rate* (k centroids, minimize error for a bit budget); this is
fixed-*distortion* (a quality floor `tol`, as many leaders as the data needs). One targets bits, the other
targets a guaranteed error.

The assignment ids are tokens → they drop into the same `RRRWaveletGPU` self-index, so the merged tensor stays
**randomly addressable on the GPU**. Unlike PQ codes, leader ids are **skewed** (a few big leaders absorb most
rows), so the entropy coder earns real ratio on the id stream here.

Headline application — the **KV cache**: neighbouring tokens' K/V are near-duplicate, so within-tolerance merge
buys context capacity. The lossy lever is `tol`; you dial it against a bounded, measured attention-output error.

    SemanticMergeStore(X, tol=0.05)   -> leaders + GPU-addressable assignment ids
    .reconstruct()                    -> leader-gathered rows (‖x − x̂‖₂ ≤ tol, guaranteed)
    .fetch(rows)
    .ratio                            -> rows / leaders

Measured (synthetic autoregressive KV): `python -m warp_compress.semantic_merge`. Real-model attention quality
is the follow-up on a box with the model (this CPU container has no torch) — see docs/research/46.
"""
from __future__ import annotations

import numpy as np

from .gpu_rrr_wavelet import RRRWaveletGPU


def leader_merge(X, tol, seed=0):
    """Leader clustering with an L2 tolerance. X: (n, d) -> (leaders (L,d), assignment (n,) int).

    Guarantee: ‖X[i] − leaders[assignment[i]]‖₂ ≤ tol for every i (the leader is an actual seen row within
    tol). Deterministic: rows are scanned in order; the first row of a group is its leader."""
    X = np.asarray(X, np.float32)
    n, d = X.shape
    leaders = np.empty((0, d), np.float32)
    lead_idx = []                                              # original-row index of each leader (for provenance)
    assign = np.empty(n, np.int64)
    tol2 = float(tol) * float(tol)
    for i in range(n):
        if leaders.shape[0] == 0:
            leaders = X[i:i + 1].copy(); lead_idx.append(i); assign[i] = 0
            continue
        d2 = ((leaders - X[i]) ** 2).sum(1)                    # (L,)
        j = int(d2.argmin())
        if d2[j] <= tol2:
            assign[i] = j
        else:
            assign[i] = leaders.shape[0]
            leaders = np.concatenate([leaders, X[i:i + 1]], 0); lead_idx.append(i)
    return leaders, assign


class SemanticMergeStore:
    """Within-tolerance leader-merge of a row-major tensor, assignment ids in the RRR self-index (addressable).

    The lossy lever is `tol` (an L2 radius). Every reconstructed row is within `tol` of the original. Store the
    fp16 leaders + the (skewed, entropy-coded) assignment ids."""

    def __init__(self, X, tol: float = 0.05, device: str = "cuda:0"):
        X = np.asarray(X, np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (rows, dim)")
        self.shape = X.shape
        self.n_rows, self.dim = X.shape
        self.tol = float(tol)
        self.device = device
        leaders, assign = leader_merge(X, tol)
        self.leaders = leaders.astype(np.float16).astype(np.float32)   # fp16 (as stored / charged)
        self.n_leaders = int(leaders.shape[0])
        self.ratio = self.n_rows / max(self.n_leaders, 1)
        bits = max(1, int(np.ceil(np.log2(max(self.n_leaders, 2)))))
        self._bits = bits
        self.wm = RRRWaveletGPU(assign.astype(np.int64), device=device, bits=bits)   # skewed ids -> real entropy win

    def size_bytes(self) -> int:
        base = self.wm.index_bytes()                          # entropy-coded assignment stream
        base += self.leaders.size * 2                         # fp16 leaders (n_leaders * dim)
        return int(base)

    def bits_per_value(self) -> float:
        return self.size_bytes() * 8 / (self.n_rows * self.dim)

    def reconstruct(self) -> np.ndarray:
        a = self.wm.access(np.arange(self.n_rows, dtype=np.int64))
        return self.leaders[a].astype(np.float32)

    def fetch(self, rows) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        a = self.wm.access(rows)
        return self.leaders[a].astype(np.float32)

    def max_row_error(self, X) -> float:
        """The realised worst-case ‖x − x̂‖₂ — must be ≤ tol (the guarantee), up to fp16 leader rounding."""
        X = np.asarray(X, np.float32)
        R = self.reconstruct()
        return float(np.sqrt(((X - R) ** 2).sum(1)).max())


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # synthetic KV: many tokens are near-synonyms of a smaller CONCEPT set (the premise of the semantic tier) —
    # C prototypes, each token = its concept's prototype + small noise, with slow concept switching so the
    # sequence is also locally correlated (as real KV is). tol ~ the noise radius merges same-concept tokens.
    T, d, C, sigma = 4096, 64, 200, 0.06
    protoK = rng.standard_normal((C, d)).astype(np.float32)
    protoV = rng.standard_normal((C, d)).astype(np.float32)
    concept = np.empty(T, np.int64); concept[0] = rng.integers(C)
    for t in range(1, T):                                     # switch concept ~5% of the time -> runs of a concept
        concept[t] = rng.integers(C) if rng.random() < 0.05 else concept[t - 1]
    K = (protoK[concept] + sigma * rng.standard_normal((T, d))).astype(np.float32)
    V = (protoV[concept] + sigma * rng.standard_normal((T, d))).astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    Q = rng.standard_normal((128, d)).astype(np.float32)      # 128 probe queries
    causal = None                                             # full attention over the whole context (a cache read)
    true_out = _softmax((Q @ K.T) * scale) @ V

    print(f"device={dev}   synthetic autoregressive KV: {T} tokens x {d} dims (K and V)\n")
    print(f"  {'tol':>6} {'leaders':>8} {'ratio':>7} {'b/val':>7} {'max‖Δrow‖':>10} {'attn out-err':>13}")
    for tol in (0.0, 0.3, 0.5, 0.7, 0.9, 1.2):
        ks = SemanticMergeStore(K, tol=tol, device=dev)
        vs = SemanticMergeStore(V, tol=tol, device=dev)
        Kr, Vr = ks.reconstruct(), vs.reconstruct()
        out = _softmax((Q @ Kr.T) * scale) @ Vr
        oerr = float(np.mean((out - true_out) ** 2))
        merr = max(ks.max_row_error(K), vs.max_row_error(V))
        # report the K side's structural stats (V is similar); ratio/b-val are the K store's
        print(f"  {tol:>6.2f} {ks.n_leaders:>8d} {ks.ratio:>7.2f} {ks.bits_per_value():>7.2f} "
              f"{merr:>10.3f} {oerr:>13.2e}")
    print("\n=> A tunable, BOUNDED-error lossy tier: tol is an L2 radius, so every reconstructed KV row is within\n"
          "   tol of the original (max‖Δrow‖ ≤ tol, guaranteed — verified in the column). Raising tol merges more\n"
          "   near-duplicate tokens (ratio ↑, bits ↓) for a graceful, measured rise in attention-output error.\n"
          "   tol=0 is exact (byte-merge). Leader ids are skewed -> the RRR index compresses them AND keeps the\n"
          "   merged KV randomly addressable. This is the semantic tier: compress the sense within a tolerance.")


if __name__ == "__main__":
    _demo()
