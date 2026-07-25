"""kv_evict — importance-based KV-cache eviction (H2O / SnapKV / StreamingLLM): keep the tokens that matter.

`kv_store` shrinks the KV cache by *quantizing* every token; `semantic_merge` shrinks it by *merging* near-
duplicate tokens. This is the third, orthogonal, and now-dominant production lever: **drop most tokens entirely**
and keep only a budget of the ones attention actually uses. The keep-set is the H2O / SnapKV / StreamingLLM
recipe — a union of three pieces:

  * **attention sinks** — the first few tokens (StreamingLLM: models dump excess attention mass onto position 0-3;
    evicting them wrecks the softmax),
  * **a recent window** — the last `n_recent` tokens (locality),
  * **heavy hitters** — among the rest, the tokens with the highest *accumulated attention mass* over a batch of
    observed queries (H2O's "heavy hitter oracle").

Everything else is evicted. What survives is `budget` token rows out of `T`, a `T/budget×` cache reduction, and
the kept K/V rows drop straight into the succinct store (`weight_store` int-quant → `RRRWaveletGPU`) so they stay
GPU-addressable and entropy-coded on top of the eviction. Eviction and quantization **compose** (evict, then
quantize the survivors).

    st = HeavyHitterKVStore(K, V, Q, budget=128, n_sink=4, n_recent=64)   # Q: observed queries (n_q, d)
    st.attention(Qnew)     # attend Qnew over the kept K/V only
    st.kept_indices        # which original positions survived
    st.compression_x       # T / budget

Measured (`python -m warp_compress.kv_evict`): on a workload where the important tokens are **spread through the
sequence** (not just recent), importance-keep beats recency- and random-keep by orders of magnitude at the same
budget. **Honest regime note:** when the important tokens *are* the recent ones, plain recency ties it — the win
is real only when attention reaches back. Lossless is not the frame here (eviction is inherently lossy on the
dropped tokens); the guarantee is exact attention over the *kept* set + addressable kept rows. Run:
`python -m warp_compress.kv_evict`.
"""
from __future__ import annotations

import numpy as np

from .kv_store import _softmax


def heavy_hitter_indices(K, Q, budget: int, n_sink: int = 4, n_recent: int = 64) -> np.ndarray:
    """The H2O keep-set: sinks ∪ recent-window ∪ top-accumulated-attention heavy hitters, capped at `budget`.

    `Q` is a batch of observed queries; the per-key importance is the attention mass each key receives summed over
    those queries (H2O's heavy-hitter oracle). Returns sorted original positions (chronological order preserved)."""
    K = np.asarray(K, np.float32)
    Q = np.asarray(Q, np.float32)
    T, d = K.shape
    budget = int(min(budget, T))
    keep = set(range(min(n_sink, T)))                          # attention sinks
    keep |= set(range(T - min(n_recent, T), T))                # recent window
    if len(keep) < budget:
        score = _softmax(Q @ K.T / np.sqrt(d), -1).sum(0)      # accumulated attention mass per key
        rest = [j for j in range(T) if j not in keep]
        rest.sort(key=lambda j: -score[j])
        keep |= set(rest[: budget - len(keep)])                # fill remaining budget with heavy hitters
    return np.array(sorted(keep), np.int64)


def recent_indices(T: int, budget: int) -> np.ndarray:
    return np.arange(max(0, T - budget), T, dtype=np.int64)


class HeavyHitterKVStore:
    """Keeps `budget` token rows of a (T, d) K/V cache — sinks + recent + heavy hitters — and attends over only
    those. The survivors are optionally int-quantized into the succinct store, so eviction and quantization stack."""

    def __init__(self, K, V, Q, budget: int, n_sink: int = 4, n_recent: int = 64,
                 quant_bits: "int | None" = None, device: str = "cuda:0"):
        K = np.asarray(K, np.float32)
        V = np.asarray(V, np.float32)
        if K.shape != V.shape:
            raise ValueError(f"K {K.shape} and V {V.shape} must match")
        self.T, self.d = K.shape
        self.device = device
        self.budget = int(min(budget, self.T))
        self.kept_indices = heavy_hitter_indices(K, Q, self.budget, n_sink, n_recent)
        self.compression_x = self.T / max(1, len(self.kept_indices))
        self.quant_bits = quant_bits
        Kk, Vk = K[self.kept_indices], V[self.kept_indices]
        if quant_bits is None:
            self._K, self._V = Kk.copy(), Vk.copy()
            self._ksz = self._vsz = None
        else:                                                  # eviction ∘ quantization: store survivors quantized
            from .weight_store import QuantizedWeightStore
            self._ksz = QuantizedWeightStore(Kk, bits=quant_bits, huffman=True, device=device)
            self._vsz = QuantizedWeightStore(Vk, bits=quant_bits, huffman=True, device=device)
            self._K = self._ksz.reconstruct().reshape(Kk.shape)
            self._V = self._vsz.reconstruct().reshape(Vk.shape)

    def keys(self) -> np.ndarray:
        return self._K

    def values(self) -> np.ndarray:
        return self._V

    def attention(self, Qnew) -> np.ndarray:
        """Attend `Qnew` (n_q, d) over the kept K/V only — the compressed-cache forward pass."""
        Qnew = np.asarray(Qnew, np.float32)
        return _softmax(Qnew @ self._K.T / np.sqrt(self.d), -1) @ self._V

    def bits_per_value(self) -> float:
        """Effective bits per ORIGINAL value: only kept rows are stored (× their quant rate), amortized over all
        T·d values — so eviction shows up as a rate reduction on top of any quantization."""
        if self.quant_bits is None:
            stored_bits = len(self.kept_indices) * self.d * 2 * 16       # fp16 K + V
        else:
            stored_bits = (self._ksz.wm.index_bytes() + self._vsz.wm.index_bytes()) * 8
        return stored_bits / (self.T * self.d * 2)


def _demo():
    rng = np.random.default_rng(0)
    T, d, nq = 512, 64, 32
    K = rng.standard_normal((T, d)).astype(np.float32)
    V = rng.standard_normal((T, d)).astype(np.float32)
    heavy = rng.choice(T, 20, replace=False)                   # important tokens SPREAD through the sequence
    Q = (rng.standard_normal((nq, d)) * 0.3).astype(np.float32)
    Q += K[heavy][rng.integers(0, 20, nq)] * 1.5               # queries align with the heavy-hitter keys

    full = _softmax(Q @ K.T / np.sqrt(d), -1) @ V
    err = lambda idx: float(np.mean((_softmax(Q @ K[idx].T / np.sqrt(d), -1) @ V[idx] - full) ** 2))

    print(f"importance-based KV eviction (H2O) vs recency / random   (T={T} tokens, d={d}, {nq} queries)\n")
    print(f"  {'budget':>8}{'reduction':>11}{'H2O err':>12}{'recent err':>12}{'random err':>12}{'H2O vs recent':>15}")
    for budget in (64, 128):
        h2o = heavy_hitter_indices(K, Q, budget, 4, 32)        # recent window < budget, so heavy hitters get room
        rec = recent_indices(T, budget)
        rnd = np.sort(rng.choice(T, budget, replace=False))
        eh, er, ex = err(h2o), err(rec), err(rnd)
        print(f"  {budget:>8}{T/budget:>10.0f}x{eh:>12.3e}{er:>12.3e}{ex:>12.3e}{er/eh:>14.0f}x")

    st = HeavyHitterKVStore(K, V, Q, budget=128, quant_bits=4, device="cpu")
    print(f"\n  eviction ∘ int4-quant survivors:  {st.compression_x:.0f}× fewer tokens, "
          f"{st.bits_per_value():.2f} bits/value effective (over all T·d)")
    print(f"  kept {len(st.kept_indices)} tokens (sinks+recent+heavy); attention(Q) shape {st.attention(Q).shape}")
    print("\n=> Eviction keeps only the tokens attention uses — sinks + a recent window + the heavy hitters (top\n"
          "   accumulated attention). When the important tokens are SPREAD (attention reaches back), importance-keep\n"
          "   beats recency/random by orders of magnitude at the same budget; when they ARE recent, recency ties it\n"
          "   (honest regime note). Kept rows stay addressable + quantizable, so eviction stacks with the other levers.")


if __name__ == "__main__":
    import warp as wp
    wp.init()
    _demo()
