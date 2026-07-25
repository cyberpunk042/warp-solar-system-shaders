"""Importance-based KV eviction (H2O): keep-set composition, exact attention over survivors, measured win + regime."""
import numpy as np

import warp as wp

from warp_compress.kv_evict import HeavyHitterKVStore, heavy_hitter_indices, recent_indices
from warp_compress.kv_store import _softmax

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _spread_workload(T=512, d=64, nq=32, seed=0):
    """K/V plus queries that align with heavy-hitter keys SPREAD through the sequence (attention reaches back)."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((T, d)).astype(np.float32)
    V = rng.standard_normal((T, d)).astype(np.float32)
    heavy = rng.choice(T, 20, replace=False)
    Q = (rng.standard_normal((nq, d)) * 0.3).astype(np.float32)
    Q += K[heavy][rng.integers(0, 20, nq)] * 1.5
    return K, V, Q, heavy


def _attn(Q, K, V):
    return _softmax(Q @ K.T / np.sqrt(K.shape[1]), -1) @ V


def test_keep_set_includes_sinks_and_recent_and_is_budget_sized():
    K, V, Q, _ = _spread_workload(seed=1)
    T = K.shape[0]
    idx = heavy_hitter_indices(K, Q, budget=128, n_sink=4, n_recent=32)
    assert len(idx) == 128
    assert set(range(4)).issubset(idx)                          # sinks kept
    assert set(range(T - 32, T)).issubset(idx)                  # recent window kept
    assert np.all(np.diff(idx) > 0)                             # sorted, unique (chronological)


def test_attention_over_kept_is_exact():
    K, V, Q, _ = _spread_workload(seed=2)
    st = HeavyHitterKVStore(K, V, Q, budget=128, n_sink=4, n_recent=32, device=_DEV)
    ref = _attn(Q, K[st.kept_indices], V[st.kept_indices])      # attention restricted to survivors
    assert np.allclose(st.attention(Q), ref, atol=1e-6)         # store's forward == attend-over-kept


def test_heavy_hitter_beats_recency_when_attention_reaches_back():
    K, V, Q, _ = _spread_workload(seed=3)
    full = _attn(Q, K, V)
    err = lambda idx: float(np.mean((_attn(Q, K[idx], V[idx]) - full) ** 2))
    T = K.shape[0]
    h2o = heavy_hitter_indices(K, Q, budget=128, n_sink=4, n_recent=32)
    assert err(h2o) < 0.1 * err(recent_indices(T, 128))         # spread heavy hitters: importance-keep far better


def test_recency_ties_when_important_tokens_are_recent():
    # honest regime: if the queries only attend to the tail, recency is as good as H2O (no win to claim)
    rng = np.random.default_rng(4)
    T, d, nq = 512, 64, 32
    K = rng.standard_normal((T, d)).astype(np.float32)
    V = rng.standard_normal((T, d)).astype(np.float32)
    Q = (rng.standard_normal((nq, d)) * 0.3).astype(np.float32) + K[-16:][rng.integers(0, 16, nq)] * 1.5
    full = _attn(Q, K, V)
    err = lambda idx: float(np.mean((_attn(Q, K[idx], V[idx]) - full) ** 2))
    h2o = heavy_hitter_indices(K, Q, budget=64, n_sink=4, n_recent=32)
    rec = recent_indices(T, 64)
    assert err(rec) < 5 * err(h2o) and err(h2o) < 5 * err(rec)  # comparable — neither dominates


def test_compression_and_bits_per_value():
    K, V, Q, _ = _spread_workload(seed=5)
    fp16 = HeavyHitterKVStore(K, V, Q, budget=128, device=_DEV)
    q4 = HeavyHitterKVStore(K, V, Q, budget=128, quant_bits=4, device=_DEV)
    assert np.isclose(fp16.compression_x, 512 / 128)            # 4x fewer tokens
    assert q4.bits_per_value() < fp16.bits_per_value()          # eviction ∘ quant stacks: even fewer bits/value
    assert q4.attention(Q).shape == (Q.shape[0], K.shape[1])    # still attends


def test_quantized_survivors_are_the_reconstruction():
    K, V, Q, _ = _spread_workload(128, 64, 16, seed=6)
    st = HeavyHitterKVStore(K, V, Q, budget=48, quant_bits=4, device=_DEV)
    assert np.allclose(st.keys(), st._ksz.reconstruct().reshape(len(st.kept_indices), K.shape[1]), atol=1e-6)
    assert st.attention(Q).shape == (16, 64)


def test_budget_larger_than_sequence_keeps_all():
    K, V, Q, _ = _spread_workload(64, 32, 8, seed=7)
    st = HeavyHitterKVStore(K, V, Q, budget=999, device=_DEV)
    assert len(st.kept_indices) == 64 and st.compression_x == 1.0
    assert np.allclose(st.attention(Q), _attn(Q, K, V), atol=1e-6)   # keeping all == full attention
