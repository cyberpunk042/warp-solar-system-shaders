"""Auto-tune: profile a data sample and pick the right transform, build-driven and self-correcting."""
import numpy as np

from warp_compress.autotune import analyze, plan
from warp_compress.chromofold import auto


def _mixed_prompts(K=100, seed=0):
    rng = np.random.default_rng(seed)
    prompts = [rng.integers(0, 50257, int(rng.integers(300, 500))).astype(np.int64) for _ in range(4)]
    return [np.concatenate([prompts[int(rng.integers(0, 4))], rng.integers(0, 50257, 16).astype(np.int64)])
            for _ in range(K)]


def _near_dups(K=48, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, 600)
    out = []
    for _ in range(K):
        s = base.copy(); s[rng.integers(0, 600, 6)] = rng.integers(0, 4, 6); out.append(s)
    return out


def _markov(n=80000, V=32, seed=2):
    rng = np.random.default_rng(seed)
    trans = rng.dirichlet(np.ones(V) * 0.3, size=V)
    s = np.empty(n, np.int64); s[0] = 0
    for i in range(1, n):
        s[i] = rng.choice(V, p=trans[s[i - 1]])
    return s


def test_mixed_prompt_batch_picks_seed():
    cfg, why, ach = plan(_mixed_prompts(), intent="serving")
    assert cfg.transform == "seed"
    assert ach["chromofold"] < ach["raw"]                 # and it actually compresses


def test_near_duplicate_batch_picks_delta():
    cfg, why, ach = plan(_near_dups())
    assert cfg.transform == "delta"
    assert ach["chromofold"] < ach["gzip"]                # delta beats gzip on aligned near-dups


def test_skewed_stream_picks_bwt_and_beats_raw():
    cfg, why, ach = plan(_markov())
    assert cfg.transform == "bwt"
    assert ach["chromofold"] < ach["raw"]


def test_uniform_noise_downgrades_to_raw():
    rng = np.random.default_rng(3)
    cfg, why, ach = plan(rng.integers(0, 50257, 60000).astype(np.int64))
    assert cfg.transform == "none"                        # honest: no ratio win, don't recommend ChromoFold
    assert "no ratio" in why[-1] or "no exploitable" in why[-1]


def test_search_intent_keeps_bwt_for_capability():
    cfg, _, _ = plan(_markov(seed=5), intent="search")
    assert cfg.transform == "bwt"                         # kept for count/locate/predict even if a codec were smaller


def test_analyze_reports_structure():
    p = analyze(_mixed_prompts(seed=6))
    assert p.kind == "batch" and p.prefix_seeds == 4 and p.prefix_share > 2.0


def test_config_auto_entry_point():
    assert auto(_near_dups(seed=7)).transform == "delta"  # ChromoFoldConfig-level auto() dispatches to plan()
