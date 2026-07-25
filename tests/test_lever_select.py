"""Lever selector: build a candidate per lever, measure, pick the best for the goal + budget (build-driven)."""
import numpy as np

import warp as wp

from warp_compress.lever_select import measure_candidates, pick_from, select_weight_codec, _structure

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _lowrank(shape=(512, 128), rank=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((shape[0], rank)).astype(np.float32)
    V = rng.standard_normal((rank, shape[1])).astype(np.float32)
    return ((U @ V) / np.sqrt(rank * shape[1]) + noise * rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def _fullrank(shape=(512, 128), seed=0):
    return (np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def _byname(rows):
    return {r["name"]: r for r in rows}


def test_returns_usable_pick():
    W = _fullrank(seed=1)
    x = np.random.default_rng(2).standard_normal((16, 128)).astype(np.float32)
    pick = select_weight_codec(W, x=x, device=_DEV)
    assert pick.store.reconstruct().shape == W.shape           # the chosen codec actually decodes W
    assert pick.bits > 0 and pick.name in {r[0] for r in pick.table}


def test_budget_is_respected():
    W = _fullrank(seed=3)
    x = np.random.default_rng(4).standard_normal((16, 128)).astype(np.float32)
    pick = select_weight_codec(W, x=x, target_bits=2.5, device=_DEV)
    assert pick.bits <= 2.5 * 1.05                             # stays within the bit budget


def test_lowrank_lever_helps_on_lowrank_tensor():
    # on a strongly low-rank tensor, the low-rank lever should beat int4 AND PQ on output error
    W = _lowrank(rank=4, noise=0.3, seed=5)
    x = np.random.default_rng(6).standard_normal((32, 128)).astype(np.float32)
    rows = _byname(measure_candidates(W, x=x, device=_DEV))
    lr = next(v for k, v in rows.items() if k.startswith("lowrank"))
    assert lr["out_err"] < rows["int4"]["out_err"]
    assert lr["out_err"] < rows["PQ 4x8"]["out_err"]


def test_lowrank_lever_does_not_help_on_fullrank_tensor():
    # on a full-rank tensor, low-rank offers nothing -> int4 (fewer or similar bits) should beat it
    W = _fullrank(seed=7)
    x = np.random.default_rng(8).standard_normal((32, 128)).astype(np.float32)
    rows = _byname(measure_candidates(W, x=x, device=_DEV))
    lr = next(v for k, v in rows.items() if k.startswith("lowrank"))
    assert rows["int4"]["out_err"] < lr["out_err"]             # element lever wins on unstructured weights


def test_structure_signal_discriminates():
    lr = _structure(_lowrank(rank=4, noise=0.3, seed=9))["sv_energy"][8]
    fr = _structure(_fullrank(seed=10))["sv_energy"][8]
    assert lr > 0.4 and fr < 0.2                               # top-8 energy separates low-rank from full-rank


def test_pick_is_argmin_within_budget():
    W = _lowrank(rank=6, seed=11)
    x = np.random.default_rng(12).standard_normal((24, 128)).astype(np.float32)
    rows = measure_candidates(W, x=x, device=_DEV)
    pick = pick_from(rows, W, x=x, target_bits=3.6)
    within = [r for r in rows if r["bits"] <= 3.6 * 1.05]
    best = min(r["out_err"] for r in within)
    assert pick.out_err <= best * 1.05                        # picked (near-)lowest output error within budget
    assert "structure" in pick.reason and "singular values" in pick.reason


def test_table_is_sorted_and_complete():
    W = _fullrank(seed=13)
    x = np.random.default_rng(14).standard_normal((16, 128)).astype(np.float32)
    rows = measure_candidates(W, x=x, device=_DEV)
    pick = pick_from(rows, W, x=x)
    assert len(pick.table) == len(rows)                       # every built candidate is on the receipts
    scores = [oerr for _, _, _, oerr in pick.table]
    assert scores == sorted(scores)                           # sorted best-first by output error
