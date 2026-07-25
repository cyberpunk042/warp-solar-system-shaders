"""lever_select — profile a weight tensor, BUILD candidate codecs across every lever, pick the best (measured).

Five weight levers now exist — scalar int (`weight_store`), product quantization (`vq_store`), residual VQ
(`rvq_store`), and low-rank + residual (`lowrank_store`) — each a different trade. Which one is best depends on
the tensor (its rank structure, its outliers) and the target (a bit budget vs a quality floor). Rather than
guess, this does what `autotune` does elsewhere in ChromoFold: it **builds a candidate from each lever and
measures it**, then picks — build-driven and self-correcting, never a heuristic label.

    pick = select_weight_codec(W, x=activations, target_bits=3.0)
    pick.store        # the chosen codec (GPU-addressable, lossless over its lossy lever)
    pick.name         # e.g. "lowrank16+PQ"
    pick.reason       # one line: why it won, with the structure signal that explains it
    pick.table        # every candidate's (name, bits, mse, out_err) — the receipts

Selection: minimise output error (‖x(W−Ŵ)‖ — what matters for a model) when activations `x` are given, else
MSE; subject to `bits ≤ target_bits` if a budget is set. Ties (within 5%) break toward fewer bits. A cheap SVD
energy profile is reported as the structure signal that explains the pick (low-rank tensors -> a low-rank lever
wins; full-rank -> an element lever).

Measured on three tensor archetypes: `python -m warp_compress.lever_select`.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Pick:
    name: str
    store: object
    bits: float
    mse: float
    out_err: float
    reason: str
    table: list


def _structure(W) -> dict:
    """Cheap SVD energy profile: what fraction of the tensor's energy the top-k singular values hold."""
    s = np.linalg.svd(np.asarray(W, np.float32), compute_uv=False)
    e = s * s
    tot = float(e.sum()) + 1e-12
    n = int(s.shape[0])
    frac = {k: float(e[:k].sum()) / tot for k in (8, 16, 32) if k <= n}
    return {"sv_energy": frac, "n_sv": n}


def measure_candidates(W, x=None, device: str = "cuda:0", seed: int = 0, kmeans_iters: int = 10) -> list:
    """Build ONE candidate per lever and measure (bits, MSE, output error). Built once; selection reuses this."""
    from .weight_store import QuantizedWeightStore
    from .vq_store import ProductQuantizedWeightStore
    from .rvq_store import ResidualVQWeightStore
    from .lowrank_store import LowRankWeightStore
    W = np.asarray(W, np.float32)
    r = min(16, *W.shape)
    builders = [
        ("int8", lambda: QuantizedWeightStore(W, bits=8, huffman=True, device=device)),
        ("int4", lambda: QuantizedWeightStore(W, bits=4, huffman=True, device=device)),
        ("int4+outliers", lambda: QuantizedWeightStore(W, bits=4, huffman=True, outliers=0.01, device=device)),
        ("PQ 4x8", lambda: ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=device,
                                                       kmeans_iters=kmeans_iters, seed=seed)),
        ("RVQ 2x4", lambda: ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=2, device=device,
                                                  kmeans_iters=kmeans_iters, seed=seed)),
        (f"lowrank{r}+PQ", lambda: LowRankWeightStore(W, rank=r, residual="pq", device=device,
                                                      pq_subdim=4, pq_bits=8)),
    ]
    fp32_out = (x @ W.T) if x is not None else None
    rows = []
    for name, build in builders:
        try:
            st = build()
        except Exception:
            continue                                            # a lever that doesn't fit this tensor is skipped
        R = st.reconstruct()
        mse = float(np.mean((R - W) ** 2))
        oerr = float(np.mean((x @ R.T - fp32_out) ** 2)) if x is not None else mse
        rows.append({"name": name, "store": st, "bits": float(st.bits_per_weight()), "mse": mse, "out_err": oerr})
    return rows


def pick_from(rows, W, x=None, target_bits: "float | None" = None) -> Pick:
    """Select the best measured candidate for the goal + budget, with a structure-signal reason."""
    score = lambda r: r["out_err"] if x is not None else r["mse"]
    budget_note = ""
    pool = rows
    if target_bits is not None:
        within = [r for r in rows if r["bits"] <= target_bits * 1.05]
        if within:
            pool, budget_note = within, f" within the {target_bits:.1f} b/w budget"
        else:
            pool = sorted(rows, key=lambda r: r["bits"])[:1]
            budget_note = f" (nothing met {target_bits:.1f} b/w; picked smallest)"
    best = min(pool, key=score)
    near = [r for r in pool if score(r) <= score(best) * 1.05]   # tie-break toward fewer bits
    best = min(near, key=lambda r: r["bits"])

    prof = _structure(W)
    k0 = min(prof["sv_energy"]) if prof["sv_energy"] else None   # the smallest-k energy is the low-rank signal
    struct = (f"top-{k0} of {prof['n_sv']} singular values hold {prof['sv_energy'][k0]:.0%} of the energy"
              if k0 is not None else "no SVD profile")
    metric = "output error" if x is not None else "MSE"
    reason = (f"{best['name']}: lowest {metric} ({score(best):.2e}) at {best['bits']:.2f} b/w{budget_note}; "
              f"structure — {struct}")
    table = [(r["name"], round(r["bits"], 2), r["mse"], r["out_err"]) for r in sorted(rows, key=score)]
    return Pick(best["name"], best["store"], best["bits"], best["mse"], best["out_err"], reason, table)


def select_weight_codec(W, x=None, target_bits: "float | None" = None, device: str = "cuda:0", seed: int = 0) -> Pick:
    """Convenience: measure every lever's candidate for W, then pick the best for the goal + budget."""
    rows = measure_candidates(W, x=x, device=device, seed=seed)
    return pick_from(rows, W, x=x, target_bits=target_bits)


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    d_out, d_in = 1024, 256
    x = rng.standard_normal((48, d_in)).astype(np.float32)

    def low_rank(r):
        U = rng.standard_normal((d_out, r)).astype(np.float32); V = rng.standard_normal((r, d_in)).astype(np.float32)
        return ((U @ V) / np.sqrt(r * d_in) + rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)

    full = (rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    heavy = full.copy(); oi = rng.choice(heavy.size, int(0.003 * heavy.size), replace=False)
    heavy.ravel()[oi] = (rng.standard_normal(oi.size) * 12 / np.sqrt(d_in)).astype(np.float32)

    for label, W in [("low-rank (r8)", low_rank(8)), ("full-rank Gaussian", full), ("heavy-tailed outliers", heavy)]:
        rows = measure_candidates(W, x=x, device=dev)             # build + measure ONCE
        print(f"\n=== {label}  {W.shape} ===")
        for tb in (2.5, 3.6, 4.5):
            pick = pick_from(rows, W, x=x, target_bits=tb)
            print(f"  [budget {tb} b/w] -> {pick.name:>14}   ({pick.reason})")
        print("     candidates (name, b/w, MSE, out-err):")
        for name, bits, mse, oerr in pick_from(rows, W, x=x).table:
            print(f"        {name:>16} {bits:>6.2f} {mse:>11.2e} {oerr:>11.2e}")
    print("\n=> One entry point over all levers: it BUILDS a candidate per lever, MEASURES bits + MSE + output\n"
          "   error, and picks the best for the goal (min output error, optionally under a bit budget), reporting\n"
          "   the SVD structure signal that explains the pick. Low-rank tensors -> a low-rank lever wins; full-rank\n"
          "   / outlier tensors -> an element lever (int / PQ / outliers). Build-driven, so it can never mis-label.")


if __name__ == "__main__":
    _demo()
