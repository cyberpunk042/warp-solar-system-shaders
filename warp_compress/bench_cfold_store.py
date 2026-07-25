"""bench_cfold_store — the random-access payoff of a .cfold multi-atom store, measured not asserted.

``format.read_atom`` claims you can pull one folded layer from an N-layer store *without decoding the
others*. This measures that claim as N grows, on two axes:

  1. **payload bytes materialised** (deterministic, the honest structural metric) — ``read_atom`` slices
     exactly one atom's bytes; ``unpack_atoms`` materialises all N via ``np.frombuffer``. The win is the
     ratio, and it grows linearly with N by construction.
  2. **wall-time** (noisy, median of repeats) — the same story in seconds, shown so the constant factors
     are visible, not hidden.

Honest caveat reported inline: BOTH paths parse the full JSON header (it lists every section + the
manifest), so header-parse is O(N) for both — the random-access win is in the *payload*, not the header.
On small stores the header-parse constant can dominate and the wall-time win is modest; the payload win
is what scales.

Pure numpy + zlib — no torch, no GPU. A real-model store (a Qwen2.5 layer bank) is the follow-on that
needs the model venv.

Run: python -m warp_compress.bench_cfold_store
"""
from __future__ import annotations

import time

import numpy as np

from . import format as fmt
from . import grouped_delta as gd


def _layer_atom(rng, h=64, w=64, n=8, noise=2):
    """A correlated group of `n` tensors → one folded grouped_delta atom (self-describing GDLT1 blob)."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    group = np.stack([base + rng.integers(-noise, noise + 1, size=(h, w), dtype=np.int64) for _ in range(n)])
    return gd.to_bytes(gd.compress(group, mode="centroid"))


def _median_time(fn, repeats=15):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def _build_store(rng, n_layers):
    atoms = {f"layer.{i}": _layer_atom(rng) for i in range(n_layers)}
    return atoms, fmt.pack_atoms("weight_store", atoms)


def run():
    rng = np.random.default_rng(0)
    print("cfold multi-atom store — random access (read_atom) vs full decode (unpack_atoms)")
    print(f"{'layers':>7} {'store KB':>9} {'atom B':>7} | "
          f"{'read_atom B':>11} {'unpack B':>9} {'bytes×':>7} | "
          f"{'read µs':>8} {'unpack µs':>10} {'time×':>6}")
    for n_layers in (8, 32, 128, 512):
        atoms, store = _build_store(rng, n_layers)
        atom_b = len(next(iter(atoms.values())))
        target = f"layer.{n_layers // 2}"                       # a middle layer — average offset walk

        # correctness gate: the measured fast path must return the real bytes
        assert fmt.read_atom(store, target) == atoms[target], "read_atom returned wrong bytes"

        # payload bytes materialised (deterministic): one atom vs all atoms
        read_bytes = len(atoms[target])
        unpack_bytes = sum(len(a) for a in atoms.values())

        t_read = _median_time(lambda: fmt.read_atom(store, target))
        t_unpack = _median_time(lambda: fmt.unpack_atoms(store))

        print(f"{n_layers:>7} {len(store)/1e3:>9.1f} {atom_b:>7} | "
              f"{read_bytes:>11} {unpack_bytes:>9} {unpack_bytes/read_bytes:>6.0f}× | "
              f"{t_read*1e6:>8.1f} {t_unpack*1e6:>10.1f} {t_unpack/max(t_read,1e-9):>5.1f}×")

    print("\nHonest read:")
    print("  • payload bytes materialised: read_atom is flat (1 atom); unpack_atoms grows linearly (N atoms)")
    print("    → the bytes-ratio == N by construction, the structural random-access win.")
    print("  • wall-time: both parse the full header (O(N) sections + manifest), so the time-ratio is")
    print("    smaller than the bytes-ratio and only opens up once payload work dominates header-parse.")
    print("  • takeaway: read_atom is the right call for pulling a few layers from a large store; for")
    print("    touching most layers, unpack_atoms once amortises the shared header parse.")


if __name__ == "__main__":
    run()
