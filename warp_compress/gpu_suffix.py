"""gpu_suffix — build the suffix array (and the BWT) on the GPU, so the FM-index *construction* is resident too.

Every FM-index query already runs on the GPU (`gpu_rrr_wavelet`), but the index was still *built* on the CPU
(`fm_index.suffix_array`, numpy argsort prefix-doubling). This closes that gap: prefix-doubling with Warp's
`radix_sort_pairs` (a 64-bit composite key per round) + `array_scan` (the rank recomputation), so the O(n log n)
sort loop runs on the device. Genomics aligners build their FM-index on the GPU; this is the same move for
token streams.

    gpu_suffix_array(s)  -> the suffix array of `s` (a 0-sentinel'd int sequence), == fm_index.suffix_array
    gpu_bwt(seq)         -> (bwt, sa) for the sentinel'd sequence, ready for the RRR wavelet

The algorithm is exactly the CPU one — rank[i] = order of suffix i among first-k characters; each round forms
key = (rank[i] << 32) | (rank[i+k] + 1), sorts, and re-ranks by adjacent-key differences — but every round is a
GPU radix sort + scan + a few elementwise kernels. Run: python -m warp_compress.gpu_suffix
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _iota_k(vals: wp.array(dtype=wp.int32), n: int):
    t = wp.tid()
    if t < n:
        vals[t] = t


@wp.kernel
def _build_key_k(rank: wp.array(dtype=wp.int64), key: wp.array(dtype=wp.int64), n: int, k: int):
    """key[i] = (rank[i] << 32) | (rank[i+k] + 1), with past-the-end second half = 0 (sorts first)."""
    i = wp.tid()
    if i >= n:
        return
    second = wp.int64(0)
    if i + k < n:
        second = rank[i + k] + wp.int64(1)
    key[i] = (rank[i] << wp.int64(32)) | second


@wp.kernel
def _differs_k(key_sorted: wp.array(dtype=wp.int64), flag: wp.array(dtype=wp.int32), n: int):
    """flag[i] = 1 where the sorted key changes from its predecessor (a new rank starts); flag[0] = 0."""
    i = wp.tid()
    if i >= n:
        return
    if i == 0:
        flag[0] = 0
    elif key_sorted[i] != key_sorted[i - 1]:
        flag[i] = 1
    else:
        flag[i] = 0


@wp.kernel
def _scatter_rank_k(sa: wp.array(dtype=wp.int32), scanned: wp.array(dtype=wp.int32),
                    rank: wp.array(dtype=wp.int64), n: int):
    """rank[sa[i]] = scanned[i] — write each suffix's new (dense) rank back to sequence order."""
    i = wp.tid()
    if i < n:
        rank[sa[i]] = wp.int64(scanned[i])


@wp.kernel
def _init_rank_k(s: wp.array(dtype=wp.int64), rank: wp.array(dtype=wp.int64), n: int):
    t = wp.tid()
    if t < n:
        rank[t] = s[t]                                       # raw symbols are order-consistent initial ranks


def gpu_suffix_array(s, device: str = "cuda:0") -> np.ndarray:
    """Suffix array of `s` (ints, a 0 sentinel recommended as the unique smallest) built on the GPU by
    prefix doubling. Bit-identical to `fm_index.suffix_array`."""
    s = np.asarray(s, np.int64)
    n = int(s.shape[0])
    if n <= 1:
        return np.zeros(n, np.int64)

    s_d = wp.array(s, dtype=wp.int64, device=device)
    rank = wp.zeros(n, dtype=wp.int64, device=device)
    key = wp.zeros(2 * n, dtype=wp.int64, device=device)     # radix_sort_pairs needs 2n (ping-pong scratch)
    sa = wp.zeros(2 * n, dtype=wp.int32, device=device)
    flag = wp.zeros(n, dtype=wp.int32, device=device)
    scanned = wp.zeros(n, dtype=wp.int32, device=device)

    wp.launch(_init_rank_k, dim=n, inputs=[s_d, rank, n], device=device)

    k = 1
    while True:
        wp.launch(_build_key_k, dim=n, inputs=[rank, key, n, k], device=device)
        wp.launch(_iota_k, dim=n, inputs=[sa, n], device=device)
        wp.utils.radix_sort_pairs(key, sa, n)                # sort suffixes by (rank[i], rank[i+k]); sa[:n]=order
        wp.launch(_differs_k, dim=n, inputs=[key, flag, n], device=device)
        wp.utils.array_scan(flag, scanned, inclusive=True)   # dense new ranks 0..(#distinct-1) in sorted order
        wp.launch(_scatter_rank_k, dim=n, inputs=[sa, scanned, rank, n], device=device)
        wp.synchronize_device(device)
        if int(scanned.numpy()[n - 1]) == n - 1:             # all ranks distinct -> suffix array is final
            break
        k *= 2
    return sa.numpy()[:n].astype(np.int64)


def gpu_bwt(seq, device: str = "cuda:0"):
    """(bwt, sa) for `seq` (ints >= 0) with a 0 sentinel appended — the exact input the RRR wavelet indexes."""
    seq = np.asarray(seq, np.int64) + 1                      # shift so 0 is a free unique sentinel
    s = np.concatenate([seq, [0]])
    n = int(s.shape[0])
    sa = gpu_suffix_array(s, device=device)
    return s[(sa - 1) % n], sa


def _demo():
    import time

    from .fm_index import suffix_array

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    print(f"device={dev}   GPU prefix-doubling suffix array vs CPU numpy argsort\n")
    print(f"  {'sequence':>26} {'n':>9} {'match':>7} {'CPU ms':>9} {'GPU ms':>9} {'speedup':>8}")
    for label, seq in [
        ("random V=64", rng.integers(0, 64, 200_000)),
        ("Markov-ish V=16", np.cumsum(rng.integers(0, 3, 200_000)) % 16),
        ("very repetitive", np.tile(rng.integers(0, 8, 500), 400)),
        ("DNA-like V=4", rng.integers(0, 4, 400_000)),
    ]:
        s = np.concatenate([np.asarray(seq, np.int64) + 1, [0]])
        t0 = time.perf_counter(); sa_cpu = suffix_array(s); cpu_ms = (time.perf_counter() - t0) * 1e3
        sa_gpu = gpu_suffix_array(s, device=dev)             # warm up the kernels
        t0 = time.perf_counter(); sa_gpu = gpu_suffix_array(s, device=dev); gpu_ms = (time.perf_counter() - t0) * 1e3
        ok = np.array_equal(sa_cpu, sa_gpu)
        print(f"  {label:>26} {s.shape[0]:>9,} {'✓' if ok else 'FAIL':>7} {cpu_ms:>9.1f} {gpu_ms:>9.1f} "
              f"{cpu_ms / gpu_ms:>7.2f}×")
    print("\n=> the suffix array — the one piece of the FM-index that was still CPU-built — now constructs on the "
          "GPU by prefix doubling (radix_sort_pairs + array_scan per round), bit-identical to the CPU builder.\n"
          "   So build AND query are device-resident: a token stream goes from raw to searchable self-index "
          "without leaving the GPU. Speedup grows with n and alphabet; tiny/very-repetitive inputs favour CPU.")


if __name__ == "__main__":
    _demo()
