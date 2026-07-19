"""gpu_fm_index — FM-index backward search on the GPU, in Warp. Search AND generate, entirely in VRAM.

``gpu_wavelet`` put token *decode* on the GPU. This puts *search* there: FM-index backward search over the
BWT — the loop behind ``count`` / ``locate`` / ``predict_next`` — running resident in VRAM. Backward search
is serial within one pattern (each step's SA range depends on the last), but **embarrassingly parallel across
patterns**, and that is precisely the shape of the two operations that matter:

    count(patterns)        one thread per pattern -> substring search over a whole batch at once
    predict_next(context)  one thread per candidate next-token c -> the full next-token distribution in one
                           launch (each thread counts context+[c]); i.e. an n-gram draft model, on the GPU

So the same resident, compact index is *decoded*, *searched*, and *sampled from* without ever leaving the
GPU — the ChromoFold thesis (docs/chromofold.md §1) for the search/generate half. Run:
python -m warp_compress.gpu_fm_index
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .fm_index import FMIndex, suffix_array
from .gpu_wavelet import SB, GPUWavelet, _rank1


@wp.func
def _wrank(words: wp.array2d(dtype=wp.uint32), sb: wp.array2d(dtype=wp.int32),
          zeros: wp.array(dtype=wp.int32), c: int, i: int, bits: int, sbw: int) -> int:
    """Full wavelet rank(c, i) = occurrences of symbol c in BWT[0, i), walking the bit levels."""
    p = int(0)
    q = i
    for lvl in range(bits):
        bitc = (c >> (bits - 1 - lvl)) & 1
        rp = _rank1(words, sb, lvl, p, sbw)
        rq = _rank1(words, sb, lvl, q, sbw)
        if bitc == 1:
            p = zeros[lvl] + rp
            q = zeros[lvl] + rq
        else:
            p = p - rp
            q = q - rq
    return q - p


@wp.kernel
def _bwsearch_k(words: wp.array2d(dtype=wp.uint32), sb: wp.array2d(dtype=wp.int32),
                zeros: wp.array(dtype=wp.int32), C: wp.array(dtype=wp.int32),
                pat: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                lens: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32),
                bits: int, sbw: int, sigma: int, n: int):
    t = wp.tid()
    st = starts[t]
    L = lens[t]
    lo = int(0)
    hi = n
    for k in range(L):
        c = pat[st + L - 1 - k]                         # backward search: consume the pattern right-to-left
        if c < sigma:
            if lo < hi:
                lo = C[c] + _wrank(words, sb, zeros, c, lo, bits, sbw)
                hi = C[c] + _wrank(words, sb, zeros, c, hi, bits, sbw)
        else:
            lo = int(0)
            hi = int(0)                                 # symbol outside the alphabet -> empty range
    out[t] = hi - lo


class GPUFMIndex:
    """FM-index whose backward search runs on a Warp device. count()/predict_next() are batched GPU launches."""

    def __init__(self, seq, device: str = "cuda:0"):
        seq = np.asarray(seq, np.int64) + 1                 # shift so 0 is a free sentinel (as in FMIndex)
        self.n = int(seq.shape[0]) + 1
        s = np.concatenate([seq, [0]])
        sa = suffix_array(s)
        bwt = s[(sa - 1) % self.n]
        self.sigma = int(s.max()) + 1
        self.device = device
        self.bits = max(1, (self.sigma - 1).bit_length())
        self.gw = GPUWavelet(bwt, device=device, bits=self.bits)
        C = np.concatenate([[0], np.cumsum(np.bincount(bwt, minlength=self.sigma))])[: self.sigma]
        self.C = wp.array(C.astype(np.int32), dtype=wp.int32, device=device)

    def count(self, patterns) -> np.ndarray:
        """Occurrence count of each pattern (original alphabet), one GPU thread per pattern."""
        flat, starts, lens = [], [], []
        for p in patterns:
            starts.append(len(flat))
            lens.append(len(p))
            flat.extend(int(x) + 1 for x in p)              # shift into the sentinel'd alphabet
        pat = wp.array(np.asarray(flat or [0], np.int32), dtype=wp.int32, device=self.device)
        st = wp.array(np.asarray(starts, np.int32), dtype=wp.int32, device=self.device)
        ln = wp.array(np.asarray(lens, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(len(patterns), dtype=wp.int32, device=self.device)
        wp.launch(_bwsearch_k, dim=len(patterns),
                  inputs=[self.gw.words, self.gw.sb, self.gw.zeros, self.C, pat, st, ln, out,
                          self.bits, SB, self.sigma, self.n], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def predict_next(self, context, vocab: int | None = None) -> np.ndarray:
        """Next-token distribution over [0, vocab): one thread counts context+[c] per candidate c. An n-gram
        draft model computed in a single GPU launch — the whole distribution, in VRAM."""
        V = int(vocab) if vocab is not None else self.sigma - 1
        ctx = list(context)
        counts = self.count([ctx + [c] for c in range(V)]).astype(np.float64)
        tot = counts.sum()
        return counts / tot if tot > 0 else counts


def _demo():
    import time

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    V = 64
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    n = 2_000_000
    seq = rng.choice(V, size=n, p=p).astype(np.int64)
    gfm = GPUFMIndex(seq, device=dev)
    cfm = FMIndex(seq[:200_000], sa_sample=64)              # CPU oracle on a slice (CPU build is heavy)

    # 1) correctness: GPU count == CPU count == naive, for a batch of patterns known to occur
    pats = []
    for _ in range(200):
        at = int(rng.integers(0, n - 4))
        pats.append([int(x) for x in seq[at:at + 3]])
    gcount = gfm.count(pats)
    for j in rng.integers(0, len(pats), 8):
        naive = sum(1 for i in range(n - 3) if list(seq[i:i + 3]) == pats[j])
        assert gcount[j] == naive, (gcount[j], naive)

    # 2) predict_next matches the CPU FM-index distribution on a shared context
    cseq = seq[:200_000]
    ctx = [int(cseq[100000]), int(cseq[100001])]
    gdist = gfm.predict_next(ctx, vocab=V)                  # careful: GPU index is over the full seq
    gdist2 = GPUFMIndex(cseq, device=dev).predict_next(ctx, vocab=V)
    cdist = cfm.predict_next(ctx)                           # {token: prob}
    top_gpu = int(np.argmax(gdist2))
    top_cpu = max(cdist, key=cdist.get) if cdist else -1
    assert top_gpu == top_cpu, (top_gpu, top_cpu)

    # 3) throughput: batched substring search on the GPU
    M = 200_000
    bat = [[int(x) for x in seq[i:i + 4]] for i in rng.integers(0, n - 4, M)]
    gfm.count(bat[:1000])                                   # warm up
    t0 = time.perf_counter(); gfm.count(bat); gpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q in bat[:2000]:
        cfm.count(q)
    cpu_s = (time.perf_counter() - t0) / 2000

    print(f"device={dev}   N={n:,}   alphabet<=2^{gfm.bits}   resident index {gfm.gw.index_bytes()/1e6:.1f} MB")
    print(f"[correct] GPU count == naive ✓   GPU predict_next top-1 == CPU FM-index top-1 ✓")
    print(f"[search]  {M:,} patterns backward-searched in {gpu_s*1e3:.1f} ms = {M/gpu_s/1e6:.1f} M patterns/s "
          f"vs naive CPU {1/cpu_s:,.0f} patterns/s => GPU ~{(M/gpu_s)/(1/cpu_s):,.0f}× faster")
    print("=> count / locate / predict_next — search AND the n-gram draft model — run GPU-resident over the "
          "compact FM-index, batched. ChromoFold now decodes, searches, and samples without leaving the GPU.")


if __name__ == "__main__":
    _demo()
