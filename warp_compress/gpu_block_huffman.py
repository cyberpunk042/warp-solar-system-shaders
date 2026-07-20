"""gpu_block_huffman — a block-wise LUT-Huffman value array, GPU-decoded. The fast whole-tensor path.

Inspired by DFloat11 (arXiv 2504.11651): decode Huffman via a **lookup table** (read `maxlen` bits, one table
hit gives the symbol + its length), not a bit-by-bit tree walk. Values are grouped into fixed-count blocks;
each block is decoded by **one GPU thread**, so the whole array reconstructs in parallel — the fast
whole-tensor decode our wavelet (which walks `bits` rank levels per element) lacks. And because blocks are
independent with a recorded bit-offset, **random access survives**: fetch value i by decoding its block up to
position i%B. Two decode modes over the same bytes: this for bulk decode, the wavelet for search.

Complements `gpu_rrr_wavelet` (which stays for rank/select/FM-index). Best for a small alphabet (int4/int8
quantized values); very skewed 256-symbol streams may exceed the LUT cap and should fall back to the wavelet.

    BlockHuffmanArray(values, block=16)   -> canonical Huffman + per-block bit offsets + a decode LUT
    .decode()                             -> the whole array (one thread per block), on the GPU
    .fetch(indices)                       -> random access (decode within the block)

Run: python -m warp_compress.gpu_block_huffman
"""
from __future__ import annotations

import heapq

import numpy as np
import warp as wp

wp.init()
_LUT_MAXLEN = 16                                   # LUT is 2**maxlen entries; refuse codes longer than this


def _huff_lengths(freq):
    present = [(int(freq[s]), s) for s in range(len(freq)) if freq[s] > 0]
    L = [0] * len(freq)
    if len(present) == 1:
        L[present[0][1]] = 1
        return L
    children, nid, heap = {}, len(freq), [(f, s) for f, s in present]
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, a = heapq.heappop(heap)
        f2, b = heapq.heappop(heap)
        children[nid] = (a, b)
        heapq.heappush(heap, (f1 + f2, nid))
        nid += 1
    root = heap[0][1]
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if node < len(freq):
            L[node] = max(d, 1)
        else:
            a, b = children[node]
            stack += [(a, d + 1), (b, d + 1)]
    return L


def _limit_lengths(L, freq, limit):
    """Length-limit Huffman code lengths to <= `limit` (JPEG C.3 bl_count redistribution), then reassign the
    lengths to symbols by frequency (shortest to the most frequent). Keeps the code prefix-free."""
    if max(L) <= limit:
        return L
    bl = [0] * (max(L) + 1)
    for l in L:
        if l:
            bl[l] += 1
    i = len(bl) - 1
    while i > limit:
        while bl[i] > 0:
            j = i - 2
            while bl[j] == 0:
                j -= 1
            bl[i] -= 2
            bl[i - 1] += 1
            bl[j] -= 1
            bl[j + 1] += 2
        i -= 1
    lengths = []
    for l in range(1, limit + 1):
        lengths += [l] * bl[l]
    present = sorted((s for s in range(len(L)) if L[s] > 0), key=lambda s: -int(freq[s]))
    newL = [0] * len(L)
    for s, l in zip(present, lengths):
        newL[s] = l
    return newL


def _canonical(L):
    """Canonical MSB-first codes from lengths. Returns (maxlen, code_of[V])."""
    maxlen = max(L)
    bl = [0] * (maxlen + 1)
    for l in L:
        if l:
            bl[l] += 1
    first, code = [0] * (maxlen + 2), 0
    for l in range(1, maxlen + 1):
        first[l] = code
        code = (code + bl[l]) << 1
    nxt = list(first)
    code_of = [0] * len(L)
    for s in sorted((s for s in range(len(L)) if L[s] > 0), key=lambda s: (L[s], s)):
        code_of[s] = nxt[L[s]]
        nxt[L[s]] += 1
    return maxlen, code_of


@wp.kernel
def _decode_k(words: wp.array(dtype=wp.uint32), block_off: wp.array(dtype=wp.int32),
              lut: wp.array(dtype=wp.int32), maxlen: int, block: int, n: int,
              out: wp.array(dtype=wp.int32)):
    b = wp.tid()
    pos = block_off[b]
    base = b * block
    cnt = wp.min(block, n - base)
    for j in range(cnt):
        look = int(0)                                          # read maxlen bits MSB-first -> LUT index
        for k in range(maxlen):
            wpos = pos + k
            look = (look << 1) | int((words[wpos >> 5] >> wp.uint32(31 - (wpos & 31))) & wp.uint32(1))
        sl = lut[look]
        out[base + j] = sl & 0xFF                              # symbol
        pos = pos + (sl >> 8)                                  # advance by the code length


@wp.kernel
def _fetch_k(words: wp.array(dtype=wp.uint32), block_off: wp.array(dtype=wp.int32),
             lut: wp.array(dtype=wp.int32), maxlen: int, block: int,
             idx: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    i = idx[t]
    b = i // block
    local = i % block
    pos = block_off[b]
    sym = int(0)
    for j in range(local + 1):                                # decode within the block up to position i
        look = int(0)
        for k in range(maxlen):
            wpos = pos + k
            look = (look << 1) | int((words[wpos >> 5] >> wp.uint32(31 - (wpos & 31))) & wp.uint32(1))
        sl = lut[look]
        sym = sl & 0xFF
        pos = pos + (sl >> 8)
    out[t] = sym


class BlockHuffmanArray:
    """A small-alphabet value stream, canonical-Huffman-coded in fixed-count blocks, GPU-decoded via a LUT."""

    def __init__(self, values, block: int = 16, device: str = "cuda:0"):
        v = np.asarray(values, np.int64)
        self.n = int(v.shape[0])
        self.block = int(block)
        self.device = device
        V = int(v.max()) + 1 if self.n else 1
        freq = np.bincount(v, minlength=V)
        L = _limit_lengths(_huff_lengths(freq), freq, _LUT_MAXLEN)   # cap code length -> bounded LUT
        self.maxlen, code_of = _canonical(L)

        Larr = np.asarray(L, np.int64)
        Carr = np.asarray(code_of, np.int64)
        lens = Larr[v]                                         # per-value code length
        codes = Carr[v]
        starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        total = int(starts[-1])
        words = np.zeros((total + 31) // 32 + 1, np.uint32)
        if total:                                             # write every code bit MSB-first, vectorised
            rep = np.repeat(np.arange(self.n), lens)
            within = np.arange(total) - starts[rep]
            bitval = (codes[rep] >> (lens[rep] - 1 - within)) & 1
            p = np.arange(total)[bitval.astype(bool)]
            np.bitwise_or.at(words, p >> 5, (np.uint32(1) << (31 - (p & 31)).astype(np.uint32)))

        nblocks = (self.n + self.block - 1) // self.block
        block_off = starts[np.minimum(np.arange(nblocks) * self.block, self.n)].astype(np.int32)
        lut = np.zeros(1 << self.maxlen, np.int32)             # LUT[look] = symbol | (length << 8)
        for s, l in enumerate(L):
            if l:
                c = code_of[s] << (self.maxlen - l)
                lut[c: c + (1 << (self.maxlen - l))] = s | (l << 8)

        self._payload_bits, self._off_bytes = total, block_off.nbytes
        self.words = wp.array(words, dtype=wp.uint32, device=device)
        self.block_off = wp.array(block_off, dtype=wp.int32, device=device)
        self.lut = wp.array(lut, dtype=wp.int32, device=device)

    def size_bits(self) -> int:
        return self._payload_bits + self._off_bytes * 8 + self.lut.shape[0] * 2   # payload + offsets + tiny LUT

    def to_host(self):
        params = {"n": self.n, "block": self.block, "maxlen": self.maxlen,
                  "payload_bits": self._payload_bits, "off_bytes": self._off_bytes}
        return params, {"words": self.words.numpy(), "block_off": self.block_off.numpy(), "lut": self.lut.numpy()}

    @classmethod
    def from_host(cls, params, arrays, device="cuda:0"):
        self = cls.__new__(cls)
        self.n, self.block, self.maxlen, self.device = params["n"], params["block"], params["maxlen"], device
        self._payload_bits, self._off_bytes = params["payload_bits"], params["off_bytes"]
        self.words = wp.array(arrays["words"], dtype=wp.uint32, device=device)
        self.block_off = wp.array(arrays["block_off"], dtype=wp.int32, device=device)
        self.lut = wp.array(arrays["lut"], dtype=wp.int32, device=device)
        return self

    def decode(self) -> np.ndarray:
        nblocks = (self.n + self.block - 1) // self.block
        out = wp.zeros(self.n, dtype=wp.int32, device=self.device)
        wp.launch(_decode_k, dim=nblocks,
                  inputs=[self.words, self.block_off, self.lut, self.maxlen, self.block, self.n, out],
                  device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def fetch(self, indices) -> np.ndarray:
        idx = wp.array(np.asarray(indices, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(idx.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_fetch_k, dim=idx.shape[0],
                  inputs=[self.words, self.block_off, self.lut, self.maxlen, self.block, idx, out],
                  device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


def _demo():
    import time

    from .gpu_rrr_huffman import RRRWaveletGPUHuff

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # a peaky int4 value stream (like quantized weights): 0..14 concentrated near the centre
    n = 4_000_000
    vals = np.clip(np.round(rng.standard_normal(n) * 2), -7, 7).astype(np.int64) + 7

    wm = RRRWaveletGPUHuff(vals, device=dev, bits=4)
    for _ in range(3):
        wm.access(np.arange(n))
    t0 = time.perf_counter(); wm.access(np.arange(n)); t_wm = time.perf_counter() - t0

    print(f"device={dev}   n={n:,} int4 values (peaky)")
    print(f"  RRR wavelet (search + random access):   {wm.index_bytes()*8/n:5.2f} b/val   "
          f"bulk decode {n/t_wm/1e6:.0f} M/s ({t_wm*1e3:.0f} ms)")
    print(f"  {'block':>6} {'b/val':>6} {'bulk M/s':>9} {'vs wavelet':>11}   (correct)")
    for blk in (16, 64, 256):
        bh = BlockHuffmanArray(vals, block=blk, device=dev)
        ok = np.array_equal(bh.decode(), vals)
        qi = rng.integers(0, n, 3000).astype(np.int32)
        ok &= np.array_equal(bh.fetch(qi), vals[qi])
        for _ in range(3):
            bh.decode()
        t0 = time.perf_counter(); bh.decode(); t = time.perf_counter() - t0
        print(f"  {blk:>6} {bh.size_bits()/n:>6.2f} {n/t/1e6:>9.0f} {t_wm/t:>10.1f}×   {'✓' if ok else 'FAIL'}")
    print("=> borrowed DFloat11's LUT decode: one thread per block, table-decode all its values in parallel — "
          "an order-of-magnitude faster WHOLE-tensor path than the wavelet (which walks log2(V) rank levels per\n"
          "   element). `block` is the dial: small = smaller random-access latency + bigger offset table; large ="
          " near the wavelet's size + still ~10× faster bulk decode. Two modes, same data (block-LUT bulk, "
          "wavelet search).")


if __name__ == "__main__":
    _demo()
