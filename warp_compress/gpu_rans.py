"""gpu_rans — a block-wise rANS value array, GPU-decoded. Near-entropy ratio, same fast/random-access shape.

Lever #2 from the web-verified v2 list. Huffman (`gpu_block_huffman`) costs up to ~1 bit/symbol over the true
entropy; **rANS** (range Asymmetric Numeral Systems, Duda 2013; the coder in zstd-FSE and nvCOMP gANS)
approaches H0. rANS decode is sequential (LIFO), so — exactly as for the Huffman coder — we cut the stream
into **independent fixed-count blocks**: whole-tensor decode is still one GPU thread per block (parallel), and
random access still works by decoding within a block. Same shape as `BlockHuffmanArray`, tighter ratio.

    BlockRANSArray(values, block=64)   -> per-block rANS streams + a shared normalized frequency table
    .decode()                          -> the whole array (one thread per block), on the GPU
    .fetch(indices)                    -> random access (decode within the block)

Best for a small alphabet (int4/int8 quantized values). Run: python -m warp_compress.gpu_rans
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

_PROB_BITS = 12                       # frequencies normalise to M = 2**_PROB_BITS
_M = 1 << _PROB_BITS
_RANS_L = 1 << 23                     # 32-bit state renorm bound, 8-bit renorm (ryg_rans style)


def _normalize(hist):
    """Normalise a histogram to sum exactly M, every present symbol getting freq >= 1."""
    V = len(hist)
    freq = np.zeros(V, np.int64)
    present = hist > 0
    total = int(hist.sum())
    freq[present] = np.maximum(1, np.round(hist[present] * _M / total).astype(np.int64))
    diff = _M - int(freq.sum())
    order = np.argsort(-freq)                         # fix the rounding drift on the largest freqs
    i = 0
    while diff != 0:
        s = order[i % V]
        if freq[s] + np.sign(diff) >= 1 or diff > 0:
            freq[s] += np.sign(diff)
            diff -= np.sign(diff)
        i += 1
    cum = np.concatenate([[0], np.cumsum(freq)]).astype(np.int64)
    slot2sym = np.zeros(_M, np.int32)
    for s in range(V):
        if freq[s]:
            slot2sym[cum[s]: cum[s] + freq[s]] = s
    return freq, cum[:-1], slot2sym


def _encode_block(syms, freq, cum):
    """rANS-encode one block (symbols encoded in reverse -> decode emits them forward). Returns (bytes, state)."""
    state = _RANS_L
    out = bytearray()
    for s in syms[::-1]:
        f = int(freq[s])
        x_max = ((_RANS_L >> _PROB_BITS) << 8) * f     # renorm: emit low bytes while state too big
        while state >= x_max:
            out.append(state & 0xFF)
            state >>= 8
        state = ((state // f) << _PROB_BITS) + (state % f) + int(cum[s])
    return bytes(out[::-1]), int(state)                # decoder reads bytes forward, starts from final state


@wp.kernel
def _decode_k(data: wp.array(dtype=wp.uint8), byte_off: wp.array(dtype=wp.int32),
              state0: wp.array(dtype=wp.uint32), slot2sym: wp.array(dtype=wp.int32),
              freq: wp.array(dtype=wp.int32), cum: wp.array(dtype=wp.int32),
              block: int, n: int, out: wp.array(dtype=wp.int32)):
    b = wp.tid()
    state = state0[b]
    pos = byte_off[b]
    base = b * block
    cnt = wp.min(block, n - base)
    for j in range(cnt):
        slot = int(state & wp.uint32(_M - 1))
        s = slot2sym[slot]
        state = wp.uint32(freq[s]) * (state >> wp.uint32(_PROB_BITS)) + wp.uint32(slot - cum[s])
        while state < wp.uint32(_RANS_L):
            state = (state << wp.uint32(8)) | wp.uint32(data[pos])
            pos = pos + 1
        out[base + j] = s


@wp.kernel
def _fetch_k(data: wp.array(dtype=wp.uint8), byte_off: wp.array(dtype=wp.int32),
             state0: wp.array(dtype=wp.uint32), slot2sym: wp.array(dtype=wp.int32),
             freq: wp.array(dtype=wp.int32), cum: wp.array(dtype=wp.int32),
             block: int, idx: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    i = idx[t]
    b = i // block
    state = state0[b]
    pos = byte_off[b]
    sym = int(0)
    for j in range((i % block) + 1):
        slot = int(state & wp.uint32(_M - 1))
        sym = slot2sym[slot]
        state = wp.uint32(freq[sym]) * (state >> wp.uint32(_PROB_BITS)) + wp.uint32(slot - cum[sym])
        while state < wp.uint32(_RANS_L):
            state = (state << wp.uint32(8)) | wp.uint32(data[pos])
            pos = pos + 1
    out[t] = sym


class BlockRANSArray:
    """A small-alphabet value stream, rANS-coded in fixed-count blocks, GPU-decoded (near-entropy ratio)."""

    def __init__(self, values, block: int = 64, device: str = "cuda:0"):
        v = np.asarray(values, np.int64)
        self.n = int(v.shape[0])
        self.block = int(block)
        self.device = device
        V = int(v.max()) + 1 if self.n else 1
        self.freq, self.cum, slot2sym = _normalize(np.bincount(v, minlength=V))

        nblocks = (self.n + self.block - 1) // self.block
        blobs, states, offs = [], np.zeros(nblocks, np.uint32), np.zeros(nblocks, np.int64)
        cursor = 0
        for b in range(nblocks):
            seg = v[b * self.block: (b + 1) * self.block]
            data, st = _encode_block(seg, self.freq, self.cum)
            offs[b] = cursor
            states[b] = st
            blobs.append(data)
            cursor += len(data)
        stream = b"".join(blobs) + b"\x00\x00\x00\x00"          # pad for the final renorm reads
        self._payload_bytes = cursor

        self.data = wp.array(np.frombuffer(stream, np.uint8).copy(), dtype=wp.uint8, device=device)
        self.byte_off = wp.array(offs.astype(np.int32), dtype=wp.int32, device=device)
        self.state0 = wp.array(states, dtype=wp.uint32, device=device)
        self.slot2sym = wp.array(slot2sym, dtype=wp.int32, device=device)
        self.freq_a = wp.array(self.freq.astype(np.int32), dtype=wp.int32, device=device)
        self.cum_a = wp.array(self.cum.astype(np.int32), dtype=wp.int32, device=device)
        self._off_bytes = offs.nbytes + states.nbytes

    def size_bits(self) -> int:
        return self._payload_bytes * 8 + self._off_bytes * 8 + self.slot2sym.shape[0] * 2   # payload + block dir + table

    def to_host(self):
        params = {"n": self.n, "block": self.block, "payload_bytes": self._payload_bytes,
                  "off_bytes": self._off_bytes}
        arrays = {k: getattr(self, a).numpy() for k, a in
                  {"data": "data", "byte_off": "byte_off", "state0": "state0",
                   "slot2sym": "slot2sym", "freq_a": "freq_a", "cum_a": "cum_a"}.items()}
        return params, arrays

    @classmethod
    def from_host(cls, params, arrays, device="cuda:0"):
        self = cls.__new__(cls)
        self.n, self.block, self.device = params["n"], params["block"], device
        self._payload_bytes, self._off_bytes = params["payload_bytes"], params["off_bytes"]
        self.data = wp.array(arrays["data"], dtype=wp.uint8, device=device)
        self.byte_off = wp.array(arrays["byte_off"], dtype=wp.int32, device=device)
        self.state0 = wp.array(arrays["state0"], dtype=wp.uint32, device=device)
        self.slot2sym = wp.array(arrays["slot2sym"], dtype=wp.int32, device=device)
        self.freq_a = wp.array(arrays["freq_a"], dtype=wp.int32, device=device)
        self.cum_a = wp.array(arrays["cum_a"], dtype=wp.int32, device=device)
        return self

    def _args(self):
        return [self.data, self.byte_off, self.state0, self.slot2sym, self.freq_a, self.cum_a]

    def decode(self) -> np.ndarray:
        nblocks = (self.n + self.block - 1) // self.block
        out = wp.zeros(self.n, dtype=wp.int32, device=self.device)
        wp.launch(_decode_k, dim=nblocks, inputs=[*self._args(), self.block, self.n, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def fetch(self, indices) -> np.ndarray:
        idx = wp.array(np.asarray(indices, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(idx.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_fetch_k, dim=idx.shape[0], inputs=[*self._args(), self.block, idx, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


def _demo():
    import math
    import time

    from .gpu_block_huffman import BlockHuffmanArray

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    n = 1_000_000
    # two regimes: multi-bit (peaky int4, H0~3) where Huffman is already near-optimal, and low-entropy
    # (very skewed, H0<1) where Huffman's up-to-1-bit-per-symbol overhead bites and rANS should win.
    peaky = np.clip(np.round(rng.standard_normal(n) * 2), -7, 7).astype(np.int64) + 7
    skewed = np.where(rng.random(n) < 0.05, rng.integers(0, 15, n), 7).astype(np.int64)   # mostly one symbol

    print(f"device={dev}   n={n:,}   rANS vs Huffman b/val, across entropy × block (both decode in parallel)")
    print(f"  {'stream':16} {'H0':>5} {'block':>6} {'Huffman':>8} {'rANS':>7} {'winner':>8}")
    for name, vals in [("peaky int4", peaky), ("very skewed", skewed)]:
        _, c = np.unique(vals, return_counts=True); p = c / c.sum(); H0 = float(-(p * np.log2(p)).sum())
        for blk in (64, 1024):
            ra = BlockRANSArray(vals, block=blk, device=dev)
            bh = BlockHuffmanArray(vals, block=blk, device=dev)
            assert np.array_equal(ra.decode(), vals) and np.array_equal(bh.decode(), vals)
            qi = rng.integers(0, n, 2000).astype(np.int32)
            assert np.array_equal(ra.fetch(qi), vals[qi])
            hb, rb = bh.size_bits() / n, ra.size_bits() / n
            print(f"  {name:16} {H0:>5.2f} {blk:>6} {hb:>8.3f} {rb:>7.3f} {('rANS' if rb < hb else 'Huffman'):>8}")
    print("\n=> honest crossover: rANS approaches H0 (no per-symbol overhead), but carries a fixed 32-bit state "
          "PER BLOCK — so at small blocks it loses to Huffman, and for multi-bit streams Huffman is already\n"
          "   near-optimal. rANS WINS on LOW-entropy streams with LARGE blocks (whole-decode / archival). So it's "
          "the coder for the skewed, bulk-decode regime; Huffman stays for small-block random access. Both "
          "decode parallel-per-block on the GPU; pick per data + access pattern.")


if __name__ == "__main__":
    _demo()
