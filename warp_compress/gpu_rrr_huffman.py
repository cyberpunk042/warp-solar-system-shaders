"""gpu_rrr_huffman — RRR with a HUFFMAN-CODED class stream, decoded in-kernel on the GPU.

The RRR bitvector (`gpu_rrr`) stores each block's class (its popcount, 0..15) as a fixed 4-bit code — a
~0.27 bit/bit floor that keeps a very-skewed plane above its true entropy. The class values are themselves
skewed (a near-empty plane is almost all class 0), so a **canonical Huffman code** over the 16 classes drops
that floor toward H0. The cost is that rank must now *decode* the class stream — done here on the GPU with a
tiny canonical decoder (a bounded bit-by-bit walk, no big table), so rank stays O(1)-ish and VRAM-resident.

Standalone + validated so it never risks the tested 4-bit path; wiring it under the wavelet / weight store is
the follow-up. Run: python -m warp_compress.gpu_rrr_huffman
"""
from __future__ import annotations

import heapq

import numpy as np
import warp as wp

from .gpu_rrr import S, T, _BINOM, _WIDTH, _decode_word, rrr_encode
from .gpu_wavelet import _popcount

wp.init()


def _huff_lengths(freq):
    """Huffman code lengths for the 16 class symbols (0 = absent). Single symbol -> length 1."""
    present = [(int(freq[s]), s) for s in range(16) if freq[s] > 0]
    L = [0] * 16
    if len(present) == 1:
        L[present[0][1]] = 1
        return L
    children, nid, heap = {}, 16, []
    for f, s in present:
        heapq.heappush(heap, (f, s))
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
        if node < 16:
            L[node] = max(d, 1)
        else:
            a, b = children[node]
            stack += [(a, d + 1), (b, d + 1)]
    return L


def _canonical(L):
    """Canonical codes from lengths. Returns (maxlen, first_code[], cnt[], fidx[], syms[], code_of[])."""
    maxlen = max(L)
    cnt = [0] * (maxlen + 1)
    for l in L:
        if l > 0:
            cnt[l] += 1
    first_code = [0] * (maxlen + 2)
    code = 0
    for l in range(1, maxlen + 1):
        first_code[l] = code
        code = (code + cnt[l]) << 1
    syms = sorted((s for s in range(16) if L[s] > 0), key=lambda s: (L[s], s))
    fidx = [0] * (maxlen + 1)
    seen = {}
    for pos, s in enumerate(syms):
        seen.setdefault(L[s], pos)
    for l in range(1, maxlen + 1):
        fidx[l] = seen.get(l, 0)
    next_code = list(first_code)
    code_of = [0] * 16
    for s in syms:
        code_of[s] = next_code[L[s]]
        next_code[L[s]] += 1
    return maxlen, first_code, cnt, fidx, syms, code_of


def _encode_class_stream(classes, L, code_of):
    """Pack the class stream MSB-first as canonical Huffman codes; return (words, per-block cumulative bits)."""
    lens = np.asarray([L[c] for c in classes], np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)])           # bit offset of each block's code
    total = int(starts[-1])
    words = np.zeros((total + 31) // 32 + 1, np.uint32)
    for j in range(classes.shape[0]):                         # build-time; MSB-first within each 32-bit word
        cl = int(classes[j]); l = L[cl]; code = code_of[cl]; p = int(starts[j])
        for b in range(l):
            if (code >> (l - 1 - b)) & 1:
                words[p >> 5] |= np.uint32(1) << np.uint32(31 - (p & 31))
            p += 1
    return words, starts


def rrr_encode_huff(bits) -> dict:
    """RRR components with a Huffman-coded class stream (offsets/superblocks reuse `rrr_encode`)."""
    e = rrr_encode(bits)
    classes = e["classes"]
    freq = np.bincount(classes, minlength=16)
    L = _huff_lengths(freq)
    maxlen, first_code, cnt, fidx, syms, code_of = _canonical(L)
    cwords, cstart = _encode_class_stream(classes, L, code_of)
    sbclass = cstart[e["sidx"]].astype(np.int32)              # class-stream bit position at each superblock
    e.update(cwords=cwords, sbclass=sbclass, class_bits=int(cstart[-1]), maxlen=maxlen,
             first_code=np.asarray(first_code, np.int32), cnt=np.asarray(cnt, np.int32),
             fidx=np.asarray(fidx, np.int32), syms=np.asarray(syms or [0], np.int32))
    return e


@wp.func
def _huff_decode(cs: wp.array(dtype=wp.uint32), first_code: wp.array(dtype=wp.int32),
                 cnt: wp.array(dtype=wp.int32), fidx: wp.array(dtype=wp.int32),
                 syms: wp.array(dtype=wp.int32), maxlen: int, cbit: int) -> int:
    """Canonical-Huffman decode at bit position `cbit`; returns symbol | (code_length << 5)."""
    code = int(0)
    for l in range(1, maxlen + 1):
        pos = cbit + l - 1
        bit = int((cs[pos >> 5] >> wp.uint32(31 - (pos & 31))) & wp.uint32(1))
        code = (code << 1) | bit
        c = cnt[l]
        if c > 0:
            off = code - first_code[l]
            if off >= 0 and off < c:
                return syms[fidx[l] + off] | (l << 5)
    return 0


@wp.func
def _readbits(stream: wp.array(dtype=wp.uint32), bitpos: int, width: int) -> int:
    if width == 0:
        return 0
    wi = bitpos >> 5
    b = bitpos & 31
    val = stream[wi] >> wp.uint32(b)
    if b + width > 32:
        val = val | (stream[wi + 1] << wp.uint32(32 - b))
    mask = (wp.uint32(1) << wp.uint32(width)) - wp.uint32(1)
    return int(val & mask)


@wp.kernel
def _rank1_huff_k(cwords: wp.array(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
                  sbrank: wp.array(dtype=wp.int32), sboff: wp.array(dtype=wp.int32),
                  sbclass: wp.array(dtype=wp.int32), width: wp.array(dtype=wp.int32),
                  binom: wp.array2d(dtype=wp.int32), first_code: wp.array(dtype=wp.int32),
                  cnt: wp.array(dtype=wp.int32), fidx: wp.array(dtype=wp.int32),
                  syms: wp.array(dtype=wp.int32), maxlen: int, pos_in: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    pos = pos_in[t]
    blk = pos // T
    b = pos % T
    sbi = blk // S
    r = sbrank[sbi]
    obit = sboff[sbi]
    cbit = sbclass[sbi]
    j = sbi * S
    while j < blk:                                            # decode + skip each in-superblock block's class
        dec = _huff_decode(cwords, first_code, cnt, fidx, syms, maxlen, cbit)
        cl = dec & 31
        cbit = cbit + (dec >> 5)
        r = r + cl
        obit = obit + width[cl]
        j = j + 1
    if b > 0:
        dec = _huff_decode(cwords, first_code, cnt, fidx, syms, maxlen, cbit)
        cl = dec & 31
        o = _readbits(offsets, obit, width[cl])
        word = _decode_word(binom, cl, o)
        mask = (wp.uint32(1) << wp.uint32(b)) - wp.uint32(1)
        r = r + _popcount(word & mask)
    out[t] = r


class GPURRRHuff:
    """RRR bitvector with a Huffman-coded class stream, resident on a Warp device; batched rank1 on the GPU."""

    def __init__(self, bits, device: str = "cuda:0"):
        self.n = int(np.asarray(bits).shape[0])
        self.device = device
        e = rrr_encode_huff(bits)
        self._class_bits, self._offset_bits = e["class_bits"], e["offset_bits"]
        self._sb_bytes = e["sbrank"].nbytes + e["sboff"].nbytes + e["sbclass"].nbytes
        self._tables_bits = e["syms"].shape[0] * 4 + (e["maxlen"] + 1) * 8      # tiny code tables
        self.maxlen = int(e["maxlen"])
        self.cwords = wp.array(e["cwords"], dtype=wp.uint32, device=device)
        self.offsets = wp.array(e["opk"], dtype=wp.uint32, device=device)
        self.sbrank = wp.array(e["sbrank"], dtype=wp.int32, device=device)
        self.sboff = wp.array(e["sboff"], dtype=wp.int32, device=device)
        self.sbclass = wp.array(e["sbclass"], dtype=wp.int32, device=device)
        self.width = wp.array(_WIDTH, dtype=wp.int32, device=device)
        self.binom = wp.array(_BINOM.astype(np.int32), dtype=wp.int32, device=device)
        self.first_code = wp.array(e["first_code"], dtype=wp.int32, device=device)
        self.cnt = wp.array(e["cnt"], dtype=wp.int32, device=device)
        self.fidx = wp.array(e["fidx"], dtype=wp.int32, device=device)
        self.syms = wp.array(e["syms"], dtype=wp.int32, device=device)

    def size_bits(self) -> int:
        return self._class_bits + self._offset_bits + self._sb_bytes * 8 + self._tables_bits

    def rank1(self, positions) -> np.ndarray:
        pos = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(pos.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_rank1_huff_k, dim=pos.shape[0],
                  inputs=[self.cwords, self.offsets, self.sbrank, self.sboff, self.sbclass, self.width,
                          self.binom, self.first_code, self.cnt, self.fidx, self.syms, self.maxlen, pos, out],
                  device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


# ==================================================================================================
# The Huffman class stream, wired UNDER the RRR wavelet: per-level Huffman tables stacked as 2-D arrays, so
# access/rank (and the FM-index / weight-store built on them) inherit the ~H0 footprint with GPU rank.
# ==================================================================================================

@wp.func
def _huff_decode_lvl(cs: wp.array(dtype=wp.uint32), fc: wp.array2d(dtype=wp.int32),
                     cnt: wp.array2d(dtype=wp.int32), fidx: wp.array2d(dtype=wp.int32),
                     syms: wp.array2d(dtype=wp.int32), maxlens: wp.array(dtype=wp.int32),
                     lvl: int, cbit: int) -> int:
    code = int(0)
    ml = maxlens[lvl]
    for l in range(1, ml + 1):
        pos = cbit + l - 1
        bit = int((cs[pos >> 5] >> wp.uint32(31 - (pos & 31))) & wp.uint32(1))
        code = (code << 1) | bit
        c = cnt[lvl, l]
        if c > 0:
            off = code - fc[lvl, l]
            if off >= 0 and off < c:
                return syms[lvl, fidx[lvl, l] + off] | (l << 5)
    return 0


@wp.func
def _rank1_lvl_h(cwords: wp.array(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
                 sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
                 sbclass: wp.array2d(dtype=wp.int32), cbase: wp.array(dtype=wp.int32),
                 obase: wp.array(dtype=wp.int32), width: wp.array(dtype=wp.int32),
                 binom: wp.array2d(dtype=wp.int32), fc: wp.array2d(dtype=wp.int32),
                 cnt: wp.array2d(dtype=wp.int32), fidx: wp.array2d(dtype=wp.int32),
                 syms: wp.array2d(dtype=wp.int32), maxlens: wp.array(dtype=wp.int32),
                 lvl: int, pos: int) -> int:
    blk = pos // T
    b = pos % T
    sbi = blk // S
    r = sbrank[lvl, sbi]
    obit = obase[lvl] + sboff[lvl, sbi]
    cbit = cbase[lvl] + sbclass[lvl, sbi]
    j = sbi * S
    while j < blk:
        dec = _huff_decode_lvl(cwords, fc, cnt, fidx, syms, maxlens, lvl, cbit)
        cl = dec & 31
        cbit = cbit + (dec >> 5)
        r = r + cl
        obit = obit + width[cl]
        j = j + 1
    if b > 0:
        dec = _huff_decode_lvl(cwords, fc, cnt, fidx, syms, maxlens, lvl, cbit)
        cl = dec & 31
        o = _readbits(offsets, obit, width[cl])
        word = _decode_word(binom, cl, o)
        mask = (wp.uint32(1) << wp.uint32(b)) - wp.uint32(1)
        r = r + _popcount(word & mask)
    return r


@wp.kernel
def _access_kh(cwords: wp.array(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
               sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
               sbclass: wp.array2d(dtype=wp.int32), cbase: wp.array(dtype=wp.int32),
               obase: wp.array(dtype=wp.int32), zeros: wp.array(dtype=wp.int32),
               width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
               fc: wp.array2d(dtype=wp.int32), cnt: wp.array2d(dtype=wp.int32),
               fidx: wp.array2d(dtype=wp.int32), syms: wp.array2d(dtype=wp.int32),
               maxlens: wp.array(dtype=wp.int32), pos_in: wp.array(dtype=wp.int32),
               out: wp.array(dtype=wp.int32), bits: int):
    t = wp.tid()
    i = pos_in[t]
    v = int(0)
    for lvl in range(bits):
        r0 = _rank1_lvl_h(cwords, offsets, sbrank, sboff, sbclass, cbase, obase, width, binom, fc, cnt, fidx,
                          syms, maxlens, lvl, i)
        r1 = _rank1_lvl_h(cwords, offsets, sbrank, sboff, sbclass, cbase, obase, width, binom, fc, cnt, fidx,
                          syms, maxlens, lvl, i + 1)
        if r1 - r0 == 1:
            v = (v << 1) | 1
            i = zeros[lvl] + r0
        else:
            v = v << 1
            i = i - r0
    out[t] = v


class RRRWaveletGPUHuff:
    """RRR wavelet whose every level's class stream is Huffman-coded (decoded in-kernel). access on the GPU."""

    def __init__(self, seq, device: str = "cuda:0", bits: int | None = None):
        seq = np.asarray(seq, np.int64)
        self.n = int(seq.shape[0])
        self.bits = int(bits) if bits is not None else max(1, int(seq.max()).bit_length())
        self.device = device

        cwl, opl, sbr, sbo, sbc, zeros, fcl, cntl, fdl, syl, mls = [], [], [], [], [], [], [], [], [], [], []
        cbase, obase = [0], [0]
        self._bits_stored = 0
        cur = seq.copy()
        for lvl in range(self.bits):
            b = ((cur >> (self.bits - 1 - lvl)) & 1).astype(np.uint8)
            e = rrr_encode_huff(b)
            cwl.append(e["cwords"]); opl.append(e["opk"])
            sbr.append(e["sbrank"]); sbo.append(e["sboff"]); sbc.append(e["sbclass"]); zeros.append(self.n - int(b.sum()))
            fcl.append(e["first_code"]); cntl.append(e["cnt"]); fdl.append(e["fidx"]); syl.append(e["syms"])
            mls.append(int(e["maxlen"]))
            cbase.append(cbase[-1] + e["cwords"].shape[0]); obase.append(obase[-1] + e["opk"].shape[0])
            self._bits_stored += e["class_bits"] + e["offset_bits"]
            order = np.concatenate([np.flatnonzero(b == 0), np.flatnonzero(b == 1)])
            cur = cur[order]

        ML = max(mls)
        pad = lambda a, L: np.pad(np.asarray(a, np.int32)[:L], (0, max(0, L - len(a))))
        self.maxlen = ML
        self._sb_bytes = np.stack(sbr).nbytes + np.stack(sbo).nbytes + np.stack(sbc).nbytes
        self.cwords = wp.array(np.concatenate(cwl), dtype=wp.uint32, device=device)
        self.offsets = wp.array(np.concatenate(opl), dtype=wp.uint32, device=device)
        self.sbrank = wp.array(np.stack(sbr), dtype=wp.int32, device=device)
        self.sboff = wp.array(np.stack(sbo), dtype=wp.int32, device=device)
        self.sbclass = wp.array(np.stack(sbc), dtype=wp.int32, device=device)
        self.cbase = wp.array(np.asarray(cbase[:-1], np.int32) * 32, dtype=wp.int32, device=device)
        self.obase = wp.array(np.asarray(obase[:-1], np.int32) * 32, dtype=wp.int32, device=device)
        self.zeros = wp.array(np.asarray(zeros, np.int32), dtype=wp.int32, device=device)
        self.width = wp.array(_WIDTH, dtype=wp.int32, device=device)
        self.binom = wp.array(_BINOM.astype(np.int32), dtype=wp.int32, device=device)
        self.fc = wp.array(np.stack([pad(a, ML + 1) for a in fcl]), dtype=wp.int32, device=device)
        self.cnt = wp.array(np.stack([pad(a, ML + 1) for a in cntl]), dtype=wp.int32, device=device)
        self.fidx = wp.array(np.stack([pad(a, ML + 1) for a in fdl]), dtype=wp.int32, device=device)
        self.syms = wp.array(np.stack([pad(a, 16) for a in syl]), dtype=wp.int32, device=device)
        self.maxlens = wp.array(np.asarray(mls, np.int32), dtype=wp.int32, device=device)

    def index_bytes(self) -> int:
        return self._bits_stored // 8 + self._sb_bytes

    def access(self, positions) -> np.ndarray:
        pos = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(pos.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_access_kh, dim=pos.shape[0],
                  inputs=[self.cwords, self.offsets, self.sbrank, self.sboff, self.sbclass, self.cbase,
                          self.obase, self.zeros, self.width, self.binom, self.fc, self.cnt, self.fidx,
                          self.syms, self.maxlens, pos, out, self.bits], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


def _demo():
    import math

    from .gpu_rrr import GPURRR

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    n = 2_000_000
    print(f"device={dev}   N={n:,}   (4-bit-class RRR vs Huffman-class RRR; rank must match)")
    print(f"  {'density p':>10} {'H0(p)':>7} {'RRR4 b/bit':>11} {'RRRhuff b/bit':>14} {'gain':>6}  rank✓")
    for p in (0.5, 0.1, 0.03, 0.005):
        bits = (rng.random(n) < p).astype(np.uint8)
        r4 = GPURRR(bits, device=dev)
        rh = GPURRRHuff(bits, device=dev)
        cum = np.concatenate([[0], np.cumsum(bits)])
        q = rng.integers(0, n + 1, 20000).astype(np.int32)
        ok = np.array_equal(rh.rank1(q), cum[q])
        h0 = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
        b4 = r4.size_bits() / n
        bh = rh.size_bits() / n
        print(f"  {p:>10.3f} {h0:>7.3f} {b4:>11.3f} {bh:>14.3f} {b4/bh:>5.2f}×  {'✓' if ok else 'FAIL'}")
    print("=> Huffman-coding the class stream lifts the RRR floor toward H0 on skewed planes, with rank still "
          "GPU-resident (a tiny canonical decoder in the scan). This is the lever that pushes the self-index\n"
          "   below the 4-bit-class floor — the payoff is int4 weights toward their H0 (~1.2 b/w). Next: wire "
          "under the RRR wavelet + weight store.")


if __name__ == "__main__":
    _demo()
