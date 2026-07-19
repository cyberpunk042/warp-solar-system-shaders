"""The genome compression engine — ONE THREAD, 8 stages, each folded tighter than the last.

A compression mechanism for a virtual visual object (the graphics card). The card is read by a scan into
ONE ordered stream of tokens; the stream pairs into ONE base-pair ladder; the ladder winds into ONE double
helix; the helix wraps into ONE string of nucleosomes; that coils into ONE 30 nm fibre; the ends cap
(telomere); the one fibre folds into ONE chromosome.

Two rules make it a compression engine and not a slideshow:

  1. **One thread.** The matter is a single ordered strand. Element ``i`` is element ``i`` at every stage —
     the order never changes, nothing is scattered or re-sorted between stages.
  2. **Each stage starts from the previous stage's exact end frame.** A stage only *folds* the positions the
     previous stage produced; it never re-derives them. So end[stage] IS start[stage+1] — continuity is
     structural, impossible to break.

``build()`` computes the ordered strand and every stage's end-frame once. ``frame(progress)`` returns the
strand's positions + colours at any point in the whole take by folding one stage's end frame toward the
next. One camera renders ``frame`` — the thread simply gets tighter (that shrink IS the compression).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .tokenize import tokenize_card

# DNA base palette A/T/G/C (a pair is its two complementary bases)
_BASES = np.array([
    [0.95, 0.35, 0.38],   # A
    [0.98, 0.80, 0.30],   # T
    [0.35, 0.62, 0.98],   # G
    [0.40, 0.90, 0.65],   # C
], dtype=np.float32)


def _hsv(h, s, v):
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(int)
    f = h6 - i
    p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], 1).astype(np.float32)


def _scan_order(pos: np.ndarray, nrows: int = 96) -> np.ndarray:
    """The tokenization SCAN: read the card as one continuous boustrophedon raster so the read order is a
    single local path (the thread). Quantise z into rows; within a row sweep x, alternating direction each
    row, so consecutive tokens in the stream are spatial neighbours — one snake through the card."""
    z, x = pos[:, 2], pos[:, 0]
    zi = np.clip(((z - z.min()) / (np.ptp(z) + 1e-9) * nrows).astype(int), 0, nrows - 1)
    key_x = np.where(zi % 2 == 0, x, -x)             # even rows +x, odd rows -x -> a snake
    return np.lexsort((key_x, zi))                    # sort by row, then serpentine-x


@dataclasses.dataclass
class Thread:
    """The one thread. All position arrays are (N,3) in the SAME token order (element i is always the same
    matter). ``read`` is the scan order that serialises the card into the strand; ``pair`` maps each token
    to its base pair. Colours are (N,3)."""
    card: np.ndarray            # stage 0/1 — tokens at their card voxels (scan order preserved by `read`)
    ladder: np.ndarray          # stage 2 — tokens laid onto the one base-pair ladder
    helix: np.ndarray           # stage 3 — the ladder wound into one double helix (twist along the thread)
    nucleo: np.ndarray          # stage 4 — the helix wrapped into beads on one string (nucleosomes)
    fibre: np.ndarray           # stage 5 — the bead string coiled into one 30 nm fibre (solenoid)
    chromo: np.ndarray          # stage 6/7 — the fibre folded into one compact chromatid (the chromosome)
    col_card: np.ndarray        # muted card material
    col_token: np.ndarray       # tokenised: coloured by merge-codec type
    col_base: np.ndarray        # base-pair colour (A/T/G/C)
    read: np.ndarray            # (N,) scan/read order — position of each token along the thread
    xspan: tuple                # (xmin, xmax) of the card, for the scan bar

    @property
    def n(self):
        return int(self.card.shape[0])


def build(sub: int = 1, block: int = 5) -> Thread:
    tc = tokenize_card(sub=sub, block=block)
    pos = tc.positions.astype(np.float32)
    ids = tc.ids.astype(np.int64)
    n = pos.shape[0]

    order = _scan_order(pos)                          # the scan reads the card in this order = the thread
    if n % 2:                                         # keep it conserved: pairs need an even count
        order = order[:-1]
        n -= 1

    # --- stage 0/1 frame: the card's own voxels (positions unchanged; tokenisation is colour only) ---
    card = pos.copy()

    # read-position of each token along the thread (0..n-1) — used by the scan colour sweep and pairing
    read = np.empty(n, np.int64)
    read[order] = np.arange(n)

    # --- stage 2 frame: ONE base-pair ladder. Consecutive tokens along the thread pair up; the pairs are
    # laid on a single boustrophedon ladder — one continuous strand, folded compactly (the compression). ---
    p = n // 2
    a_tok = order[0::2]                               # first token of each pair (backbone A)
    b_tok = order[1::2]                               # second token (backbone B)

    # ONE continuous strand, rolled into a flat spiral (a clock-spring) — you can follow the single thread
    # from the outside in. Much smaller than the spread-out card: that shrink is the compression.
    i = np.arange(p)
    t = i / max(p - 1, 1)                              # 0..1 along the one thread
    _LT = 24.0                                         # spiral turns
    lth = t * _LT * 2.0 * np.pi
    lr = 0.18 + 1.30 * t
    rung = np.stack([lr * np.cos(lth), np.full(p, 0.55, np.float32), lr * np.sin(lth)], 1).astype(np.float32)
    up = np.array([0.0, 0.012, 0.0], np.float32)       # the two backbones sit just above/below the strand

    ladder = np.empty_like(card)
    ladder[a_tok] = rung - up                          # each token flows to its backbone site on the strand
    ladder[b_tok] = rung + up

    # --- stage 3 frame: wind the ONE ladder into ONE double helix. The thread's path (rung centres) stays
    # put; along it the two backbones twist around each other — real B-DNA (~10.5 bp/turn). Local frame
    # (tangent/normal/binormal) so the twist follows the snaking thread. Chains from the ladder exactly. ---
    tan = np.zeros_like(rung)
    tan[1:-1] = rung[2:] - rung[:-2]
    tan[0] = rung[1] - rung[0]
    tan[-1] = rung[-1] - rung[-2]
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    binormal = np.cross(tan, np.array([0.0, 1.0, 0.0], np.float32))
    binormal /= np.maximum(np.linalg.norm(binormal, axis=1, keepdims=True), 1e-6)
    normal = np.cross(binormal, tan)
    theta = (np.arange(p) * (2.0 * np.pi / 10.5))[:, None]
    r_h = 0.028
    off = r_h * (np.cos(theta) * normal + np.sin(theta) * binormal)
    helix = np.empty_like(card)
    helix[a_tok] = rung + off
    helix[b_tok] = rung - off

    # --- stage 4 frame: wrap the helix into BEADS ON ONE STRING (nucleosomes). Group G pairs per bead; the
    # bead centres run as one compact serpentine (fewer than the ladder — tighter), and inside each bead the
    # strand wraps ~1.75 turns around the core. Same ordered pairs, folded tighter. ---
    tw = 0.010                                        # the dsDNA is thin at this compaction
    small = np.array([0.0, tw, 0.0], np.float32)
    G = 180
    nb = (p + G - 1) // G
    bead = np.arange(p) // G
    lf = (np.arange(p) % G).astype(np.float32)
    bt = (np.arange(nb) + 0.5) / nb                   # bead position along the one thread
    bth = bt * _LT * 2.0 * np.pi
    br = 0.18 + 1.30 * bt
    bead_c = np.stack([br * np.cos(bth), np.full(nb, 0.55, np.float32),
                       br * np.sin(bth)], 1).astype(np.float32)   # beads strung along the SAME spiral
    phi = (lf / G) * (1.75 * 2.0 * np.pi)
    wrap = np.stack([0.045 * np.cos(phi), (lf / G - 0.5) * 0.06, 0.045 * np.sin(phi)], 1).astype(np.float32)
    nucleo_c = bead_c[bead] + wrap
    nucleo = np.empty_like(card)
    nucleo[a_tok] = nucleo_c + small
    nucleo[b_tok] = nucleo_c - small

    # --- stage 5 frame: coil the ONE bead string into ONE 30 nm FIBRE (a solenoid of beads). The bead
    # centres now spiral up a compact helix instead of lying flat — the string coils, the thread shrinks. ---
    fth = bt * 40.0 * 2.0 * np.pi                     # the flat spiral lifts into a tight 3-D solenoid
    fy = 0.55 + (bt - 0.5) * 2.4
    fib_bead = np.stack([0.36 * np.cos(fth), fy, 0.36 * np.sin(fth)], 1).astype(np.float32)
    fibre_c = fib_bead[bead] + wrap
    fibre = np.empty_like(card)
    fibre[a_tok] = fibre_c + small
    fibre[b_tok] = fibre_c - small

    # --- stage 6/7 frame: fold the ONE fibre into ONE compact CHROMATID — the chromosome. The fibre's bead
    # axis coils into a short fat rod, pinched at the centromere (narrow middle, fatter arms). Ultimate
    # compaction: the whole card, now one tight body. Chains from the fibre exactly. ---
    yb = (np.arange(nb) / max(nb - 1, 1) - 0.5) * 2.0     # -1..1 along the thread
    pinch = 0.30 + 0.70 * np.abs(yb)                      # centromere pinch in the middle
    angc = np.arange(nb) * (2.0 * np.pi / 6.0)            # the fibre coils up the chromatid axis
    chromo_bead = np.stack([0.16 * pinch * np.cos(angc), 0.55 + yb * 0.85,
                            0.16 * pinch * np.sin(angc)], 1).astype(np.float32)
    chromo_c = chromo_bead[bead] + wrap
    chromo = np.empty_like(card)
    chromo[a_tok] = chromo_c + small
    chromo[b_tok] = chromo_c - small

    # --- colours ---
    col_card = np.tile(np.array([0.30, 0.36, 0.32], np.float32), (n, 1))   # muted board material
    hue = (ids[:n].astype(np.float64) * 0.61803) % 1.0
    col_token = _hsv(hue, 0.60, 0.88)                 # tokenised: coloured by type id
    col_base = col_token.copy()
    a_base = (ids[a_tok] & 3)
    col_base[a_tok] = _BASES[a_base]
    col_base[b_tok] = _BASES[a_base ^ 1]              # complementary base on the partner strand

    return Thread(card=card, ladder=ladder, helix=helix, nucleo=nucleo, fibre=fibre, chromo=chromo,
                  col_card=col_card, col_token=col_token, col_base=col_base, read=read,
                  xspan=(float(pos[:, 0].min()), float(pos[:, 0].max())))


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# stage boundaries in global progress [0,1]: card -> tokenize -> ladder -> helix -> nucleosome -> fibre.
# Each interval folds one end frame toward the next; the boundaries are exact hand-offs (end IS start).
_STAGES = [
    ("scan", 0.15),      # scan sweeps the card, colouring it into tokens (positions fixed)
    ("pair", 0.30),      # token stream folds onto the one base-pair ladder
    ("helix", 0.44),     # the ladder winds into one double helix
    ("nucleo", 0.60),    # the helix wraps into beads on one string
    ("fibre", 0.78),     # the bead string coils into one 30 nm fibre
    ("chromo", 1.00),    # the fibre folds into one compact chromatid (the chromosome)
]


def frame(th: Thread, progress: float):
    """Positions + colours of the whole thread at global ``progress`` in [0,1]. Folds one stage's end frame
    toward the next — never re-derives — so it is always continuous."""
    g = float(np.clip(progress, 0.0, 1.0))

    # SCAN (positions fixed on the card; colour sweeps card -> token)
    lo = 0.0
    if g <= _STAGES[0][1]:
        x0, x1 = th.xspan
        front = x0 + (x1 - x0 + 0.2) * _smooth(g / _STAGES[0][1])
        rev = np.clip((front - th.card[:, 0]) * 3.0, 0.0, 1.0)[:, None]
        return th.card, th.col_card * (1.0 - rev) + th.col_token * rev

    # position keyframes per fold stage, and the colour they carry
    frames = [th.card, th.ladder, th.helix, th.nucleo, th.fibre, th.chromo]
    for k in range(1, len(_STAGES)):
        a_lo, a_hi = _STAGES[k - 1][1], _STAGES[k][1]
        if g <= a_hi or k == len(_STAGES) - 1:
            t = _smooth((g - a_lo) / max(a_hi - a_lo, 1e-6))
            pos = frames[k - 1] * (1.0 - t) + frames[k] * t
            if k == 1:                                # card->ladder: colour token -> base pair
                col = th.col_token * (1.0 - t) + th.col_base * t
            else:
                col = th.col_base
            return pos, col
    return th.fibre, th.col_base
