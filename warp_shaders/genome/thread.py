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
    bound: np.ndarray           # stage 2a — tokens bound in twos into base-pair rungs, in place on the card
    ladder: np.ndarray          # stage 2b — the rungs laid onto the one base-pair ladder
    a_tok: np.ndarray           # (P,) token index of each pair's backbone-A; b_tok its backbone-B
    b_tok: np.ndarray
    helix: np.ndarray           # stage 3 — the ladder wound into one double helix (twist along the thread)
    nucleo: np.ndarray          # stage 4 — the helix wrapped into beads on one string (nucleosomes)
    fibre: np.ndarray           # stage 5 — the bead string coiled into one 30 nm fibre (solenoid)
    chromo: np.ndarray          # stage 6/7 — the fibre folded into one compact chromatid (the chromosome)
    col_card: np.ndarray        # muted card material
    col_token: np.ndarray       # tokenised: coloured by merge-codec type
    col_base: np.ndarray        # base-pair colour (A/T/G/C)
    col_chromo: np.ndarray      # chromosome: purple banded chromatin (the metaphase X)
    read: np.ndarray            # (N,) scan/read order — position of each token along the thread
    xspan: tuple                # (xmin, xmax) of the card, for the scan bar
    # --- chromatid-fold internals (per pair), so a chromatid can be re-folded with a different centromere
    # position / size — e.g. a metacentric X vs an acrocentric Y. (Defaults None for older callers.) ---
    bead: np.ndarray = None     # (P,) which nucleosome bead each pair belongs to
    wrap: np.ndarray = None     # (P,3) the pair's offset within its bead (the nucleosome wrap)
    nb: int = 0                 # number of beads
    small: np.ndarray = None    # (3,) backbone half-separation at chromatid compaction

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

    # --- stage 2a frame: BIND in twos. Consecutive tokens along the read are neighbours on the card, so
    # each pair's two tokens meet in place and stand up as a base-pair rung (A below, B above) — the pairing,
    # before anything moves to the ladder. ---
    pair_mid = 0.5 * (card[a_tok] + card[b_tok])
    ubind = np.array([0.0, 0.03, 0.0], np.float32)
    bound = np.empty_like(card)
    bound[a_tok] = pair_mid - ubind
    bound[b_tok] = pair_mid + ubind

    # ONE base-pair ladder of UPRIGHT RUNGS (base A on top, base B below), ordered as a single boustrophedon
    # thread. Base pairing is COMPRESSION — two tokens combine into one pair — so this field is TIGHTER than
    # the spread card (~half its footprint): the tokens draw in and combine, the thread shrinks. Upright,
    # dense rungs = the base-pair look, but compact. (Later folds still spiral; `_LT` kept for them.)
    i = np.arange(p)
    _LT = 24.0
    _FNX = 220
    row = i // _FNX
    colr = np.where(row % 2 == 0, i % _FNX, _FNX - 1 - (i % _FNX))    # boustrophedon: one snaking thread
    nrow = int(row.max()) + 1
    _frac = lambda v: v - np.floor(v)
    jx = _frac(np.sin(i * 12.9898) * 43758.5453) - 0.5               # break the lattice -> organic
    jz = _frac(np.sin(i * 78.2330 + 2.0) * 43758.5453) - 0.5
    sx, sz = 0.020, 0.019                                            # tight spacing -> ~half the card area
    cx = ((colr - _FNX * 0.5) + 0.6 * jx) * sx
    cz = ((row - nrow * 0.5) + 0.6 * jz) * sz
    rung = np.stack([cx, np.full(p, 0.9, np.float32), cz], 1).astype(np.float32)
    up = np.array([0.0, 0.075, 0.0], np.float32)                     # upright rung: A on top, B below

    ladder = np.empty_like(card)
    ladder[a_tok] = rung + up                          # each token flows to its backbone site on the rung
    ladder[b_tok] = rung - up

    # --- stage 3 frame: wind the ONE ladder into ONE double helix. The thread's path (rung centres) stays
    # put; along it the two backbones twist around each other — real B-DNA (~10.5 bp/turn). Local frame
    # (tangent/normal/binormal) so the twist follows the snaking thread. Chains from the ladder exactly. ---
    # the base-pair ladder winds into a FOREST OF DOUBLE HELICES (the dedicated warp_helix look): the one
    # thread runs column by column (boustrophedon), and within each column the two backbones spiral up at
    # 10.5 bp/turn — tall twisting strands. The footprint is tighter than the flat base-pair field
    # (compression continues) but the pairs now stand up into the forest.
    _HG = 90                                                   # base pairs per helical column
    ncolh = (p + _HG - 1) // _HG
    hcol = i // _HG
    hl_pos = (i % _HG).astype(np.float32)                      # 0..HG within the column
    hnx = max(int(round(np.sqrt(ncolh * 1.7))), 1)
    hcrow = hcol // hnx
    hccol = np.where(hcrow % 2 == 0, hcol % hnx, hnx - 1 - (hcol % hnx))   # boustrophedon: one thread
    hnz = int(hcrow.max()) + 1
    _hcx = (hccol - hnx * 0.5) * 0.135                         # compact column grid (tighter than the field)
    _hcz = (hcrow - hnz * 0.5) * 0.135
    hy = 0.35 + (hl_pos / _HG) * 1.35                          # the column rises — tall strands
    hcenter = np.stack([_hcx, hy, _hcz], 1).astype(np.float32)
    theta = (hl_pos * (2.0 * np.pi / 10.5))[:, None]           # real B-DNA: 10.5 bp per turn, twisting up
    off = np.concatenate([0.045 * np.cos(theta), np.zeros_like(theta), 0.045 * np.sin(theta)], 1)
    helix = np.empty_like(card)
    helix[a_tok] = hcenter + off.astype(np.float32)           # the two backbones spiral around the column
    helix[b_tok] = hcenter - off.astype(np.float32)

    # --- stage 4 frame: wrap the helix into BEADS ON ONE STRING (nucleosomes). Group G pairs per bead; the
    # bead centres run as one compact serpentine (fewer than the ladder — tighter), and inside each bead the
    # strand wraps ~1.75 turns around the core. Same ordered pairs, folded tighter. ---
    tw = 0.008                                        # the dsDNA is thin at this compaction
    small = np.array([0.0, tw, 0.0], np.float32)
    G = 180                                           # ~180 bp per nucleosome bead
    nb = (p + G - 1) // G
    bead = np.arange(p) // G
    lf = (np.arange(p) % G).astype(np.float32)
    bt = (np.arange(nb) + 0.5) / nb                   # bead position along the one thread (used by fibre)
    # bead centres on ONE compact serpentine STRING (beads on a string) — tighter than the helix.
    bnx = max(int(round(np.sqrt(nb * 1.5))), 1)
    brow = np.arange(nb) // bnx
    bcol = np.where(brow % 2 == 0, np.arange(nb) % bnx, bnx - 1 - (np.arange(nb) % bnx))
    bnz = int(brow.max()) + 1
    bead_c = np.stack([(bcol - bnx * 0.5) * 0.15, np.full(nb, 0.55, np.float32),
                       (brow - bnz * 0.5) * 0.15], 1).astype(np.float32)
    # inside each bead the DNA wraps ~1.75 turns around the histone core -> a squat disc (the nucleosome)
    phi = (lf / G) * (1.75 * 2.0 * np.pi)
    wrap = np.stack([0.058 * np.cos(phi), (lf / G - 0.5) * 0.03, 0.058 * np.sin(phi)], 1).astype(np.float32)
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
    # ONE chromatid of the metaphase X: a banded rod pinched at the centromere (origin), TILTED so that this
    # arm-pair is one diagonal of the X; its mirror (rendered by the scene) is the sister -> the four-arm X.
    yb = (np.arange(nb) / max(nb - 1, 1) - 0.5) * 2.0     # -1..1 along the chromatid
    pinch = 0.30 + 0.70 * np.abs(yb)                      # centromere pinch in the middle
    angc = np.arange(nb) * (2.0 * np.pi / 6.0)            # the fibre coils up the chromatid axis
    _angx = 0.34
    _cax, _sax = np.cos(_angx), np.sin(_angx)
    _xl = 0.16 * pinch * np.cos(angc)
    _yl = yb * 0.95
    chromo_bead = np.stack([_xl * _cax - _yl * _sax, _xl * _sax + _yl * _cax,
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

    # chromosome: purple chromatin with banding along the thread (the metaphase X look)
    band = 0.5 + 0.5 * np.sin(np.arange(p) * 0.5 + 0.4) * np.sin(np.arange(p) * 0.13)
    _dark = np.array([0.40, 0.28, 0.60], np.float32)
    _lite = np.array([0.82, 0.72, 0.94], np.float32)
    cpair = (_dark[None] + (_lite - _dark)[None] * band[:, None]).astype(np.float32)
    col_chromo = np.empty_like(card)
    col_chromo[a_tok] = cpair
    col_chromo[b_tok] = cpair

    return Thread(card=card, bound=bound, ladder=ladder, a_tok=a_tok, b_tok=b_tok,
                  helix=helix, nucleo=nucleo, fibre=fibre, chromo=chromo,
                  col_card=col_card, col_token=col_token, col_base=col_base, col_chromo=col_chromo,
                  read=read, xspan=(float(pos[:, 0].min()), float(pos[:, 0].max())),
                  bead=bead, wrap=wrap, nb=nb, small=small)


def fold_chromatid(th: Thread, centromere: float = 0.0, size: float = 1.0, tilt: float = 0.34,
                   arm: float = 0.16):
    """Re-fold the thread into ONE chromatid with the centromere at ``centromere`` (in [-1,1] along the arm
    axis) and overall ``size``. centromere=0 => metacentric (the symmetric X); centromere>0 => acrocentric,
    a short arm-pair and a long arm-pair (a real Y). ``arm`` is the coil radius — smaller = thinner, cleaner
    rods (a Y wants tight arms so it reads as a narrow chromosome, not a flaring vase). The chromatid is
    centred on its centromere so the scene's x-mirror joins the two sisters there. Same bead/wrap fold as
    ``build``."""
    nb = int(th.nb)
    yb = (np.arange(nb) / max(nb - 1, 1) - 0.5) * 2.0
    pinch = 0.30 + 0.70 * np.abs(yb - float(centromere))          # waist (min) sits at the centromere
    pinch = np.minimum(pinch, 1.0)                                # cap the flare so long arms stay clean rods
    angc = np.arange(nb) * (2.0 * np.pi / 6.0)
    ca, sa = np.cos(tilt), np.sin(tilt)
    xl = arm * pinch * np.cos(angc)
    yl = yb * 0.95
    cb = np.stack([xl * ca - yl * sa, xl * sa + yl * ca, arm * pinch * np.sin(angc)], 1).astype(np.float32)
    c = (cb[th.bead] + th.wrap) * float(size)
    out = np.empty((th.n, 3), np.float32)
    out[th.a_tok] = c + th.small * float(size)
    out[th.b_tok] = c - th.small * float(size)
    waist = np.abs(yb[th.bead] - float(centromere)) < 0.12        # tokens at the centromere -> centre on them
    m = np.zeros(th.n, bool)
    m[th.a_tok[waist]] = True
    m[th.b_tok[waist]] = True
    out -= out[m].mean(0) if m.any() else out.mean(0)
    return out


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# stage boundaries in global progress [0,1]: card -> tokenize -> ladder -> helix -> nucleosome -> fibre.
# Each interval folds one end frame toward the next; the boundaries are exact hand-offs (end IS start).
_STAGES = [
    ("scan", 0.14),      # scan sweeps the card, colouring it into tokens (positions fixed)
    ("bind", 0.26),      # tokens bind in twos into base-pair rungs, in place
    ("pair", 0.40),      # the rungs arrange onto the one base-pair ladder
    ("helix", 0.52),     # the ladder winds into one double helix
    ("nucleo", 0.66),    # the helix wraps into beads on one string
    ("fibre", 0.82),     # the bead string coils into one 30 nm fibre
    ("chromo", 1.00),    # the fibre folds into one compact chromatid (the chromosome)
]


def scan_end() -> float:
    """Global progress at which the tokenization scan finishes (the card is fully read into tokens)."""
    return _STAGES[0][1]


def scan_front(th: Thread, progress: float) -> float:
    """World x of the scan wavefront at ``progress`` — the board is solid ahead of it (x > front) and has
    been read into token particles behind it (x <= front)."""
    g = float(np.clip(progress, 0.0, 1.0))
    x0, x1 = th.xspan
    return x0 + (x1 - x0 + 0.2) * _smooth(g / _STAGES[0][1])


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
    frames = [th.card, th.bound, th.ladder, th.helix, th.nucleo, th.fibre, th.chromo]
    last = len(_STAGES) - 1
    for k in range(1, len(_STAGES)):
        a_lo, a_hi = _STAGES[k - 1][1], _STAGES[k][1]
        if g <= a_hi or k == last:
            t = _smooth((g - a_lo) / max(a_hi - a_lo, 1e-6))
            if k == last:
                # FEED, honest: the fibre is drawn through the telomere 3' TIP (one conduit point) and laid
                # onto the chromatid in READ ORDER. Each pair routes fibre -> tip -> chromatid as the feed
                # front passes it (never a straight morph — the whole strand threads through the one tip).
                src = frames[k - 1]                       # the fibre (the source being reeled in)
                dst = frames[k]                           # its slot on the chromatid
                tip = np.array([0.30, 1.05, 0.0], np.float32)     # the telomere 3' tip — the conduit
                pfrac = ((th.read // 2) / max(th.n // 2, 1)).astype(np.float32)
                g = np.clip((t * 1.18 - pfrac) / 0.12 + 0.5, 0.0, 1.0)[:, None]   # 0=fibre .5=tip 1=chromatid
                to_tip = np.clip(g * 2.0, 0.0, 1.0)
                from_tip = np.clip((g - 0.5) * 2.0, 0.0, 1.0)
                pos = np.where(g <= 0.5, src + (tip - src) * to_tip, tip + (dst - tip) * from_tip)
                col = th.col_base * (1.0 - g) + th.col_chromo * g
                return pos.astype(np.float32), col.astype(np.float32)
            pos = frames[k - 1] * (1.0 - t) + frames[k] * t
            col = th.col_token * (1.0 - t) + th.col_base * t if k == 1 else th.col_base
            return pos, col
    return th.chromo, th.col_chromo
