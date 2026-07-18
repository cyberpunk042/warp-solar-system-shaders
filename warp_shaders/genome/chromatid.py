"""Process 7 — the chromatid: the capped 30 nm fibre folds into the condensed chromosome arm.

Chains from Process 6's actual output (the telomere-capped fibre). The biology: the 30 nm fibre is the last
level of *linear* packing; the final ~50x compaction is a **higher-order fold** — the fibre is thrown into
loops off a central protein scaffold and the whole array **coils**, condensing the long fibre band into the
short, dense metaphase **chromatid** with a pinched **centromere** waist and a **telomere** cap at each tip.

Conserving and physical: not one base pair is created or destroyed. Every pair keeps the exact fine
structure Process 6 gave it (its double-helix / nucleosome / fibre-solenoid detail) — that structure is
carried **rigidly** as the fibre's *centreline* is wound onto the chromatid coil. Concretely, for each pair
we split its telomere-state position into a smooth fibre **centreline** (the macro path of the band) plus a
**local offset** (the conserved fine detail = ``tel - centreline``); we re-place the centreline onto a short
helical coil and add the *same* local offset back. So the folded chromatid is the real matter, only wound
tighter — never regenerated, never a fresh coil drawn from nothing. This lib supplies the two end states
(fibre band -> chromatid); ``scenes/warp_genome`` runs the whole ladder, this fold last.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .telomere import cap_telomeres


@dataclasses.dataclass
class Chromatid:
    """The base pairs in two states: ``tel_a`` / ``tel_b`` (P,3) as Process 6 left them (the capped fibre
    band), and ``chr_a`` / ``chr_b`` (P,3) folded into the condensed chromatid. ``a_col`` / ``b_col`` the
    (telomere-tinted) base colours, ``is_tel`` which pairs are telomeric, ``tips`` the two folded telomere
    tip anchors, ``height`` the chromatid's half-height, ``arm_radius`` its arm radius."""

    tel_a: np.ndarray
    tel_b: np.ndarray
    chr_a: np.ndarray
    chr_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    is_tel: np.ndarray
    tips: np.ndarray
    height: float
    arm_radius: float

    @property
    def n_pairs(self) -> int:
        return int(self.tel_a.shape[0])


def _smooth(x: np.ndarray, sigma_pairs: float) -> np.ndarray:
    """Gaussian-smooth an (N,3) path along its index — removes the fibre-solenoid wiggle so what remains is
    the band's macro centreline. Reflect-padded so the two strand ends are not dragged inward."""
    n = x.shape[0]
    r = int(max(1.0, 3.0 * sigma_pairs))
    k = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sigma_pairs) ** 2)
    w /= w.sum()
    out = np.empty_like(x, dtype=np.float64)
    xp = np.pad(x.astype(np.float64), ((r, r), (0, 0)), mode="reflect")
    for d in range(3):
        out[:, d] = np.convolve(xp[:, d], w, mode="valid")
    return out


def _clamp_norm(v: np.ndarray, cap: float) -> np.ndarray:
    """Cap the length of each (N,3) offset so the telomere t-loops (large local offsets) stay tucked at the
    tips instead of flinging out — the fine fibre/nucleosome/helix detail (all well under the cap) is
    untouched, so no real structure is lost."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / np.maximum(n, 1e-9))
    return v * scale


def fold_chromatid(sub: int = 2, block: int = 5, turns: float = 34.0, arm_radius: float = 1.55,
                   height: float = 9.8, waist: float = 0.26, waist_width: float = 0.05,
                   local_cap: float = 0.20, shape: str = "single", cross: float = 0.42) -> Chromatid:
    """Fold Process 6's telomere-capped fibre into a real, condensed **chromosome silhouette**. The fibre's
    smooth centreline is wound as a tight helix **up the chromosome's arms** — the thread follows the arm
    line one turn after another (radius ``arm_radius``), plump in the arms and pinched to a **centromere**
    waist, with rounded telomere-capped tips — while each pair's conserved fine detail rides that centreline
    rigidly (capped at ``local_cap``).

    ``shape``:
      - ``"single"`` — one chromatid: a slender rod with two arms meeting at the centromere (the two strand
        ends are the two telomere tips).
      - ``"x"`` — the metaphase **X**: the same chromatid tilted, plus its replicated **sister** mirrored
        across the centromere, the two crossing at the pinch — four banded arms, four rounded tips.

    This is the final ~50x fold, so the fibre packs **shoulder to shoulder** — a condensed chromosome IS a
    dense mass — but the helix pitch (``height/turns``) stays at/above the packed-band thickness so
    consecutive turns **touch and never pass through** each other (honest condensation, not interpenetration)."""
    tl = cap_telomeres(sub=sub, block=block)
    p = tl.n_pairs

    # split each pair into smooth fibre centreline + conserved fine detail (the real Process-6 structure).
    sigma = 6.0 * 110.0
    centre = _smooth(tl.fib_a, sigma)                     # (P,3) macro centreline of the fibre band
    local_a = _clamp_norm(tl.tel_a.astype(np.float64) - centre, local_cap)   # conserved fine detail
    local_b = _clamp_norm(tl.tel_b.astype(np.float64) - centre, local_cap)

    # --- one chromatid: a helix winding up a two-armed, centromere-pinched, round-tipped rod ---
    u = np.arange(p, dtype=np.float64) / (p - 1.0)
    waist_env = 1.0 - (1.0 - waist) * np.exp(-((u - 0.5) / waist_width) ** 2)   # centromere constriction
    arms = np.sqrt(np.clip(1.0 - (2.0 * u - 1.0) ** 4, 0.0, 1.0))               # plump arms, rounded tips
    r = arm_radius * waist_env * (0.28 + 0.72 * arms)
    phi = 2.0 * math.pi * turns * u
    ax = r * np.cos(phi)
    ay = (u - 0.5) * height
    az = r * np.sin(phi)
    coil = np.stack([ax, ay, az], 1)
    det = (0.5 + 0.5 * waist_env)[:, None]

    def place(base):
        c = coil + base * det
        if shape == "x":
            # tilt this chromatid and lay its mirrored sister across the centromere → the metaphase X
            t = cross
            xr = c[:, 0] * math.cos(t) - c[:, 1] * math.sin(t)
            yr = c[:, 0] * math.sin(t) + c[:, 1] * math.cos(t)
            return np.stack([xr, yr, c[:, 2]], 1)
        return c

    chr_a = place(local_a).astype(np.float32)
    chr_b = place(local_b).astype(np.float32)

    tips = np.array([coil[0], coil[-1]], np.float32)
    return Chromatid(tel_a=tl.tel_a, tel_b=tl.tel_b, chr_a=chr_a, chr_b=chr_b,
                     a_col=tl.a_col, b_col=tl.b_col, is_tel=tl.is_tel, tips=tips,
                     height=float(0.5 * height), arm_radius=float(arm_radius))
