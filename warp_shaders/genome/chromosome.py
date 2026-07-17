"""Process 7 — the chromosome: the telomere-capped fibre folds into a chromosome.

Chains from Process 6's actual output (the fibre with its two t-loop telomere caps). Two honest forms, both
selectable:

- **single chromatid** (``fold_chromosome``): one continuous strand condenses into a single rod — a
  centromere constriction at the middle, the two real **telomere** t-loops capping its two ends. Fully
  conserving: nothing is copied.
- **metaphase X** (``replicate_chromosome`` in ``replication.py``): the strand first **replicates** (the
  one place biology legitimately makes a copy — S-phase), then the two identical sister chromatids condense
  side by side, joined at the centromere — the classic X, four telomeres.

Conserving and physical: every base pair is folded (not regenerated) onto the rod, continuously; the
telomere caps are carried intact to the tips; nothing spawned (bar the explicit, shown replication in the
X form), nothing teleports. This lib supplies the two end states; ``scenes/warp_chromosome`` animates it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .telomere import cap_telomeres


@dataclasses.dataclass
class Chromosome:
    """The base pairs in two states: ``tel_a`` / ``tel_b`` (P,3) as Process 6 left them (fibre + t-loops),
    and ``chr_a`` / ``chr_b`` (P,3) folded into the single-chromatid chromosome. ``a_col`` / ``b_col`` the
    (telomere-tinted) base colours, ``is_tel`` (P,) telomeric pairs, ``arm_s`` (P,) position along the
    chromatid (0 = one telomere, 0.5 = centromere, 1 = the other telomere)."""

    tel_a: np.ndarray
    tel_b: np.ndarray
    chr_a: np.ndarray
    chr_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    is_tel: np.ndarray
    arm_s: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.tel_a.shape[0])


def fold_chromosome(sub: int = 2, block: int = 5, height: float = 5.2, rod: float = 1.5,
                    n_turns: float = 20.0, coil_frac: float = 0.60, rope: float = 0.55) -> Chromosome:
    """Fold Process 6's telomere-capped strand into a single condensed chromatid — a **real, pretty**
    chromosome, built procedurally from the physics of condensation.

    Condensation is **hierarchical coiling**: the 30 nm fibre from Process 5 does not smear into a blob, it
    **coils** — the chromonema winds as a dense helical **solenoid** up each arm (the well-known coiled-coil
    of a metaphase chromatid). That is what this fold does:

    - a smooth **envelope** shapes the body: two fat sausage arms with **rounded caps** at the telomere
      ends and a distinct **centromere** waist (a Gaussian pinch) at the middle;
    - the fibre winds a dense **helix** (``n_turns``) around each arm's axis at fraction ``coil_frac`` of the
      local radius, and each turn is a **thick rope** (``rope``, a golden-angle micro-disk fill) so the arm
      is **opaque** — the front of the coil occludes the back, no see-through, with the helix reading as the
      chromosome's characteristic coil grooves;
    - the two real **telomere** t-loops tuck into the two rounded tips as small green knots.

    Every base pair is *folded* onto the coil (nothing regenerated, nothing spawned); conservation holds."""
    tl = cap_telomeres(sub=sub, block=block)
    p = tl.n_pairs
    i = np.arange(p)
    s = (i / (p - 1)).astype(np.float64)                 # 0 at end-0 telomere, 1 at end-1 telomere

    # --- envelope: rounded-cap sausage arms + centromere waist ---------------------------------------
    env = np.sqrt(np.clip(1.0 - (2.0 * s - 1.0) ** 8, 0.0, 1.0))      # ~1 across the arms, rounds at the tips
    notch = 0.52 * np.exp(-((s - 0.5) / 0.055) ** 2)                  # the centromere primary constriction
    radius = np.clip(rod * (env - notch), 0.045, None)               # local body radius along the chromatid
    axis_y = height * (1.0 - 2.0 * s)                                 # tip(+y) → centromere(0) → tip(−y)

    # --- the chromonema solenoid: a dense helix, wound as a THICK rope so the arm is opaque -----------
    phi = s * n_turns * 2.0 * np.pi                                   # macro coil angle (the visible grooves)
    frac = lambda v: v - np.floor(v)
    micro_r = rope * radius * np.sqrt(frac(i * 0.6180339887))         # golden-angle micro-disk → fills the rope
    micro_a = i * 2.399963229
    rho = radius * coil_frac                                          # rope centre-line radius on the arm axis

    def wind(phase, dr):
        cphi, sphi = np.cos(phi + phase), np.sin(phi + phase)         # radial (x,z) direction of the coil
        r = rho + dr                                                  # the paired rail sits a touch off-centre
        cx = r * cphi + micro_r * np.cos(micro_a) * cphi              # rope centre + micro fill in the radial…
        cz = r * sphi + micro_r * np.cos(micro_a) * sphi
        cyv = axis_y + micro_r * np.sin(micro_a)                     # …and vertical directions → a fat tube
        return np.stack([cx, cyv, cz], 1).astype(np.float32)

    chr_a = wind(0.0, +0.10 * rod)
    chr_b = wind(np.pi, -0.10 * rod)                     # partner rail: opposite side of the coil → double-helix rope

    # --- the two telomere t-loops, small green knots tucked onto the two rounded tips -----------------
    cap = 0.26
    for end, mask, tip_y in ((0, i < tl.tel_len, height), (1, i >= p - tl.tel_len, -height)):
        m = mask
        loop = tl.tel_a[m]
        loop_off = tl.tel_b[m] - tl.tel_a[m]
        anchor = loop.mean(axis=0)                        # centre the knot on its own centroid
        tip = np.array([0.0, tip_y * 1.0, 0.0], np.float32)
        chr_a[m] = (tip + (loop - anchor) * cap).astype(np.float32)
        chr_b[m] = (tip + (loop - anchor + loop_off) * cap).astype(np.float32)

    return Chromosome(tel_a=tl.tel_a, tel_b=tl.tel_b, chr_a=chr_a, chr_b=chr_b,
                      a_col=tl.a_col.copy(), b_col=tl.b_col.copy(), is_tel=tl.is_tel,
                      arm_s=s.astype(np.float32))
