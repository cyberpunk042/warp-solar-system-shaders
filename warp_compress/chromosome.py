"""The chromosome — hierarchical grammar coiling (Re-Pair), the DNA of the compressor.

Coiling wraps the sequence layer by layer. In each pass the **most frequent adjacent pair** of
symbols is replaced everywhere by a fresh symbol and a rule ``new -> (a, b)`` is recorded — one
*nucleosome*, a pair wrapped into a bead. Because a rule's members may themselves be earlier
rules, the wrapping compounds: nucleosomes coil into fibers, fibers into loops. What remains is a
compact **chromosome**: a rule dictionary (the reusable histone scaffold) plus a short top-level
strand (the chromatid). Uncoiling expands every rule back to its two members until only literal
symbols remain — exact, lossless decompression.

The alphabet convention: literal symbols are ``0..base-1`` (``base = 256`` for bytes); every rule
gets an id ``>= base`` assigned in creation order, so a serialized rule list needs no explicit ids.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Chromosome:
    """A coiled sequence: the ordered rule pairs (histone scaffold) + the top strand (chromatid)."""

    base: int
    rules: List[Tuple[int, int]]          # rule i has id (base + i) -> (a, b)
    top: List[int]

    @property
    def layers(self) -> int:
        """Deepest coil = longest expansion chain of any rule (the chromatin condensation level)."""
        depth: Dict[int, int] = {}

        def d(sym: int) -> int:
            if sym < self.base:
                return 0
            if sym in depth:
                return depth[sym]
            a, b = self.rules[sym - self.base]
            depth[sym] = 1 + max(d(a), d(b))
            return depth[sym]

        return max((d(self.base + i) for i in range(len(self.rules))), default=0)

    def stats(self) -> Dict[str, int]:
        return {
            "nucleosomes": len(self.rules),   # number of wrapped pairs (grammar rules)
            "top_symbols": len(self.top),     # length of the residual top strand
            "layers": self.layers,            # coiling depth
        }


def coil(seq: List[int], base: int = 256, min_count: int = 2, max_rules: int = 1 << 20) -> Chromosome:
    """Coil ``seq`` (symbols in ``0..base-1``) into a :class:`Chromosome` by Re-Pair.

    Stops when no adjacent pair repeats ``min_count`` times, or ``max_rules`` is reached."""
    work: List[int] = list(seq)
    rules: List[Tuple[int, int]] = []
    next_id = base

    while len(rules) < max_rules and len(work) >= 2:
        counts: Counter = Counter()
        prev_pair = None
        i = 0
        # count non-overlapping-safe: standard occurrence count of each adjacent pair
        while i < len(work) - 1:
            counts[(work[i], work[i + 1])] += 1
            i += 1

        if not counts:
            break
        (a, b), c = counts.most_common(1)[0]
        if c < min_count:
            break

        rules.append((a, b))
        rid = next_id
        next_id += 1

        # replace non-overlapping left-to-right occurrences of (a, b) with the new nucleosome
        out: List[int] = []
        i = 0
        n = len(work)
        while i < n:
            if i < n - 1 and work[i] == a and work[i + 1] == b:
                out.append(rid)
                i += 2
            else:
                out.append(work[i])
                i += 1
        work = out
        _ = prev_pair  # (kept for readability; counting above is position-based)

    return Chromosome(base=base, rules=rules, top=work)


def coil_snapshots(seq: List[int], base: int = 256, min_count: int = 2,
                   max_rules: int = 1 << 20, max_snaps: int = 80):
    """Coil like :func:`coil`, but also return the strand **after every pass**.

    Returns ``(snapshots, chromosome)`` where ``snapshots[0]`` is the raw strand and each later
    entry is the strand one nucleosome further coiled (subsampled to at most ``max_snaps`` frames,
    always keeping the first and last). This is what an animation replays to *watch* the folding
    happen layer by layer instead of only seeing before/after."""
    work: List[int] = list(seq)
    rules: List[Tuple[int, int]] = []
    next_id = base
    snaps: List[List[int]] = [list(work)]

    while len(rules) < max_rules and len(work) >= 2:
        counts: Counter = Counter()
        i = 0
        while i < len(work) - 1:
            counts[(work[i], work[i + 1])] += 1
            i += 1
        if not counts:
            break
        (a, b), c = counts.most_common(1)[0]
        if c < min_count:
            break
        rules.append((a, b))
        rid = next_id
        next_id += 1
        out: List[int] = []
        i = 0
        n = len(work)
        while i < n:
            if i < n - 1 and work[i] == a and work[i + 1] == b:
                out.append(rid)
                i += 2
            else:
                out.append(work[i])
                i += 1
        work = out
        snaps.append(list(work))

    chrom = Chromosome(base=base, rules=rules, top=work)

    if max_snaps and len(snaps) > max_snaps:
        keep = sorted({round(k * (len(snaps) - 1) / (max_snaps - 1)) for k in range(max_snaps)})
        snaps = [snaps[k] for k in keep]
    return snaps, chrom


def symbol_metrics(chrom: Chromosome):
    """Per-symbol ``(expanded_length, coil_depth)`` maps for every rule id in ``chrom``.

    ``expanded_length`` = how many literal symbols a nucleosome ultimately unwraps to (its mass);
    ``coil_depth`` = how many wrapping layers deep it sits. Literals (``< base``) are length 1,
    depth 0."""
    base = chrom.base
    size: Dict[int, int] = {}
    depth: Dict[int, int] = {}

    def sz(s: int) -> int:
        if s < base:
            return 1
        if s in size:
            return size[s]
        a, b = chrom.rules[s - base]
        size[s] = sz(a) + sz(b)
        return size[s]

    def dp(s: int) -> int:
        if s < base:
            return 0
        if s in depth:
            return depth[s]
        a, b = chrom.rules[s - base]
        depth[s] = 1 + max(dp(a), dp(b))
        return depth[s]

    for i in range(len(chrom.rules)):
        sz(base + i)
        dp(base + i)
    return size, depth


def uncoil(chrom: Chromosome) -> List[int]:
    """Expand a :class:`Chromosome` back to its literal symbol sequence (exact inverse of coil)."""
    base = chrom.base
    rules = chrom.rules
    out: List[int] = []
    # iterative expansion with an explicit stack (avoids recursion limits on deep coils)
    stack: List[int] = list(reversed(chrom.top))
    while stack:
        sym = stack.pop()
        if sym < base:
            out.append(sym)
        else:
            a, b = rules[sym - base]
            stack.append(b)
            stack.append(a)
    return out
