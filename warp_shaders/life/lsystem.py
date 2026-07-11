"""L-Systems — the rewriting grammar under everything that grows here.

A Lindenmayer system rewrites a string of *modules* in parallel, every module at
once, over discrete generations — the mathematics of plant growth. This is the
foundation library (rendering-agnostic): it produces a word (a list of modules)
that the :mod:`warp_shaders.life.turtle` interpreter later turns into geometry.

Four classes are supported through one uniform :class:`Rule`:

- **D0L** — deterministic, context-free (one successor per symbol).
- **Stochastic** — several successors with weights; a seeded RNG picks one,
  reproducibly.
- **Context-sensitive** (IL / 1L / 2L) — a rule matches only with the required
  left / right neighbour (bracket-aware, with an ``ignore`` set).
- **Parametric** — modules carry numeric parameters; rules gate on a condition
  and compute successor parameters with arithmetic.

Source: Przemysław Prusinkiewicz & Aristid Lindenmayer, *The Algorithmic Beauty
of Plants* (Springer, 1990) — "ABOP" — chapters 1 (D0L, bracketed, stochastic,
context-sensitive) and 1.10 (parametric).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union


class Module:
    """One symbol with optional numeric parameters, e.g. ``F`` or ``A(1.5, 30)``."""

    __slots__ = ("sym", "params")

    def __init__(self, sym: str, params: Sequence[float] = ()):
        self.sym = sym
        self.params = tuple(float(p) for p in params)

    def __eq__(self, other):
        return (isinstance(other, Module) and self.sym == other.sym
                and self.params == other.params)

    def __hash__(self):
        return hash((self.sym, self.params))

    def __repr__(self):
        if self.params:
            return f"{self.sym}({','.join(f'{p:g}' for p in self.params)})"
        return self.sym


def parse(s: str) -> List[Module]:
    """Parse ``"F(1)[+(30)F]F"`` into a list of :class:`Module`. Whitespace ignored."""
    mods: List[Module] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        i += 1
        params: List[float] = []
        if i < n and s[i] == "(":
            j = s.index(")", i)
            params = [float(x) for x in s[i + 1:j].split(",") if x.strip() != ""]
            i = j + 1
        mods.append(Module(c, params))
    return mods


def word_to_str(word: Sequence[Module]) -> str:
    """Inverse of :func:`parse` (round-trips a parsed word)."""
    return "".join(repr(m) for m in word)


# --- rules -------------------------------------------------------------------

# A production computes the successor from (module, left_ctx, right_ctx).
Producer = Callable[[Module, Optional[Module], Optional[Module]], List[Module]]


@dataclass
class Rule:
    """One production. Match gates: symbol, optional left/right context, optional
    parametric condition; ``weight`` selects among matches for stochastic rules."""

    pred: str
    produce: Producer
    left: Optional[str] = None
    right: Optional[str] = None
    cond: Optional[Callable[[tuple], bool]] = None
    weight: float = 1.0


def _template_producer(template: str) -> Producer:
    mods = parse(template)
    return lambda m, l, r: [Module(x.sym, x.params) for x in mods]


# A rule spec in the `rules` dict may be:
#   "FF"                          (D0L string)
#   [("FF", 0.5), ("F", 0.5)]     (stochastic string successors + weights)
#   Rule(...) or [Rule(...), ...] (context-sensitive / parametric)
RuleSpec = Union[str, Rule, Sequence]


def _normalize(pred: str, spec: RuleSpec) -> List[Rule]:
    if isinstance(spec, str):
        return [Rule(pred, _template_producer(spec))]
    if isinstance(spec, Rule):
        return [spec]
    rules: List[Rule] = []
    for item in spec:
        if isinstance(item, Rule):
            rules.append(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            template, weight = item
            rules.append(Rule(pred, _template_producer(template), weight=float(weight)))
        else:
            raise TypeError(f"bad rule spec for {pred!r}: {item!r}")
    return rules


class LSystem:
    """An L-system: an axiom, a set of rules, and :meth:`derive` to grow it.

    ``rules`` maps a predecessor symbol to its :data:`RuleSpec`. ``ignore`` is the
    set of symbols skipped when finding context neighbours (turtle orientation
    symbols, typically). ``seed`` makes stochastic derivation reproducible.
    """

    def __init__(self, axiom: Union[str, Sequence[Module]],
                 rules: dict, ignore: str = "", seed: int = 0):
        self.axiom: List[Module] = parse(axiom) if isinstance(axiom, str) else list(axiom)
        self.rules: dict = {}
        for pred, spec in rules.items():
            self.rules.setdefault(pred, []).extend(_normalize(pred, spec))
        self.ignore = set(ignore)
        self.seed = seed

    # -- context (bracket-aware, ignoring the `ignore` set) --
    def _left_context(self, word, i) -> Optional[Module]:
        depth = 0
        for k in range(i - 1, -1, -1):
            s = word[k].sym
            if s == "]":
                depth += 1
            elif s == "[":
                if depth > 0:
                    depth -= 1        # skip a completed sibling branch
                else:
                    continue          # ascend to the branch's parent
            elif depth == 0 and s not in self.ignore:
                return word[k]
        return None

    def _right_context(self, word, i) -> Optional[Module]:
        for k in range(i + 1, len(word)):
            s = word[k].sym
            if s == "]":
                return None           # branch ended before a real neighbour
            if s == "[":
                continue              # right context may reach into a branch
            if s not in self.ignore:
                return word[k]
        return None

    def _matches(self, rule: Rule, m: Module, left, right) -> bool:
        if rule.left is not None and (left is None or left.sym != rule.left):
            return False
        if rule.right is not None and (right is None or right.sym != rule.right):
            return False
        if rule.cond is not None and not rule.cond(m.params):
            return False
        return True

    def step(self, word: List[Module], gen: int) -> List[Module]:
        """Rewrite every module once (one generation)."""
        rng = random.Random(f"{self.seed}:{gen}")
        out: List[Module] = []
        for i, m in enumerate(word):
            cands = self.rules.get(m.sym)
            if not cands:
                out.append(m)
                continue
            left = self._left_context(word, i)
            right = self._right_context(word, i)
            matches = [r for r in cands if self._matches(r, m, left, right)]
            if not matches:
                out.append(m)
                continue
            chosen = matches[0]
            if len(matches) > 1:
                total = sum(r.weight for r in matches)
                x = rng.random() * total
                acc = 0.0
                for r in matches:
                    acc += r.weight
                    if x <= acc:
                        chosen = r
                        break
            out.extend(chosen.produce(m, left, right))
        return out

    def derive(self, n: int) -> List[Module]:
        """Grow the axiom for ``n`` generations and return the resulting word."""
        word = list(self.axiom)
        for g in range(n):
            word = self.step(word, g)
        return word

    def derive_str(self, n: int) -> str:
        return word_to_str(self.derive(n))
