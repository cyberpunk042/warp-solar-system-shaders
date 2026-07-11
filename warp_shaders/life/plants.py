"""A small library of plant grammars + a grow-to-mesh helper.

Each entry returns a :class:`PlantSpec` (an :class:`~warp_shaders.life.lsystem.LSystem`
plus a :class:`~warp_shaders.life.turtle.TurtleConfig` and a full-growth generation
count). :func:`grow_mesh` derives a spec to a generation, interprets it, and
tessellates it — with a small per-(spec, generation) cache so animating growth
doesn't re-derive every frame.

Grammars follow ABOP: the parametric tapering tree is a ternary 3D tree
(fig 1.25 / the parametric style of §1.10); the herb is a stochastic bracketed
plant with leaves; grass is a tuft of arching blades.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lsystem import LSystem, Module, Rule, parse
from .mesh import Mesh, build_mesh
from .turtle import TurtleConfig, interpret


@dataclass
class PlantSpec:
    lsystem: LSystem
    cfg: TurtleConfig
    gens: int          # generations for full growth
    sides: int = 5     # tube tessellation


# --- parametric 3D tree (ternary, tapering) ----------------------------------

def _tree_producer(cl=0.65, cw=0.72, ang=32.0, wmin=0.055, leaf=5.0):
    def produce(m: Module, l, r):
        length, width = m.params
        if width <= wmin:                       # twig tip -> a cluster of leaves
            return parse(f"'(2)[L({length * leaf:.4f})][/(120)L({length * leaf:.4f})]"
                         f"[/(240)L({length * leaf:.4f})]")
        L2, W2 = length * cl, width * cw
        child = f"[&({ang:g})A({L2:.4f},{W2:.4f})]"
        s = (f"!({width:.4f})F({length:.4f})"
             f"{child}/(120){child}/(120){child}")
        return parse(s)
    return produce


def tree() -> PlantSpec:
    ls = LSystem("A(1.4,0.30)", {"A": Rule("A", _tree_producer())})
    cfg = TurtleConfig(step=1.0, angle=32.0, radius=0.30, leaf_size=1.0)
    return PlantSpec(ls, cfg, gens=7, sides=5)


# --- leafy herb (stochastic bracketed plant) ---------------------------------

def herb(seed: int = 3) -> PlantSpec:
    # a central stem with leafy side branches in golden-angle (137.5) phyllotaxis;
    # B is a short branch that carries stems (F) AND leaves, so it spreads
    rules = {
        "X": [
            ("F[&(40)B]/(137.5)X", 0.5),
            ("F[&(40)B]/(137.5)[^(18)B]/(137.5)X", 0.3),
            ("F[&(48)B]/(137.5)X", 0.2),
        ],
        "B": "F[+(28)'(1)L]F[-(28)'(1)L]F'(1)L",
    }
    ls = LSystem("X", rules, seed=seed)
    cfg = TurtleConfig(step=0.5, angle=26.0, radius=0.045, leaf_size=0.9)
    return PlantSpec(ls, cfg, gens=6, sides=5)


# --- grass tuft (arching blades) ---------------------------------------------

def grass(seed: int = 1) -> PlantSpec:
    # one blade b arches upward, extending one internode per generation;
    # the tuft rolls several blades around the base
    blade = "'(1)!(0.045)F(0.5)&(10)b"
    tuft = "[b]/(67)[b]/(129)[b]/(198)[b]/(271)[b]"
    ls = LSystem(tuft, {"b": [(blade, 1.0)]}, seed=seed)
    cfg = TurtleConfig(step=0.5, angle=10.0, radius=0.045, leaf_size=0.6)
    return PlantSpec(ls, cfg, gens=9, sides=4)


# --- environmentally-responsive plants (ABOP §2.3.4) -------------------------
# These carry no tropism in their own cfg; the scene sets the environment
# (light target, gravity susceptibility, leaf-fold) on a per-frame TurtleConfig
# so the same grammar visibly responds to a moving light or to rain. This is
# the "obvious rules" substrate the future mind layer will steer.

def sapling(seed: int = 5) -> PlantSpec:
    # a young leafy stem: straight-ish internodes with paired leaves, tall
    # enough that a phototropic bend along its length reads clearly
    rules = {
        "X": [
            ("F[+(35)'(2)L]F[-(35)'(2)L]FX", 0.6),
            ("F[-(35)'(2)L]F[+(35)'(2)L]FX", 0.4),
        ],
    }
    ls = LSystem("X", rules, seed=seed)
    cfg = TurtleConfig(step=0.34, angle=30.0, radius=0.05, leaf_size=0.85)
    return PlantSpec(ls, cfg, gens=7, sides=5)


def weeper(seed: int = 2) -> PlantSpec:
    # a whippy plant whose side shoots sag: many internodes per shoot so a
    # gentle gravitropism arches them into a weeping form
    rules = {
        "X": [
            ("F[&(20)/(90)S]/(137.5)X", 0.6),
            ("F[&(25)/(90)S][&(25)/(270)S]/(137.5)X", 0.4),
        ],
        "S": "FFF[+(15)'(2)L]FFF[-(15)'(2)L]FFF'(2)L",
    }
    ls = LSystem("X", rules, seed=seed)
    cfg = TurtleConfig(step=0.3, angle=24.0, radius=0.05, leaf_size=0.7)
    return PlantSpec(ls, cfg, gens=6, sides=5)


_REGISTRY = {"tree": tree, "herb": herb, "grass": grass,
             "sapling": sapling, "weeper": weeper}
_spec_cache: dict = {}
_mesh_cache: dict = {}
_word_cache: dict = {}


def get_spec(name: str) -> PlantSpec:
    """Return the (memoized) PlantSpec for a name, so the mesh cache stays warm."""
    if name not in _spec_cache:
        _spec_cache[name] = _REGISTRY[name]()
    return _spec_cache[name]


def grow_mesh(spec: PlantSpec, gen: int) -> Mesh:
    """Derive `spec` to generation `gen`, interpret + tessellate (cached)."""
    gen = max(0, min(gen, spec.gens))
    key = (id(spec.lsystem), gen, spec.sides)
    m = _mesh_cache.get(key)
    if m is None:
        word = spec.lsystem.derive(gen)
        geo = interpret(word, spec.cfg)
        m = build_mesh(geo, sides=spec.sides)
        _mesh_cache[key] = (m, geo.bounds())
    return _mesh_cache[key]


def derive_word(spec: PlantSpec, gen: int):
    """Derive `spec` to generation `gen` (word cached; interpretation isn't)."""
    gen = max(0, min(gen, spec.gens))
    key = (id(spec.lsystem), gen)
    w = _word_cache.get(key)
    if w is None:
        w = spec.lsystem.derive(gen)
        _word_cache[key] = w
    return w


def grow_mesh_env(spec: PlantSpec, gen: int, cfg: TurtleConfig) -> Mesh:
    """Grow with an environment-modified turtle config (uncached).

    The word for a generation is cached, but interpretation runs every call so
    a moving light / changing rain re-shapes the same structure per frame.
    Returns ``(Mesh, (lo, hi))`` like :func:`grow_mesh`.
    """
    word = derive_word(spec, gen)
    geo = interpret(word, cfg)
    return build_mesh(geo, sides=spec.sides), geo.bounds()
