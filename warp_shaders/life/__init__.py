"""warp_shaders.life — growing things, starting with L-Systems.

The rewriting grammar under everything alive here. This package builds up from
the pure L-System core (rendering-agnostic) toward geometry the Warp engine can
render, so plants can be *grown* and *shown*:

- **lsystem** — the L-System core + its classes (D0L / stochastic /
  context-sensitive / parametric). Source: Prusinkiewicz & Lindenmayer, *The
  Algorithmic Beauty of Plants* (1990).

Later layers (turtle interpreter → mesh → Warp renderer → growing scenes) attach
on top; the grand roadmap runs DNA → cell → grass → plant → tree, with a "mind"
layer to come.
"""

from .lsystem import LSystem, Module, Rule, parse, word_to_str
from .turtle import Geometry, Leaf, Segment, TurtleConfig, interpret
from .mesh import Mesh, build_mesh
from .plants import PlantSpec, get_spec, grow_mesh, grow_mesh_env
from .molecular import build_helix, build_protein

__all__ = [
    "LSystem", "Rule", "Module", "parse", "word_to_str",
    "interpret", "TurtleConfig", "Geometry", "Segment", "Leaf",
    "build_mesh", "Mesh",
    "PlantSpec", "get_spec", "grow_mesh", "grow_mesh_env",
    "build_helix", "build_protein",
]
