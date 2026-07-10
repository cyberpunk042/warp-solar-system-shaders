"""Particle-simulation subsystem: real Warp physics (gravity, buoyancy, drag)
driving nuclear / thermonuclear blast simulations."""

from .blast import simulate
from .engine import ParticleSystem

__all__ = ["simulate", "ParticleSystem"]
