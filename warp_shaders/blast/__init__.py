"""Nuclear-detonation physics — declassified scaling laws driving the
`tsar_bomba` / `super_tsar` / `super_tsar_space` scenes. See
``docs/research/15-nuclear-fireball.md``."""

from .physics import (
    BlastParams, SUPER_TSAR, TSAR, blast_falloff, debris_shell_radius,
    destruction_radius, fireball_radius, fireball_temp, fireball_temp_at,
    light_radius, mushroom_height, overpressure_radius, severe_radius,
    shock_radius, smoothstep, thermal_radius,
)

__all__ = [
    "BlastParams", "TSAR", "SUPER_TSAR",
    "fireball_radius", "thermal_radius", "overpressure_radius",
    "destruction_radius", "severe_radius", "light_radius", "shock_radius",
    "mushroom_height", "fireball_temp", "debris_shell_radius",
    "fireball_temp_at", "smoothstep", "blast_falloff",
]
