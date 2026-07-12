"""Buildings — a parametric SDF architecture kit (towers, houses, blocks, and a
domain-repeated city). See ``docs/research/17-buildings.md``."""

from .sdf import (
    city_de, sd_block, sd_house, sd_tower, sd_triprism, window_mask,
)

__all__ = [
    "sd_tower", "sd_house", "sd_block", "sd_triprism", "city_de", "window_mask",
]
