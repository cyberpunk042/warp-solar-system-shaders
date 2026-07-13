"""Smoke tests for the electronics component scenes.

Renders each at a tiny resolution and asserts finite, in-range, non-degenerate
output — silicon materials, discrete components, packaging, and the memory /
logic cells that a RAM / SSD / GPU is built from.

    python -m tests.test_electronics
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_SCENES = [
    "silicon_ingot", "silicon_crystal", "silicon_wafer", "pn_junction",
    "resistor", "capacitor", "led", "inductor", "crystal_oscillator",
    "pcb", "ic_package", "bga", "bond_wire",
    "dram_cell", "nand_flash_cell", "cmos_inverter", "sram_cell",
]


def _check(name, time):
    img = np.asarray(ws.render(name, width=96, height=72, time=time), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def main():
    wp.init()
    for n in _SCENES:
        _check(n, 0.4)
    print(f"ALL PASSED ({len(_SCENES)} scenes)")


if __name__ == "__main__":
    main()
