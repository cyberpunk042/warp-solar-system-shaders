"""The HF Space compute (hf_space/demo.py) — the CPU KV-compression measurement behind the gradio app."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_space.demo import measure, summary_markdown


def test_measure_reports_sane_metrics_on_cpu():
    m = measure(1024, bits=4, layers=2, heads=4, dim=64, device="cpu")
    assert m["chromofold_mb"] < m["fp16_mb"]                    # compressed below fp16
    assert m["vram_x"] > 1.5                                    # int4 KV is a few x smaller
    assert m["attn_mse"] < 1e-2                                 # attention error is small (lossless over quant)
    assert m["window_ms"] > 0 and m["decode_all_ms"] > 0
    assert set(("context_len", "bits", "b_per_val", "speedup_x")) <= set(m)


def test_int8_is_near_lossless_and_less_compressed_than_int4():
    m4 = measure(1024, bits=4, layers=2, device="cpu")
    m8 = measure(1024, bits=8, layers=2, device="cpu")
    assert m8["attn_mse"] <= m4["attn_mse"]                     # more bits -> lower error
    assert m8["vram_x"] < m4["vram_x"]                          # more bits -> less compression


def test_summary_markdown_renders():
    md = summary_markdown(measure(512, bits=4, layers=2, device="cpu"))
    assert "VRAM" in md and "ChromoFold" in md and md.startswith("### Context")
