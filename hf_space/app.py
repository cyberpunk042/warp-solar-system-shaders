"""ChromoFold — Hugging Face Space (gradio). A compressed, searchable KV cache you can poke at, on CPU.

Deploy: create a Gradio Space, add this file + requirements.txt, and make `chromofold` importable (see README).
Local:  python hf_space/app.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from hf_space.demo import (SCALE_ALL_MS, SCALE_CF_MB, SCALE_FP16_MB, SCALE_N, SCALE_WIN_MS, measure,
                           summary_markdown)

_COPPER, _BRICK, _INK = "#B9691F", "#B23A2E", "#3B4653"


def _scale_fig(mark_n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=120)
    # left: access cost vs context (log y) — windowed flat, decode-all exploding
    ax1.plot(SCALE_N, SCALE_ALL_MS, "-o", color=_BRICK, ms=4, label="decode-all (O(context))")
    ax1.plot(SCALE_N, SCALE_WIN_MS, "-o", color=_COPPER, ms=4, label="windowed fetch (O(window))")
    ax1.set_yscale("log"); ax1.set_xscale("log", base=2)
    ax1.set_title("Access cost vs context (measured, Qwen2.5-1.5B)", fontsize=9)
    ax1.set_xlabel("context length (tokens)"); ax1.set_ylabel("ms")
    ax1.legend(fontsize=7); ax1.grid(alpha=.3, which="both")
    # right: VRAM vs context
    ax2.plot(SCALE_N, SCALE_FP16_MB, "-o", color=_BRICK, ms=4, label="fp16 KV")
    ax2.plot(SCALE_N, SCALE_CF_MB, "-o", color=_COPPER, ms=4, label="ChromoFold KV")
    ax2.set_xscale("log", base=2)
    ax2.set_title("KV VRAM vs context (one layer)", fontsize=9)
    ax2.set_xlabel("context length (tokens)"); ax2.set_ylabel("MB")
    ax2.legend(fontsize=7); ax2.grid(alpha=.3)
    for ax in (ax1, ax2):
        if SCALE_N[0] <= mark_n <= SCALE_N[-1]:
            ax.axvline(mark_n, color=_INK, ls="--", lw=1, alpha=.5)
    fig.tight_layout()
    return fig


def run(context_len, bits):
    m = measure(int(context_len), int(bits), device="cpu")
    return summary_markdown(m), _scale_fig(int(context_len))


_INTRO = """# 🧬 ChromoFold — a compressed KV cache that stays *searchable*

ChromoFold keeps an LLM's key-value cache **compressed *and* randomly addressable in memory**: a window fetch
decodes only what it reads, so per-token access cost does **not** grow with context length. It composes on top
of quantization (it's the lossless entropy + random-access layer, not the quantizer).

Pick a context length and KV bit-width below — the demo builds a real ChromoFold KV store **on CPU** and reports
the memory saving, the attention error, and the windowed-vs-decode-all access cost. The plots show the same
behavior **measured on a real model (Qwen2.5-1.5B)**: windowed access stays flat to 64K tokens while
decompress-all explodes (264× at 64K), at ~8.9× less VRAM than fp16.
"""

_HONEST = """---
**Honest scope:** quantization is the lossy lever; ChromoFold's entropy + index layer is *lossless over the
chosen quantization* and randomly addressable. This CPU demo uses synthetic real-shaped KV; the plots are the
measured on-GPU numbers. It does not beat `xz` on raw ratio — different job (it competes with a raw KV cache,
not an archiver). [Docs & integration →](https://github.com/cyberpunk042/warp-solar-system-shaders)
"""

with gr.Blocks(title="ChromoFold — compressed searchable KV cache", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_INTRO)
    with gr.Row():
        ctx = gr.Slider(256, 8192, value=2048, step=256, label="context length (tokens)")
        bits = gr.Radio([2, 4, 8], value=4, label="KV bit-width")
    btn = gr.Button("Measure", variant="primary")
    out_md = gr.Markdown()
    out_plot = gr.Plot()
    gr.Markdown(_HONEST)
    btn.click(run, [ctx, bits], [out_md, out_plot])
    demo.load(run, [ctx, bits], [out_md, out_plot])

if __name__ == "__main__":
    demo.launch()
