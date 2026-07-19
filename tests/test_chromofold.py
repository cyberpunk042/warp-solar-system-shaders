"""ChromoFold config pipeline: presets resolve, pipelines render, backends map honestly, validation bites."""
import dataclasses

from warp_compress.chromofold import ChromoFoldConfig, PRESETS, preset


def test_every_preset_is_valid_and_describable():
    for name in PRESETS:
        cfg = preset(name)
        assert isinstance(cfg, ChromoFoldConfig)
        p = cfg.pipeline()
        assert "decode@" in p and "→" in p
        mod, note = cfg.backend()
        assert note                                        # always an honest note
        assert mod is None or mod.startswith("warp_compress.")


def test_preset_returns_a_fresh_mutable_copy():
    a = preset("rag"); b = preset("rag")
    a.sa_sample = 999
    assert b.sa_sample != 999                              # tuning one copy must not touch the preset/others


def test_bwt_and_delta_backends_resolve_to_built_modules():
    assert preset("prompt-cache").backend()[0] == "warp_compress.fm_index.FMIndex"
    assert preset("lora-library").backend()[0] == "warp_compress.super_chromosome.build_delta"


def test_dense_weights_is_honestly_roadmap():
    mod, note = preset("weights-dense").backend()
    assert mod is None and "roadmap" in note.lower()       # don't pretend a path we haven't wired


def test_invalid_knob_is_rejected():
    try:
        ChromoFoldConfig(transform="huffman")             # not in the vocabulary
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pipeline_reflects_quantize_and_random_access_flags():
    cfg = dataclasses.replace(preset("moe-experts"))
    assert "quantize:int4" in cfg.pipeline()
    off = ChromoFoldConfig(transform="bwt", random_access=False)
    assert "index:" not in off.pipeline()                 # dropping the index shows in the pipeline
