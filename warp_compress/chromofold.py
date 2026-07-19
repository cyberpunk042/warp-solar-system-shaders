"""chromofold — the configurable pipeline that ties the pieces together.

ChromoFold is NOT a fixed format; it is a dial-set over a stack of succinct-data-structure stages
(serialize → dedup → transform → code → index). This module is the single place those dials live, plus
per-workload presets (sweet spots) and a dispatcher to the backends that already implement each path.

See ``docs/chromofold.md`` for the thesis (effective gain > ratio) and the honest metaphor/mechanism split.
Nothing here is load-bearing on the DNA geometry — that's visualization; this is the machinery.

    cfg = preset("rag")
    print(cfg.pipeline())          # human-readable stage chain
    backend, note = cfg.backend()  # which module implements it (or None => roadmap)

Run: python -m warp_compress.chromofold
"""
from __future__ import annotations

import dataclasses
from typing import Optional

# knob vocabularies (kept as plain strings so configs serialise trivially)
SERIALIZE = ("hilbert", "scan", "identity")
DEDUP = ("merge", "none")
TRANSFORM = ("bwt", "delta", "seed", "none")            # "seed" = N typed seed chromosomes (multi-prefix)
CODE = ("rrr", "rans", "none")
QUANTIZE = (None, "int8", "int4", "fp4", "nf4")
TARGET = ("gpu", "cpu")


@dataclasses.dataclass
class ChromoFoldConfig:
    """One point in the ChromoFold pipeline space. Every field is a tunable; presets pick sweet spots."""
    serialize: str = "hilbert"       # locality-preserving read order
    dedup: str = "merge"             # content-addressed block dedup (V unique + id stream)
    transform: str = "bwt"           # "bwt" => searchable self-index; "delta" => reference+sparse-diff tree
    code: str = "rrr"                # entropy coder over the id stream (keeps O(1) rank for rrr)
    quantize: Optional[str] = None   # lossy, opt-in; composes with the lossless stack (weights strata)
    target: str = "gpu"              # decode in VRAM ("gpu") vs cold ratio-first storage ("cpu")
    random_access: bool = True       # keep the index (partial unfold) vs drop it for archival ratio
    block: int = 5                   # dedup/merge block granularity
    sa_sample: int = 32              # suffix-array sampling: memory⇄locate-latency dial
    branch: int = 4                  # hierarchical fan-out for coarse→fine memory
    n_seeds: "int | None" = None     # transform="seed": number of typed seed chromosomes (None=auto, 1=global)
    seed_sig_len: int = 32           # transform="seed": prefix-signature length used to cluster to anchors

    def __post_init__(self):
        for name, vocab in [("serialize", SERIALIZE), ("dedup", DEDUP), ("transform", TRANSFORM),
                            ("code", CODE), ("quantize", QUANTIZE), ("target", TARGET)]:
            v = getattr(self, name)
            if v not in vocab:
                raise ValueError(f"{name}={v!r} not in {vocab}")

    def pipeline(self) -> str:
        stages = []
        if self.quantize:
            stages.append(f"quantize:{self.quantize}")
        if self.serialize != "identity":
            stages.append(f"serialize:{self.serialize}")
        if self.dedup != "none":
            stages.append(f"dedup:{self.dedup}(block={self.block})")
        if self.transform != "none":
            if self.transform == "bwt" and self.random_access:
                extra = f"(sa_sample={self.sa_sample})"
            elif self.transform == "seed":
                extra = f"(n_seeds={self.n_seeds or 'auto'})"
            else:
                extra = ""
            stages.append(f"transform:{self.transform}{extra}")
        if self.code != "none":
            stages.append(f"code:{self.code}")
        if self.random_access and self.transform not in ("delta", "seed"):
            stages.append("index:wavelet+sa")
        stages.append(f"decode@{self.target}")
        return "  →  ".join(stages)

    def backend(self):
        """Which built module implements this config today (or None => roadmap). Honest status, not a promise."""
        if self.transform == "bwt":
            return ("warp_compress.fm_index.FMIndex",
                    "compressed + addressable + searchable + predict_next (n-gram LM)")
        if self.transform == "delta":
            return ("warp_compress.super_chromosome.build_delta",
                    "reference/delta tree; O(depth) fetch; beats gzip across divergence")
        if self.transform == "seed":
            return ("warp_compress.multi_seed.MultiSeedStore",
                    "N typed seed chromosomes: cluster a mixed batch to prefix anchors, share each once")
        if self.transform == "none" and self.random_access:
            return ("warp_compress.token_chromosome.compress",
                    "positional O(1) addressing (Hilbert), no self-index")
        return (None, "roadmap: needs a stage not yet wired (e.g. rANS coder / GPU kernels)")


# per-workload sweet spots — see docs/chromofold.md §3/§4. `target=gpu` where partial unfold is the win.
PRESETS = {
    "prompt-cache":     ChromoFoldConfig(transform="bwt",   dedup="merge", code="rrr"),
    "conversation":     ChromoFoldConfig(transform="delta", dedup="none",  code="rans", serialize="identity"),
    "system-prompt":    ChromoFoldConfig(transform="none",  dedup="merge", code="rans"),
    "shared-prefix":    ChromoFoldConfig(transform="delta", dedup="merge", code="rans", serialize="identity"),
    "mixed-prompt-cache": ChromoFoldConfig(transform="seed", dedup="none", code="none", serialize="identity"),
    "rag":              ChromoFoldConfig(transform="bwt",   dedup="merge", code="rrr", branch=4),
    "dataset":          ChromoFoldConfig(transform="bwt",   dedup="merge", code="rans", target="cpu"),
    "spec-decode":      ChromoFoldConfig(transform="bwt",   dedup="none",  code="rrr"),
    "kv-sparse":        ChromoFoldConfig(transform="delta", dedup="none",  code="rrr", serialize="identity"),
    "moe-experts":      ChromoFoldConfig(transform="none",  dedup="merge", code="rans", quantize="int4"),
    "lora-library":     ChromoFoldConfig(transform="delta", dedup="none",  code="rans", serialize="identity"),
    "weights-dense":    ChromoFoldConfig(transform="none",  dedup="none",  code="rans", quantize="int4",
                                         serialize="identity", random_access=False),
    "archive":          ChromoFoldConfig(transform="bwt",   dedup="merge", code="rans", target="cpu",
                                         random_access=False),
}


def auto(data, intent: str | None = None) -> ChromoFoldConfig:
    """Auto-detect a config from a data sample (batch of sequences or one stream). Profiles the structure,
    builds the candidate, and keeps a compressing transform only if it beats raw. `intent`:
    serving|search|archival|dataset|None. For the rationale + achieved bytes/token, call `autotune.plan`."""
    from .autotune import plan
    return plan(data, intent=intent)[0]


def preset(name: str) -> ChromoFoldConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])          # a fresh, mutable copy to tune


def _demo():
    print("ChromoFold — configurable pipeline. Per-workload sweet spots (first roll; tune from here):\n")
    for name in PRESETS:
        cfg = PRESETS[name]
        mod, note = cfg.backend()
        tag = "✓" if mod else "…"
        print(f"  {tag} {name:14s} {cfg.pipeline()}")
        print(f"       {('→ ' + mod) if mod else '→ roadmap'}  —  {note}")
    print("\n  ✓ = a built backend implements this path today   … = roadmap (needs rANS / GPU kernels)")
    print("  See docs/chromofold.md. Ratio is one term; the win is GPU-resident, partial, searchable unfold.")


if __name__ == "__main__":
    _demo()
