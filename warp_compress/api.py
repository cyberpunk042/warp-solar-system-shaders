"""api — one front door to ChromoFold. `compress(data, intent)` -> an Artifact you can decode / fetch / save.

The package has many stores (weights, token self-index, seed/delta/dedup clusters); this is the single
high-level entry point that profiles the data (via `autotune`), dispatches to the right backend, and hands
back a uniform `Artifact`. See `docs/chromofold_positioning.md` for *when* to use it.

    art = compress(W)                       # a 2-D float weight tensor  -> quantized + entropy-coded
    art = compress(tokens)                  # a 1-D int token stream      -> addressable RRR self-index
    art = compress([s1, s2, ...])           # a batch of sequences        -> seed / delta / dedup (auto)
    art.decode()          # reconstruct the whole thing
    art.fetch(idx)        # random access (weights / tokens)
    art.size_bytes()      # compressed footprint
    blob = art.save();  Artifact.load(blob) # portable .cfold container (weights & tokens)

Run: python -m warp_compress.api
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import format as fmt


@dataclasses.dataclass
class Artifact:
    kind: str                 # 'weights' | 'tokens' | 'seed' | 'delta' | 'dedup'
    store: object
    meta: dict

    def size_bytes(self) -> int:
        s = self.store
        for attr in ("size_bytes", "index_bytes"):
            if hasattr(s, attr):
                return getattr(s, attr)()
        return 0

    def decode(self):
        s = self.store
        if self.kind == "weights":
            return s.reconstruct()
        if self.kind == "tokens":
            return s.access(np.arange(self.meta["n"], dtype=np.int64))
        k = self.meta.get("k", 0)
        if self.kind == "seed":
            return [s.recover_request(i) for i in range(k)]
        if self.kind == "dedup":
            return [s.decode(i) for i in range(k)]
        if self.kind == "delta":
            return [s.decode_leaf(i) for i in range(k)]
        raise ValueError(self.kind)

    def fetch(self, idx):
        if self.kind == "weights":
            return self.store.fetch(idx)
        if self.kind == "tokens":
            return self.store.access(np.asarray(idx, np.int64))
        raise NotImplementedError(f"fetch not defined for kind={self.kind} (use decode / recover)")

    def summary(self) -> str:
        return f"ChromoFold Artifact [{self.kind}]  {self.size_bytes()/1e3:.1f} KB  meta={self.meta}"

    # --- serialisation (weights & tokens; clusters compose from component blobs — a future extension) ---
    def save(self) -> bytes:
        if self.kind == "weights":
            return self.store.save()
        if self.kind == "tokens":
            p, a = self.store.to_host()
            cfg = {"transform": "wavelet", "code": "huffman"}
            mono = {"rank_a", "off_a", "cls_a", "cbase", "obase", "offbase"} & set(a)   # two-level anchors
            return fmt.pack("token_index", cfg, p, a, compress=mono)
        raise NotImplementedError(f"save() supports 'weights' and 'tokens'; kind={self.kind} is decode-only")

    @classmethod
    def load(cls, data: bytes, device: str = "cuda:0") -> "Artifact":
        header, arrays = fmt.unpack(data)
        obj = header["object"]
        if obj == "weight_store":
            from .weight_store import QuantizedWeightStore
            st = QuantizedWeightStore.load(data, device)
            return cls("weights", st, {"shape": list(st.shape), "bits": st.bits})
        if obj == "token_index":
            from .gpu_rrr_huffman import RRRWaveletGPUHuff
            st = RRRWaveletGPUHuff.from_host(header["params"], arrays, device)
            return cls("tokens", st, {"n": st.n, "bits": st.bits})
        raise ValueError(f"unknown container object {obj!r}")


def compress(data, intent: str | None = None, bits: int = 8, huffman: bool = True,
             device: str = "cuda:0") -> Artifact:
    """Profile `data` and compress it with the right ChromoFold backend. Returns an Artifact."""
    # a batch of sequences -> cluster (auto-detect seed vs delta vs dedup)
    if isinstance(data, (list, tuple)):
        from .autotune import analyze
        seqs = [np.asarray(s, np.int64) for s in data]
        p = analyze(seqs)
        if p.prefix_share > 2.0:
            from .multi_seed import MultiSeedStore
            st = MultiSeedStore(seqs, device=device)
            return Artifact("seed", st, {"k": len(seqs), "n_seeds": st.n_seeds})
        if p.near_dup_div == p.near_dup_div and p.near_dup_div < 0.15:
            from .super_chromosome import build_delta
            from .gpu_delta import GPUDeltaCluster
            st = GPUDeltaCluster(seqs, device=device)
            return Artifact("delta", st, {"k": len(seqs)})
        from .dedup import DedupStore                          # mostly-unique cluster -> content dedup
        st = DedupStore(seqs, device=device)
        return Artifact("dedup", st, {"k": len(seqs)})

    arr = np.asarray(data)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):   # a weight tensor
        from .weight_store import QuantizedWeightStore
        st = QuantizedWeightStore(arr.astype(np.float32), bits=bits, huffman=huffman, device=device)
        return Artifact("weights", st, {"shape": list(arr.shape), "bits": bits})

    seq = arr.astype(np.int64).ravel()                        # a token / id stream -> addressable RRR self-index
    from .gpu_rrr_huffman import RRRWaveletGPUHuff
    st = RRRWaveletGPUHuff(seq, device=device)
    return Artifact("tokens", st, {"n": int(seq.shape[0]), "V": int(seq.max()) + 1})


def _demo():
    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)

    print("ChromoFold — one API, right backend per data shape\n")

    W = (rng.standard_normal((512, 256)) / 16).astype(np.float32)
    aw = compress(W, bits=4, device=dev)
    aw2 = Artifact.load(aw.save(), device=dev)
    print(f"  weights  {aw.summary()}   save→load exact: {np.array_equal(aw.decode(), aw2.decode())}")

    toks = rng.integers(0, 4000, 40000).astype(np.int64)
    at = compress(toks, device=dev)
    at2 = Artifact.load(at.save(), device=dev)
    print(f"  tokens   {at.summary()}   fetch==decode: {np.array_equal(at.fetch([1, 99, 9999]), at.decode()[[1,99,9999]])}"
          f"   save→load exact: {np.array_equal(at.decode(), at2.decode())}")

    base = rng.integers(0, 4, 300)
    dups = [(lambda s: (s.__setitem__(rng.integers(0, 300, 4), rng.integers(0, 4, 4)), s)[1])(base.copy())
            for _ in range(24)]
    ad = compress(dups, device=dev)
    print(f"  cluster  {ad.summary()}   decode round-trips: {all(np.array_equal(a, b) for a, b in zip(ad.decode(), dups))}")

    print("\n=> compress(data, intent) profiles the data and picks the backend; the Artifact decodes / fetches / "
          "saves. Weights & tokens serialise to a portable .cfold container; clusters decode in place. One door.")


if __name__ == "__main__":
    _demo()
