"""bench_dataset_dedup — near-duplicate tokenized dataset: where the reference/delta tree genuinely earns it.

The second workload flagged as compelling: a tokenized dataset with exact + near duplicates (web-scraped
corpora are full of them). Content-aware dedup (`dedup.DedupStore`) stores exact dups as a reference, near-dups
as a sparse delta vs the true nearest document, uniques once — and, unlike gzip/zstd, keeps **random access**:
fetch any document's tokens on the GPU without decompressing the dataset.

Honest scope, stated up front: gzip/zstd win pure RATIO (they also compress the token entropy of the unique
documents that DedupStore leaves raw). DedupStore beats raw and preserves O(1) random access — that is the
niche, not archival ratio. (Note the *positional* delta needs aligned dups; big insert/delete shifts break it —
LZ/FM-index handle those.) Requires transformers + zstandard. Run: python -m warp_compress.bench_dataset_dedup
"""
from __future__ import annotations

import glob
import gzip
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _dataset(tok, L=256, seed=0):
    """A tokenized dataset of fixed-length documents: unique + exact dups + aligned near-dups + unaligned."""
    rng = np.random.default_rng(seed)
    ids = tok("".join(open(f).read() for f in sorted(glob.glob("warp_compress/*.py")))).input_ids
    ids = np.asarray(ids, np.int64)
    uniq = [ids[i:i + L] for i in range(0, len(ids) - L, L)][:120]     # base documents

    docs, kind = [], []
    for d in uniq:
        docs.append(d); kind.append("unique")
        if rng.random() < 0.5:                                         # exact duplicate
            docs.append(d.copy()); kind.append("exact-dup")
        if rng.random() < 0.5:                                         # aligned near-dup (same length, few edits)
            e = d.copy(); f = rng.integers(0, L, 6); e[f] = rng.integers(0, 50257, 6)
            docs.append(e); kind.append("near-dup")
    order = rng.permutation(len(docs))
    return [docs[i] for i in order], [kind[i] for i in order]


def main():
    from transformers import AutoTokenizer
    import zstandard
    import warp as wp

    from .dedup import DedupStore
    from collections import Counter

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")

    print("=" * 96)
    print("Near-duplicate tokenized dataset — content-aware dedup vs gzip/zstd, with random access")
    print("=" * 96)
    docs, kinds = _dataset(tok, seed=0)
    total = sum(len(d) for d in docs)
    b16 = np.concatenate(docs).astype(np.uint16).tobytes()

    store = DedupStore(docs, device=dev)
    assert all(np.array_equal(store.decode(k), docs[k]) for k in range(0, len(docs), 17)), "dedup lossless"
    dedup_bpt = store.size_bytes() / total
    gz_bpt = len(gzip.compress(b16, 9)) / total
    zs_bpt = len(zstandard.ZstdCompressor(level=19).compress(b16)) / total

    zc = zstandard.ZstdCompressor(level=19).compress(b16)
    zd = zstandard.ZstdDecompressor()
    for _ in range(3):
        store.decode(3)
    t0 = time.perf_counter()
    for _ in range(50):
        store.decode(3)
    cf_us = (time.perf_counter() - t0) / 50 * 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        zd.decompress(zc)
    zs_us = (time.perf_counter() - t0) / 20 * 1e6

    print(f"\n  {len(docs)} docs × 256 tok = {total:,} tokens   composition: {dict(Counter(kinds))}")
    print(f"  detected: {store.n_bases} unique bases, {store.n_near} near-dups, {store.n_exact} exact dups")
    print(f"  bytes/token   raw u16 2.00   DedupStore {dedup_bpt:.2f}   gzip {gz_bpt:.2f}   zstd19 {zs_bpt:.2f}")
    print(f"  fetch 1 doc   DedupStore {cf_us:.1f} µs (GPU, O(1), no full decode)   vs zstd decompress-all "
          f"{zs_us:.1f} µs")
    print("\n=> honest read: DedupStore beats RAW ({:.2f}×) by collapsing exact dups to a ref and near-dups to a "
          "sparse delta, and keeps O(1) GPU random access. gzip/zstd win pure RATIO because they also compress\n"
          "   the token entropy of the UNIQUE documents (which DedupStore leaves raw) — but they have no random "
          "access: any document costs a full-stream decompress. Dedup+random-access is the niche, not ratio."
          .format(2.0 / dedup_bpt))


if __name__ == "__main__":
    main()
