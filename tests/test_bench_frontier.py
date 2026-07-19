"""bench_frontier plumbing: zstd sizing (the corpora path needs a tokenizer, exercised by the demo, not here)."""
import numpy as np

from warp_compress.bench_frontier import _zstd_bytes


def test_zstd_bytes_compresses_redundant_data():
    redundant = (np.arange(64, dtype=np.uint16).tobytes()) * 500   # very repetitive
    assert _zstd_bytes(redundant) < len(redundant)                # zstd shrinks it

    import numpy as _np
    rnd = _np.random.default_rng(0).integers(0, 65536, 20000).astype(_np.uint16).tobytes()
    assert _zstd_bytes(rnd) > _zstd_bytes(redundant)              # noise compresses far worse than structure
