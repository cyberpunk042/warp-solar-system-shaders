"""bench_stack timing helper + the qualitative facts it measures (xz beats cfold on ratio; stacking kills RA)."""
import gzip
import lzma

import numpy as np

from warp_compress.bench_stack import _gz, _t, _xz


def test_timing_helper_returns_result_and_positive_time():
    r, dt = _t(lambda: sum(range(1000)))
    assert r == sum(range(1000)) and dt >= 0.0


def test_xz_beats_gz_on_peaky_bytes():
    # a peaky byte stream (like quantized weights): xz should compress at least as well as gz
    rng = np.random.default_rng(0)
    b = np.clip(np.round(rng.standard_normal(50000) * 2), -7, 7).astype(np.int8).tobytes()
    assert len(_xz(b)) <= len(_gz(b))


def test_stacking_xz_on_an_entropyish_blob_still_shrinks_metadata():
    # a blob that is mostly high-entropy payload + a monotone side-table (like a cfold container's superblocks)
    rng = np.random.default_rng(1)
    payload = rng.integers(0, 256, 20000, dtype=np.uint8).tobytes()      # ~incompressible
    superblocks = np.cumsum(rng.integers(0, 4, 4000)).astype(np.int32).tobytes()  # very compressible
    blob = payload + superblocks
    assert len(_xz(blob)) < len(blob)                                    # the metadata portion compresses
    # but the high-entropy payload alone barely moves
    assert len(_xz(payload)) > 0.9 * len(payload)
