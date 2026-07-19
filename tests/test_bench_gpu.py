"""bench_gpu plumbing: metadata capture and repetition statistics (no heavy GPU work)."""
import numpy as np

from warp_compress.bench_gpu import _stats, metadata


def test_metadata_has_all_reproducibility_fields():
    md = metadata()
    for k in ("gpu", "driver", "cuda", "pcie_gen", "pcie_width", "sm_clock", "cpu", "ram",
              "warp", "python", "numpy", "os", "commit"):
        assert k in md
    assert md["warp"] and md["python"] and md["numpy"]        # always available even without nvidia-smi


def test_stats_reports_median_p5_p95_std():
    s = _stats(np.arange(1, 101))                             # 1..100
    assert abs(s["median"] - 50.5) < 1e-6
    assert s["p5"] < s["median"] < s["p95"]
    assert s["std"] > 0
