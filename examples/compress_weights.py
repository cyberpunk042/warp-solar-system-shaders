"""Compress a weight tensor with ChromoFold, keep random access, save a portable .cfold — offline, no torch.

    python examples/compress_weights.py
"""
import numpy as np

import chromofold as cf


def main():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((2048, 2048)) / 45).astype(np.float32)   # a Gaussian weight layer

    art = cf.compress(W)                       # quantize + entropy-code, GPU-resident, addressable
    recon = art.decode()                       # dequantized tensor (lossless over the quantization)
    idx = rng.integers(0, W.size, 4)
    fetched = art.fetch(idx)                   # O(1) random access on the GPU — no full decode

    print(f"weights {W.shape} ({W.size:,})  ->  {art.size_bytes()/1e6:.2f} MB "
          f"({art.size_bytes()*8/W.size:.2f} b/weight)")
    print(f"random-access fetch matches reconstruct: {np.allclose(fetched, recon.ravel()[idx], atol=1e-5)}")

    blob = art.save()                          # a portable, versioned container
    reloaded = cf.Artifact.load(blob)
    print(f".cfold blob {len(blob)/1e6:.2f} MB  ->  reload round-trips exact: "
          f"{np.array_equal(reloaded.decode(), recon)}")


if __name__ == "__main__":
    main()
