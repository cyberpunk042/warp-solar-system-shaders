"""chromofold command-line interface.

    chromofold info                 # version, offline guarantee, GPU availability
    chromofold selftest             # compress -> reconstruct round-trip (validate an install / air-gap deploy)
    chromofold inspect FILE.cfold   # header of a portable ChromoFold container

Entry point (installed): `chromofold ...`. Runs offline; no network I/O.
"""
from __future__ import annotations

import argparse
import sys


def _info(_args) -> int:
    import chromofold as cf
    print(f"chromofold {cf.__version__}")
    print("  network I/O: none (core is offline; only the optional transformers model loader can reach a")
    print("              network, and only when you fetch weights — gate with HF_HUB_OFFLINE=1)")
    try:
        import warp as wp
        ndev = wp.get_cuda_device_count()
        print(f"  cuda devices (via warp): {ndev}" + (f"  [device 0 ready]" if ndev else "  [CPU only]"))
    except Exception as e:  # pragma: no cover
        print(f"  warp: not importable ({e.__class__.__name__})")
    try:
        import torch  # noqa: F401
        print("  torch: available (transformers KV cache usable)")
    except Exception:
        print("  torch: not installed (install `chromofold[torch]` for the transformers KV cache)")
    return 0


def _selftest(_args) -> int:
    """A quick end-to-end round-trip — the check to run right after an install / air-gap deploy."""
    import numpy as np
    import warp as wp
    import chromofold as cf
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)

    W = (rng.standard_normal((512, 512)) / 22).astype(np.float32)   # a weight tensor
    art = cf.compress(W)
    recon = art.decode()
    idx = rng.integers(0, W.size, 1000)
    ra_ok = np.allclose(art.fetch(idx), recon.ravel()[idx], atol=1e-5)
    blob = art.save()
    reloaded = cf.Artifact.load(blob) if hasattr(cf.Artifact, "load") else None
    save_ok = reloaded is not None and np.array_equal(reloaded.decode(), recon)

    print(f"chromofold selftest  (device={dev})")
    print(f"  compress 512x512 weights -> {art.size_bytes()/1024:.1f} KB  ({art.size_bytes()*8/W.size:.2f} b/weight)")
    print(f"  random access fetch == reconstruct : {'OK' if ra_ok else 'FAIL'}")
    print(f"  .cfold save/load round-trip        : {'OK' if save_ok else 'FAIL'}")
    ok = ra_ok and save_ok
    print("  =>", "PASS — the install works, offline." if ok else "FAIL")
    return 0 if ok else 1


def _inspect(args) -> int:
    from warp_compress import format as fmt
    with open(args.file, "rb") as f:
        header, arrays = fmt.unpack(f.read())
    print(f"chromofold container: {args.file}")
    print(f"  object : {header.get('object')}")
    print(f"  format : {header.get('format')} v{header.get('version')}")
    print(f"  config : {header.get('config')}")
    total = sum(int(a.nbytes) for a in arrays.values())
    print(f"  arrays : {len(arrays)} sections, {total/1024:.1f} KB payload")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chromofold", description="GPU-resident compression for LLM data.")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("info", help="version, offline guarantee, GPU availability").set_defaults(fn=_info)
    sub.add_parser("selftest", help="compress->reconstruct round-trip (validate an install)").set_defaults(fn=_selftest)
    ins = sub.add_parser("inspect", help="show a .cfold container header")
    ins.add_argument("file")
    ins.set_defaults(fn=_inspect)
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
