"""Command-line demo: ``python -m warp_compress <file>`` (or no arg for a built-in sample).

Compresses lossless, round-trips, and prints the chromosome shape + size against gzip; then
sweeps the lossy dial. This is a demonstration harness, not a production CLI.
"""

import gzip
import sys

from . import compress, decompress, describe


def _report(name: str, data: bytes) -> None:
    blob = compress(data, mode="lossless")
    assert decompress(blob) == data, "lossless round-trip failed"
    d = describe(blob)
    gz = len(gzip.compress(data, 9))
    print(f"[{name}] {len(data)} bytes")
    print(f"  lossless : warp {d['packed_bytes']}  (+zlib {d['packed_plus_zlib']})  vs gzip {gz}"
          f"   fold={d['fold']} nucleosomes={d['nucleosomes']} top={d['top_symbols']}")
    for q in (8, 16, 32):
        lb = compress(data, mode="lossy", q=q)
        back = decompress(lb)
        err = sum(abs(a - b) for a, b in zip(data, back)) / max(1, len(data))
        print(f"  lossy q={q:2d} : warp {len(lb):6d}   mean|err|={err:5.2f}  (<= {q // 2})")


def main(argv):
    if len(argv) > 1:
        with open(argv[1], "rb") as f:
            _report(argv[1], f.read())
    else:
        _report("sample", (b"the chromosome folds words layer by layer, like DNA. " * 120))


if __name__ == "__main__":
    main(sys.argv)
