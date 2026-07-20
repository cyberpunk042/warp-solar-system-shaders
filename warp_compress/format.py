"""format — the ChromoFold container: a self-describing, versioned on-disk schema for a compressed object.

A ChromoFold object (an entropy-coded weight tensor, an RRR self-index, a delta cluster, …) is a small set of
scalar parameters plus a handful of typed binary arrays. This is the container that serialises exactly that,
so a compressed artifact is a single portable blob you can write to disk, ship, and reload — the format spec
lives in ``docs/chromofold_format.md``.

Layout (all integers little-endian):

    0      8 bytes   MAGIC  = b"CHROMOF\\x01"     (7 ASCII + 1 format-version byte)
    8      uint32    HEADER_LEN
    12     HEADER_LEN bytes  HEADER  (UTF-8 JSON: format, version, object, config, params, sections[])
    …      SECTION DATA  (each array's raw little-endian bytes, concatenated in `sections` order)

The header's ``sections`` list gives every array's name / dtype / shape / nbytes, so the payload is sliced back
into named numpy arrays with no ambiguity. ``config`` records the pipeline that produced it (the
``ChromoFoldConfig`` view); ``params`` holds the object's scalars (bits, shape, scales, …).

    pack(object_type, config, params, arrays) -> bytes
    unpack(bytes) -> (header_dict, {name: np.ndarray})

Run: python -m warp_compress.format
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np

MAGIC = b"CHROMOF\x01"          # 7 ASCII bytes + container-format version byte (0x01)
VERSION = [1, 0]               # [major, minor] of the schema


def pack(object_type: str, config: dict, params: dict, arrays: dict, compress=()) -> bytes:
    """Serialise a ChromoFold object into one container blob. `arrays` maps section name -> numpy array.

    Sections named in `compress` are stored **delta+zlib** — the RRR index metadata (superblocks, word bases)
    is monotone, so this shrinks the blob toward a stream compressor's ratio WITHOUT touching the resident
    (int32, O(1)-rank) form: they decode back to the same arrays on load. The high-entropy payload
    (RRR/Huffman bitstreams) is left raw. Losslessly reversible (delta→cumsum)."""
    compress = set(compress)
    sections, blobs = [], []
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        if name in compress and np.issubdtype(a.dtype, np.integer):
            delta = np.diff(a, axis=-1, prepend=0).astype(a.dtype)   # keep dtype (prepend=0 else upcasts)
            payload = zlib.compress(np.ascontiguousarray(delta).tobytes(), 9)
            sections.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape),
                             "nbytes": len(payload), "codec": "delta+zlib"})
            blobs.append(payload)
        else:
            sections.append({"name": name, "dtype": a.dtype.str, "shape": list(a.shape), "nbytes": int(a.nbytes)})
            blobs.append(a.tobytes())
    header = {"format": "chromofold", "version": VERSION, "object": object_type,
              "config": config, "params": params, "sections": sections}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(hb)) + hb + b"".join(blobs)


def unpack(data: bytes):
    """Parse a container blob into (header_dict, {name: np.ndarray}). Validates magic + section lengths."""
    if data[:8] != MAGIC:
        raise ValueError("not a ChromoFold container (bad magic)")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12:12 + hlen].decode("utf-8"))
    if header.get("format") != "chromofold":
        raise ValueError("header is not a chromofold header")
    off = 12 + hlen
    arrays = {}
    for s in header["sections"]:
        n = int(s["nbytes"])
        raw = data[off:off + n]
        off += n
        dt = np.dtype(s["dtype"])
        if s.get("codec") == "delta+zlib":
            d = np.frombuffer(zlib.decompress(raw), dtype=dt).reshape(s["shape"])
            arrays[s["name"]] = np.cumsum(d, axis=-1).astype(dt)
        else:
            arrays[s["name"]] = np.frombuffer(raw, dtype=dt).reshape(s["shape"])
    return header, arrays


def summary(data: bytes) -> str:
    """One-line human summary of a container without materialising the arrays."""
    header, _ = unpack(data)
    payload = sum(int(s["nbytes"]) for s in header["sections"])
    return (f"chromofold v{'.'.join(map(str, header['version']))}  object={header['object']}  "
            f"pipeline={header.get('config', {})}  {len(header['sections'])} sections  "
            f"{payload/1e3:.1f} KB payload / {len(data)/1e3:.1f} KB total")


def _demo():
    cfg = {"quantize": "int4", "transform": "none", "code": "huffman", "group_size": 128}
    params = {"bits": 4, "shape": [64, 32], "zero": 7}
    arrays = {"values": np.arange(2048, dtype=np.uint32) % 15,
              "scales": (np.random.default_rng(0).random(16) * 0.01).astype(np.float16)}
    blob = pack("weight_store", cfg, params, arrays)
    header, back = unpack(blob)
    ok = all(np.array_equal(back[k], np.ascontiguousarray(v)) for k, v in arrays.items())
    print("packed:", summary(blob))
    print("round-trip arrays identical:", ok, "  header params:", header["params"])
    print("=> a compressed ChromoFold object is one portable, self-describing, versioned blob. See "
          "docs/chromofold_format.md for the full protocol.")


if __name__ == "__main__":
    _demo()
