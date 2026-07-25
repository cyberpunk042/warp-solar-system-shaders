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
import mmap
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


def pack_atom(object_type: str, atom_bytes: bytes, config: dict | None = None,
              params: dict | None = None) -> bytes:
    """Embed a self-describing codec atom (e.g. ``grouped_delta.to_bytes`` / ``super_elastic.to_bytes``) as a
    single raw section in a ``.cfold`` container.

    The container carries the atom byte-exact; the atom's own header (``GDLT1`` / ``SELA1`` magic + varint
    fields) self-describes its interior — so the two serialisation layers compose without either duplicating
    the other's schema. ``params`` records the atom's codec magic so ``summary`` can report it without decoding."""
    arr = np.frombuffer(atom_bytes, dtype=np.uint8)
    p = dict(params or {})
    p.setdefault("atom_magic", atom_bytes[:5].decode("ascii", "replace"))
    p.setdefault("atom_nbytes", int(arr.nbytes))
    return pack(object_type, config or {}, p, {"atom": arr})


def unpack_atom(data: bytes):
    """Inverse of :func:`pack_atom`: return ``(object_type, atom_bytes)`` — feed ``atom_bytes`` straight to the
    codec's ``from_bytes``. Raises if the container has no ``atom`` section (not an atom container)."""
    header, arrays = unpack(data)
    if "atom" not in arrays:
        raise ValueError("container has no 'atom' section (not an atom container)")
    return header["object"], arrays["atom"].tobytes()


_ATOM_PREFIX = "atom:"          # section-name prefix for a named atom in a multi-atom store


def pack_atoms(object_type: str, atoms: dict, config: dict | None = None,
               params: dict | None = None) -> bytes:
    """Pack a **named set** of codec atoms into one ``.cfold`` container — the substrate for a real
    ``.cfold``-embedded weight / adapter store that holds many folded layers.

    Each ``name -> atom_bytes`` becomes a raw ``uint8`` section ``atom:<name>``; a ``manifest`` in the header
    records each atom's codec magic + byte length + section order, so :func:`read_atom` can pull one atom by
    name via offset arithmetic alone — no decode of the others (the random-access payoff)."""
    arrays, manifest = {}, {}
    off = 0
    for name, ab in atoms.items():
        if _ATOM_PREFIX + name in arrays:
            raise ValueError(f"duplicate atom name {name!r}")
        arrays[_ATOM_PREFIX + name] = np.frombuffer(ab, dtype=np.uint8)
        manifest[name] = {"magic": ab[:5].decode("ascii", "replace"), "nbytes": len(ab), "order": len(manifest)}
        off += len(ab)
    p = dict(params or {})
    p["atom_manifest"] = manifest
    p["atom_count"] = len(atoms)
    return pack(object_type, config or {}, p, arrays)


def unpack_atoms(data: bytes):
    """Inverse of :func:`pack_atoms`: return ``(object_type, {name: atom_bytes})`` for every atom in the store."""
    header, arrays = unpack(data)
    out = {k[len(_ATOM_PREFIX):]: v.tobytes() for k, v in arrays.items() if k.startswith(_ATOM_PREFIX)}
    if not out:
        raise ValueError("container holds no atoms (not a multi-atom store)")
    return header["object"], out


def read_atom(data: bytes, name: str) -> bytes:
    """Random-access read of a single atom by name from a multi-atom store — parses only the header, then slices
    that one section's bytes. Cost is O(sections) offset arithmetic, independent of the other atoms' sizes: you
    do not pay to materialise a 100-layer store to pull one layer."""
    if data[:8] != MAGIC:
        raise ValueError("not a ChromoFold container (bad magic)")
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12:12 + hlen].decode("utf-8"))
    target = _ATOM_PREFIX + name
    off = 12 + hlen
    for s in header["sections"]:
        n = int(s["nbytes"])
        if s["name"] == target:
            if s.get("codec"):
                raise ValueError(f"atom section {name!r} unexpectedly codec-wrapped ({s['codec']})")
            return data[off:off + n]
        off += n
    raise KeyError(f"no atom named {name!r} in this container")


def write_cfold(path, data: bytes) -> int:
    """Write a container blob to a ``.cfold`` file. Returns bytes written — the container's on-disk form
    (the module's whole promise: a compressed object is one portable, self-describing file you can ship)."""
    with open(path, "wb") as f:
        return f.write(data)


def read_cfold(path) -> bytes:
    """Read a whole ``.cfold`` file back into a container blob (validates magic)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path} is not a ChromoFold container (bad magic)")
    return data


def _header_from_prefix(read_prefix) -> dict:
    """Parse the container header given a callable ``read_prefix(a, b) -> bytes`` over ``[a:b)``."""
    if read_prefix(0, 8) != MAGIC:
        raise ValueError("not a ChromoFold container (bad magic)")
    (hlen,) = struct.unpack("<I", read_prefix(8, 12))
    return json.loads(read_prefix(12, 12 + hlen).decode("utf-8")), 12 + hlen


def read_atom_file(path, name: str) -> bytes:
    """Random-access read of one atom from an on-disk ``.cfold`` store via ``mmap`` — the deployable form of
    :func:`read_atom`. Parses the header, then slices that one section straight out of the memory-mapped file:
    only the header pages + the target atom's pages are faulted in, so you pull one layer from a multi-GB
    store **without loading the file**. Cost is independent of the other atoms' sizes."""
    target = _ATOM_PREFIX + name
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header, off = _header_from_prefix(lambda a, b: mm[a:b])
            for s in header["sections"]:
                n = int(s["nbytes"])
                if s["name"] == target:
                    if s.get("codec"):
                        raise ValueError(f"atom section {name!r} unexpectedly codec-wrapped ({s['codec']})")
                    return bytes(mm[off:off + n])           # copy out only this atom's pages
                off += n
    raise KeyError(f"no atom named {name!r} in {path}")


def atom_names_file(path) -> list:
    """List the atom names in an on-disk store from its header alone — no payload read (mmap, header pages only)."""
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header, _ = _header_from_prefix(lambda a, b: mm[a:b])
    return [s["name"][len(_ATOM_PREFIX):] for s in header["sections"] if s["name"].startswith(_ATOM_PREFIX)]


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

    # atom embedding: a folded codec atom (grouped_delta) rides inside the container byte-exact
    from warp_compress import grouped_delta as gd
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, size=(16, 16), dtype=np.int64)
    group = np.stack([base + rng.integers(-2, 3, size=base.shape) for _ in range(8)])
    atom = gd.to_bytes(gd.compress(group, mode="centroid"))
    ablob = pack_atom("grouped_delta", atom)
    obj_type, atom_back = unpack_atom(ablob)
    recon = gd.decompress(gd.from_bytes(atom_back))
    print("atom container:", summary(ablob))
    print(f"  object={obj_type}  atom byte-exact through container: {atom_back == atom}  "
          f"lossless reconstruct: {np.array_equal(recon, group)}")

    # multi-atom store: many folded layers in one container, one pulled by name without decoding the rest
    layers = {f"layer.{i}": gd.to_bytes(gd.compress(
        np.stack([base + rng.integers(-2, 3, size=base.shape) for _ in range(8)]), mode="centroid"))
        for i in range(6)}
    store = pack_atoms("weight_store", layers)
    one = read_atom(store, "layer.3")                       # random access: header-only slice
    _, all_back = unpack_atoms(store)
    print("multi-atom store:", summary(store))
    print(f"  {len(layers)} layers  random-access read_atom('layer.3') byte-exact: {one == layers['layer.3']}  "
          f"full unpack all byte-exact: {all(all_back[k] == v for k, v in layers.items())}")

    # on-disk: write the store to a .cfold file, pull one layer via mmap without loading the file
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "chromofold_demo.cfold")
    write_cfold(path, store)
    disk_one = read_atom_file(path, "layer.3")
    print(f"  on-disk .cfold ({os.path.getsize(path)/1e3:.1f} KB): mmap read_atom_file('layer.3') "
          f"byte-exact: {disk_one == layers['layer.3']}  names={atom_names_file(path)[:3]}…")
    os.remove(path)


if __name__ == "__main__":
    _demo()
