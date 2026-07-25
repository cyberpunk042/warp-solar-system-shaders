"""test_format_atom — the folded codec atoms ride inside the real .cfold container, byte-exact.

`grouped_delta` / `super_elastic` gained standalone `to_bytes`/`from_bytes` (self-describing GDLT1/SELA1 blobs).
This proves the deployability closure: `format.pack_atom` embeds such an atom as a raw section in a ChromoFold
container and `format.unpack_atom` returns it byte-identical, so the codec's `from_bytes` reconstructs the
original bit-exact. The two serialisation layers (container + atom) compose without either owning the other's
schema.

Run: .venv/bin/python -m pytest tests/test_format_atom.py -q
     .venv/bin/python tests/test_format_atom.py
"""
import numpy as np

from warp_compress import format as fmt
from warp_compress import grouped_delta as gd
from warp_compress import super_elastic as se


def _correlated_group(n=8, shape=(16, 16), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=shape, dtype=np.int64)
    return np.stack([base + rng.integers(-2, 3, size=shape) for _ in range(n)])


def test_grouped_delta_atom_roundtrips_through_container():
    group = _correlated_group(seed=1)
    for label, kw in (("lossless", dict(rank=None)),
                      ("float32-factors", dict(rank=4)),
                      ("int8-factors", dict(rank=4, quant_factors=True))):
        atom = gd.to_bytes(gd.compress(group, mode="centroid", **kw))
        blob = fmt.pack_atom("grouped_delta", atom, config={"codec": "grouped_delta", "variant": label})
        assert blob[:8] == fmt.MAGIC                              # a real ChromoFold container
        obj_type, atom_back = fmt.unpack_atom(blob)
        assert obj_type == "grouped_delta"
        assert atom_back == atom, f"{label} atom not byte-exact through container"
        recon = gd.decompress(gd.from_bytes(atom_back))
        if kw.get("rank") is None:                               # lossless variant reconstructs bit-exact
            assert np.array_equal(recon, group), "lossless atom must reconstruct exactly through container"


def test_super_elastic_atom_roundtrips_through_container():
    group = _correlated_group(seed=2)
    # lossless stack (final exact residual) must reconstruct bit-exact through the container
    atom = se.to_bytes(se.compress(group, layers=3, rank=4, mode="centroid", final_residual=True))
    blob = fmt.pack_atom("super_elastic", atom, config={"codec": "super_elastic"})
    assert blob[:8] == fmt.MAGIC
    obj_type, atom_back = fmt.unpack_atom(blob)
    assert obj_type == "super_elastic"
    assert atom_back == atom
    recon = se.decompress(se.from_bytes(atom_back))
    assert np.array_equal(recon, group)


def test_atom_params_recorded_in_header():
    atom = gd.to_bytes(gd.compress(_correlated_group(seed=3), mode="centroid"))
    blob = fmt.pack_atom("grouped_delta", atom)
    header, arrays = fmt.unpack(blob)
    assert header["params"]["atom_magic"] == "GDLT1"             # codec magic surfaced without decoding
    assert header["params"]["atom_nbytes"] == len(atom)
    assert arrays["atom"].dtype == np.uint8 and arrays["atom"].nbytes == len(atom)


def test_multi_atom_store_roundtrips_and_random_access():
    # a store of many folded layers (grouped_delta) + a super_elastic adapter, all in one container
    groups = {f"layer.{i}": _correlated_group(seed=10 + i) for i in range(6)}
    atoms = {name: gd.to_bytes(gd.compress(g, mode="centroid")) for name, g in groups.items()}
    atoms["adapter"] = se.to_bytes(se.compress(_correlated_group(seed=99), layers=3, rank=4,
                                               mode="centroid", final_residual=True))
    store = fmt.pack_atoms("weight_store", atoms, config={"kind": "folded-layer-store"})
    assert store[:8] == fmt.MAGIC

    # random access: pull one layer by name, header-only slice, byte-exact vs the others
    for name in ("layer.0", "layer.3", "layer.5", "adapter"):
        assert fmt.read_atom(store, name) == atoms[name], f"random-access read_atom({name}) not byte-exact"

    # full unpack returns every atom byte-exact; each reconstructs through its own codec
    obj_type, back = fmt.unpack_atoms(store)
    assert obj_type == "weight_store"
    assert set(back) == set(atoms) and all(back[k] == atoms[k] for k in atoms)
    for name, g in groups.items():
        assert np.array_equal(gd.decompress(gd.from_bytes(back[name])), g)
    assert np.array_equal(se.decompress(se.from_bytes(back["adapter"])), _correlated_group(seed=99))

    # manifest records each atom's codec magic + order without decoding
    header, _ = fmt.unpack(store)
    man = header["params"]["atom_manifest"]
    assert header["params"]["atom_count"] == len(atoms)
    assert man["layer.0"]["magic"] == "GDLT1" and man["adapter"]["magic"] == "SELA1"


def test_random_access_reads_only_the_target_atom():
    # the structural random-access invariant behind bench_cfold_store: read_atom returns exactly one atom's
    # bytes from a large store, byte-identical, regardless of the store's total size.
    n = 64
    atoms = {f"layer.{i}": gd.to_bytes(gd.compress(_correlated_group(seed=i), mode="centroid"))
             for i in range(n)}
    store = fmt.pack_atoms("weight_store", atoms)
    total = sum(len(a) for a in atoms.values())
    for name in ("layer.0", "layer.31", "layer.63"):
        got = fmt.read_atom(store, name)
        assert got == atoms[name]                            # exact target
        assert len(got) < total // 2                         # materialised << whole store (one atom of 64)


def test_read_atom_missing_name_raises():
    store = fmt.pack_atoms("weight_store", {"a": gd.to_bytes(gd.compress(_correlated_group(seed=4), mode="centroid"))})
    try:
        fmt.read_atom(store, "nope")
    except KeyError:
        pass
    else:
        raise AssertionError("read_atom should raise KeyError for an unknown atom name")


def test_pack_atoms_rejects_empty_and_unpack_atoms_rejects_plain():
    plain = fmt.pack("weight_store", {}, {}, {"values": np.arange(8, dtype=np.uint32)})
    try:
        fmt.unpack_atoms(plain)
    except ValueError as e:
        assert "atom" in str(e)
    else:
        raise AssertionError("unpack_atoms should reject a container with no atom sections")


def test_unpack_atom_rejects_non_atom_container():
    plain = fmt.pack("weight_store", {}, {}, {"values": np.arange(8, dtype=np.uint32)})
    try:
        fmt.unpack_atom(plain)
    except ValueError as e:
        assert "atom" in str(e)
    else:
        raise AssertionError("unpack_atom should reject a container with no 'atom' section")


if __name__ == "__main__":
    test_grouped_delta_atom_roundtrips_through_container()
    test_super_elastic_atom_roundtrips_through_container()
    test_atom_params_recorded_in_header()
    test_multi_atom_store_roundtrips_and_random_access()
    test_random_access_reads_only_the_target_atom()
    test_read_atom_missing_name_raises()
    test_pack_atoms_rejects_empty_and_unpack_atoms_rejects_plain()
    test_unpack_atom_rejects_non_atom_container()
    print("ALL PASSED")
