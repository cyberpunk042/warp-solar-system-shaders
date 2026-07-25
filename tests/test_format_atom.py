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
    test_unpack_atom_rejects_non_atom_container()
    print("ALL PASSED")
