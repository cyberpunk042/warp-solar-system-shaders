"""ChromoFold container format: pack/unpack round-trip, weight-store + model save/load, versioning rules."""
import numpy as np

import warp as wp

from warp_compress import format as fmt
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def test_pack_unpack_round_trips_arrays_and_header():
    arrays = {"a": np.arange(10, dtype=np.uint32), "b": np.array([[1.5, 2.5]], np.float32),
              "c": (np.arange(6, dtype=np.uint8))}
    blob = fmt.pack("weight_store", {"code": "rrr"}, {"bits": 4}, arrays)
    header, back = fmt.unpack(blob)
    assert header["object"] == "weight_store" and header["version"] == fmt.VERSION
    assert header["params"]["bits"] == 4 and header["config"]["code"] == "rrr"
    for k, v in arrays.items():
        assert back[k].dtype == np.ascontiguousarray(v).dtype and np.array_equal(back[k], v)


def test_bad_magic_is_rejected():
    try:
        fmt.unpack(b"NOTACHROMOFOLD" + b"\x00" * 40)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_payload_length_matches_sections():
    blob = fmt.pack("x", {}, {}, {"v": np.zeros(100, np.int32)})
    header, _ = fmt.unpack(blob)
    payload = sum(s["nbytes"] for s in header["sections"])
    assert payload == 100 * 4


def _wroundtrip(bits, huffman, group_size, seed=0):
    W = (np.random.default_rng(seed).standard_normal((128, 96)) / 10).astype(np.float32)
    st = QuantizedWeightStore(W, bits=bits, huffman=huffman, device=_DEV, group_size=group_size)
    st2 = QuantizedWeightStore.load(st.save(), device=_DEV)
    assert np.array_equal(st.reconstruct(), st2.reconstruct())      # byte-identical after serialise+load
    assert st2.shape == st.shape and st2.bits == st.bits and st2.group_size == st.group_size


def test_weight_store_save_load_all_configs():
    for bits in (4, 8):
        for huffman in (False, True):
            for gs in (None, 64):
                _wroundtrip(bits, huffman, gs, seed=bits + int(huffman) + (gs or 0))


def test_weight_store_blob_is_self_describing():
    W = (np.random.default_rng(1).standard_normal((64, 64)) / 8).astype(np.float32)
    blob = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, group_size=128).save()
    header, _ = fmt.unpack(blob)
    assert header["object"] == "weight_store"
    assert header["config"]["quantize"] == "int4" and header["config"]["code"] == "huffman"
    assert header["params"]["group_size"] == 128
    assert "chromofold v1.0" in fmt.summary(blob) and "weight_store" in fmt.summary(blob)


def test_model_container_round_trips():
    import torch
    import torch.nn as nn
    from warp_compress.model_store import compress_model, save_model, load_model, apply_model
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
    stores, _ = compress_model(model, bits=8, min_numel=10_000, device=_DEV)
    reloaded = load_model(save_model(model, stores), device=_DEV)
    # a reference model with the reconstructed weights == a model loaded purely from the blob
    ref = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
    ref.load_state_dict(model.state_dict())
    from warp_compress.model_store import reconstruct_into
    reconstruct_into(ref, stores)
    with torch.no_grad():
        for n, p in ref.named_parameters():          # the file keeps small tensors in fp16 — match that
            if n not in stores:
                p.copy_(p.half().float())
    m2 = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
    apply_model(m2, reloaded)
    x = torch.randn(3, 128)
    with torch.no_grad():
        assert torch.allclose(ref(x), m2(x), atol=1e-5)            # loaded-from-file == reconstructed (+ fp16 kept)
