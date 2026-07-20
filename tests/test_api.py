"""The high-level ChromoFold API: compress() dispatches by data shape; Artifact decode/fetch/save round-trip."""
import numpy as np

import warp as wp

from warp_compress.api import Artifact, compress

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def test_weights_artifact_roundtrips_and_saves():
    W = (np.random.default_rng(0).standard_normal((256, 128)) / 16).astype(np.float32)
    art = compress(W, bits=4, device=_DEV)
    assert art.kind == "weights"
    r = art.decode()
    assert r.shape == W.shape
    back = Artifact.load(art.save(), device=_DEV)
    assert np.array_equal(r, back.decode())              # portable .cfold round-trip is byte-identical
    idx = np.array([0, 5, 100])
    assert np.allclose(art.fetch(idx), r.ravel()[idx])


def test_tokens_artifact_is_addressable_and_saves():
    toks = np.random.default_rng(1).integers(0, 3000, 20000).astype(np.int64)
    art = compress(toks, device=_DEV)
    assert art.kind == "tokens"
    full = art.decode()
    assert np.array_equal(full, toks)                    # RRR self-index over the sequence is lossless
    assert np.array_equal(art.fetch([1, 500, 19999]), toks[[1, 500, 19999]])
    assert np.array_equal(Artifact.load(art.save(), device=_DEV).decode(), toks)


def test_batch_dispatches_to_a_cluster_backend():
    rng = np.random.default_rng(2)
    # a mixed batch of shared-prefix requests -> seed
    prompts = [rng.integers(0, 5000, 200).astype(np.int64) for _ in range(3)]
    seqs = [np.concatenate([prompts[int(rng.integers(0, 3))], rng.integers(0, 5000, 12).astype(np.int64)])
            for _ in range(40)]
    art = compress(seqs, device=_DEV)
    assert art.kind in ("seed", "delta", "dedup")
    dec = art.decode()
    assert all(np.array_equal(a, b) for a, b in zip(dec, seqs))


def test_size_and_summary():
    W = (np.random.default_rng(3).standard_normal((128, 128)) / 10).astype(np.float32)
    art = compress(W, bits=8, device=_DEV)
    assert art.size_bytes() > 0
    assert "ChromoFold Artifact [weights]" in art.summary()


def test_cluster_save_is_honestly_unimplemented():
    seqs = [np.arange(50, dtype=np.int64) for _ in range(4)]
    art = compress(seqs, device=_DEV)
    try:
        art.save()
        assert art.kind in ("weights", "tokens")         # only these serialise; clusters decode in place
    except NotImplementedError:
        pass
