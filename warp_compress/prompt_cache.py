"""prompt_cache — the shared-prefix prompt-cache experiment (the LLM workload ChromoFold is built for).

A batch of K requests share a large system prompt but differ in a short suffix — the exact redundancy
ChromoFold exploits, and it avoids over-claiming on dense weights or KV replacement. Two independent things
are measured, and kept honestly separate:

  1. ChromoFold's job — the TOKEN STORE. Store the shared prefix ONCE + per-request suffixes; recover any
     request's span on the GPU in O(1). Reports storage vs raw-duplicated, batch capacity, span-recovery
     latency. This is what ChromoFold contributes.
  2. Why the batch shape matters — PREFILL / TTFT. Computing the shared-prefix KV once and reusing it across
     requests (standard prefix caching) cuts prefill and time-to-first-token. Measured on real gpt2. ChromoFold
     does NOT compute the KV — it provides the compact token store + span recovery that makes holding and
     routing such a batch cheap. Framed as such, no conflation.

Requires transformers for part 2 (downloads gpt2). Run: python -m warp_compress.prompt_cache
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import warp as wp

warnings.filterwarnings("ignore")
wp.init()


@wp.kernel
def _recover_k(prefix: wp.array(dtype=wp.int32), plen: int, suf: wp.array(dtype=wp.int32),
               suf_start: wp.array(dtype=wp.int32), req_in: wp.array(dtype=wp.int32),
               pos_in: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    r = req_in[t]
    p = pos_in[t]
    if p < plen:
        out[t] = prefix[p]                             # in the shared prefix (one copy for all requests)
    else:
        out[t] = suf[suf_start[r] + (p - plen)]        # in this request's private suffix


class SharedPrefixStore:
    """K requests = one shared prefix + K private suffixes, resident on the GPU; O(1) span recovery per token."""

    def __init__(self, prefix, suffixes, device: str = "cuda:0"):
        self.device = device
        self.prefix = np.asarray(prefix, np.int64)
        self.plen = int(self.prefix.shape[0])
        self.suf_len = np.asarray([len(s) for s in suffixes], np.int64)
        suf_flat = np.concatenate([np.asarray(s, np.int64) for s in suffixes]) if suffixes else np.zeros(1, np.int64)
        self.suf_start = np.concatenate([[0], np.cumsum(self.suf_len)])[:-1].astype(np.int64)
        self.K = len(suffixes)
        self._prefix = wp.array(self.prefix.astype(np.int32), dtype=wp.int32, device=device)
        self._suf = wp.array(suf_flat.astype(np.int32), dtype=wp.int32, device=device)
        self._suf_start = wp.array(self.suf_start.astype(np.int32), dtype=wp.int32, device=device)
        self._suf_total = int(suf_flat.shape[0])

    def req_len(self, r: int) -> int:
        return self.plen + int(self.suf_len[r])

    def size_bytes(self) -> int:
        """Shared-prefix storage: the prefix once + all suffixes (uint16 tokens)."""
        return (self.plen + self._suf_total) * 2

    def raw_duplicated_bytes(self) -> int:
        """Naive prompt cache: every request stores its own full copy of prefix + suffix."""
        return int(sum(self.req_len(r) for r in range(self.K))) * 2

    def recover(self, reqs, positions) -> np.ndarray:
        r = wp.array(np.asarray(reqs, np.int32), dtype=wp.int32, device=self.device)
        p = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(r.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_recover_k, dim=r.shape[0],
                  inputs=[self._prefix, self.plen, self._suf, self._suf_start, r, p, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def recover_request(self, r: int) -> np.ndarray:
        L = self.req_len(r)
        return self.recover(np.full(L, r, np.int32), np.arange(L, dtype=np.int32))


def _demo():
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)

    # ---- part 1: the token store (no model needed) ----
    K, PLEN, SLEN = 256, 800, 24
    prefix = rng.integers(0, 50257, PLEN).astype(np.int64)
    suffixes = [rng.integers(0, 50257, SLEN).astype(np.int64) for _ in range(K)]
    store = SharedPrefixStore(prefix, suffixes, device=dev)

    # correctness: recovered request == prefix + its suffix
    ok = all(np.array_equal(store.recover_request(r),
                            np.concatenate([prefix, suffixes[r]])) for r in (0, 100, 255))
    # span-recovery latency: a batch of random (request, position) reads
    Q = 1 << 18
    rq = rng.integers(0, K, Q).astype(np.int32)
    pp = np.array([rng.integers(0, store.req_len(r)) for r in rq], np.int32)
    for _ in range(3):
        store.recover(rq, pp)
    t0 = time.perf_counter()
    for _ in range(20):
        store.recover(rq, pp)
    ns = (time.perf_counter() - t0) / 20 / Q * 1e9

    dup, shared = store.raw_duplicated_bytes(), store.size_bytes()
    print(f"device={dev}   prompt-cache token store: K={K} requests, prefix={PLEN}, suffix={SLEN} tokens each")
    print(f"[correct] GPU span recovery == prefix+suffix ✓" if ok else "[correct] FAIL")
    print(f"[store]  raw-duplicated {dup/1e3:8.1f} KB   ChromoFold shared-prefix {shared/1e3:7.1f} KB   "
          f"=> {dup/shared:.1f}× smaller (⇒ ~{dup/shared:.0f}× more requests per VRAM budget)")
    print(f"[recover] {Q:,} random span tokens in {(Q*ns/1e9)*1e3:.2f} ms  ({ns:.1f} ns/token, O(1) on the GPU)")

    # ---- part 2: prefill / TTFT on a REAL model (prefix-KV sharing) ----
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        print("\n(part 2 skipped — transformers/torch not available)")
        return

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    sys_prompt = ("You are a careful assistant. Read the following policy and answer strictly according to it. "
                  * 20)
    P = tok(sys_prompt, return_tensors="pt").input_ids[:, :512]
    Km = 24
    sufs = [tok(f" Question {i}: what does section {i%9} say?", return_tensors="pt").input_ids for i in range(Km)]

    with torch.no_grad():
        # naive: prefill the whole (prefix+suffix) for every request
        t0 = time.perf_counter()
        for s in sufs:
            model(torch.cat([P, s], 1))
        naive = (time.perf_counter() - t0)

        # prefix-shared: compute the shared-prefix KV ONCE, reuse it; each request only prefills its suffix
        t0 = time.perf_counter()
        past = model(P, use_cache=True).past_key_values
        ttft = []
        for s in sufs:
            t1 = time.perf_counter()
            model(s, past_key_values=past, use_cache=True)
            ttft.append(time.perf_counter() - t1)
        shared_t = (time.perf_counter() - t0)

    print(f"\nprefill / TTFT on gpt2 (prefix={P.shape[1]} tok, {Km} requests, suffix≈{sufs[0].shape[1]} tok)")
    print(f"[prefill] naive (full prompt each) {naive*1e3:7.1f} ms   prefix-shared (KV once + suffixes) "
          f"{shared_t*1e3:7.1f} ms   => {naive/shared_t:.1f}× faster")
    print(f"[TTFT]    naive ≈ {naive/Km*1e3:.1f} ms/request   prefix-shared ≈ {np.median(ttft)*1e3:.1f} ms/request "
          f"(suffix-only forward on the shared KV)")
    print("\n=> honest split: the prefill/TTFT win is prefix-KV SHARING (a standard technique this batch shape "
          "enables). ChromoFold's contribution is the compact TOKEN STORE + O(1) GPU span recovery that makes\n"
          "   holding & routing a large shared-prefix batch cheap (22×+ smaller than duplicated). Together: more "
          "requests resident, each prefilled by reusing one shared prefix. No KV-replacement claim.")


if __name__ == "__main__":
    _demo()
