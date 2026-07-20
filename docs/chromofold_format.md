# The ChromoFold Container Format & Protocol — v1.0

The on-disk / on-wire schema for a ChromoFold-compressed object, the encode/decode algorithm, and the
byte-level layout of the succinct structures. A ChromoFold artifact is **one self-describing, versioned blob**:
a compressed weight tensor, a whole model, an RRR self-index, a delta cluster — all share the same container.

Reference implementation: [`warp_compress/format.py`](../warp_compress/format.py) (`pack` / `unpack`),
with per-object `save`/`load` on the stores.

---

## 1. Design goals

1. **Self-describing** — a reader needs nothing but the blob: the header names every array's dtype, shape, and
   length, and records the *pipeline* that produced it.
2. **Versioned** — a magic byte + a `[major, minor]` version; readers reject unknown majors, ignore unknown
   minors, and skip unknown sections.
3. **Zero-copy friendly** — the payload is raw little-endian array bytes concatenated in header order, so a
   reader slices directly into `numpy`/GPU buffers with no parsing of the data region.
4. **Composable** — a container may nest containers (a model file is a container of weight-store blobs), so the
   same 4 primitives describe a single tensor or a whole model.
5. **Codec-agnostic** — the header's `config` records the transform/coder used, so new object types and coders
   extend the format without breaking old readers.

---

## 2. Container byte layout

All integers **little-endian**. Offsets in bytes.

```
┌─────────┬──────────────────────────────────────────────────────────────────┐
│ 0  ..  7 │ MAGIC = "CHROMOF" (7 ASCII) + 0x01 (container-format version byte) │
│ 8  .. 11 │ HEADER_LEN : uint32                                                │
│ 12 .. 11+HEADER_LEN │ HEADER : UTF-8 JSON (see §3)                            │
│ 12+HEADER_LEN .. EOF │ PAYLOAD : each section's raw bytes, in `sections` order│
└─────────┴──────────────────────────────────────────────────────────────────┘
```

A conforming reader: (1) checks `MAGIC[:7] == b"CHROMOF"`; (2) reads `MAGIC[7]` — the container-format version,
`0x01` here; (3) reads `HEADER_LEN`, parses the JSON; (4) walks `sections`, slicing `nbytes` per section from
the payload, in order, into arrays of the stated `dtype`/`shape`.

---

## 3. Header JSON schema

```jsonc
{
  "format":  "chromofold",          // literal; reject otherwise
  "version": [1, 0],                // [major, minor] of the SCHEMA (distinct from the magic byte)
  "object":  "weight_store",        // object type — selects the section registry (§4)
  "config":  { ... },               // the ChromoFoldConfig view: the pipeline that produced the payload
  "params":  { ... },               // object scalars: bits, shape, scale(s), n, group_size, wavelet params …
  "sections":[                      // ordered; the payload is these arrays back-to-back
    {"name":"cwords","dtype":"<u4","shape":[N],"nbytes":4N},
    {"name":"_scales","dtype":"<f4","shape":[G],"nbytes":4G},
    …
  ]
}
```

- `dtype` is a numpy type-string (`"<u4"`, `"<i4"`, `"<f4"`, `"<f2"`, `"|u1"`). Endianness is little.
- `config` is advisory metadata (the pipeline) — e.g. `{"quantize":"int4","transform":"none","code":"huffman","group_size":128}`.
- `params` carries everything the object needs to rebuild besides the arrays (see §4).
- Unknown sections/params **must be ignored**, not errored, to allow minor-version growth.

---

## 4. Object types & section registries

### 4.1 `weight_store` — an entropy-coded quantized tensor

The headline object: a weight tensor quantized to `bits`, its quantized values entropy-coded by the RRR
wavelet (optionally with the class-stream Huffman), and per-tensor or per-group dequant scales.

`params`: `{bits, shape, zero, n, group_size, huffman, scale?, wm}` where `zero = 2^(bits-1)-1` (the offset that
maps signed levels to `[0, 2·zero]`), `scale` is present iff `group_size` is null, and `wm` are the wavelet's
scalar params (§4.3/§4.4).

`sections`: the wavelet's arrays (§4.3 or §4.4) plus, when `group_size` is set, `_scales` (`<f4`, one per group).

**Dequant:** `W[i] = (value[i] − zero) · scale_of(i)`, where `scale_of(i) = scale` (per-tensor) or
`_scales[i // group_size]` (per-group). `value[i]` is `wavelet.access(i)`.

### 4.2 `model` — a whole compressed model

A container of containers. `params.manifest` is an ordered list of `{name, kind}`:

- `kind == "store"` → section `"S:<name>"` is a **nested `weight_store` blob** (as raw `|u1` bytes).
- `kind == "tensor"` → section `"T:<name>"` is a small tensor kept in fp16 (`<f2`), with `shape`.

To load: for each manifest entry, either `weight_store.load(section_bytes)` or read the fp16 tensor. This is how
a full model (e.g. gpt2 int8) becomes a single ~93 MB `.cfold` file that reloads and generates.

### 4.3 `rrr_wavelet` sections (4-bit class stream)

`params.wm`: `{n, bits, bits_stored, sb_bytes}`. `sections`:

| name       | dtype | meaning                                                            |
|------------|-------|-------------------------------------------------------------------|
| `classes`  | `<u4` | packed 4-bit class per RRR block, per level (see §5)               |
| `offsets`  | `<u4` | variable-width enumerative offsets, bit-concatenated, all levels   |
| `sbrank`   | `<i4` | `[levels, nsb+1]` superblock cumulative popcount                   |
| `sboff`    | `<i4` | `[levels, nsb+1]` superblock cumulative offset-bit position        |
| `offbase`  | `<i4` | `[levels]` per-level word base into `offsets`                      |
| `zeros`    | `<i4` | `[levels]` #zeros at each wavelet level (the LF split point)       |

`width`/`binom` (the offset-width table and Pascal's triangle) are **constants**, recomputed on load, not stored.

### 4.4 `huff_wavelet` sections (Huffman class stream)

As §4.3, but the class stream is canonical-Huffman-coded per level, so it replaces `classes` with `cwords`
(MSB-first Huffman bitstream), adds `sbclass` (`[levels, nsb+1]` superblock class-bit positions) and `cbase`
(`[levels]` per-level word base into `cwords`), and stores the per-level canonical decode tables `fc`, `cnt`,
`fidx` (`[levels, maxlen+1]`) and `syms` (`[levels, 16]`), plus `maxlens` (`[levels]`). `params.wm` adds `maxlen`.

---

## 5. The compression algorithm (encode ⇄ decode)

ChromoFold is a **pipeline** (see `docs/chromofold.md` §2); the container stores the output of whichever stages
ran. For the `weight_store` path:

**Encode**
1. **Quantize** the fp32 tensor to `bits` with a per-tensor or per-group max-abs scale →
   `q[i] ∈ [0, 2^bits)` (signed levels shifted by `zero`).
2. **Wavelet-decompose** `q` into `bits` bitplanes (the wavelet matrix: stable-partition by the level bit).
3. **RRR-code each plane**: split into length-`T=15` blocks; store each block as
   `(class = popcount, offset = enumerative rank among all T-bit words of that popcount)`; sample cumulative
   rank / offset-bit / class-bit every `S=64` blocks (the superblocks).
4. **Entropy-code the class stream**: fixed 4-bit (`rrr_wavelet`) or canonical Huffman over the 16 class values
   (`huff_wavelet`), decoded in-kernel.
5. **Serialize** the resulting arrays + scalars into the container (§2–§4).

**Decode (random access to `value[i]`)** — no full decompression:
1. `access(i)` walks the `bits` wavelet levels. At each level it needs `rank1(pos)` on that level's RRR plane.
2. `rank1(pos)`: jump to the superblock (`sbrank`), then scan the in-superblock blocks summing their `class`
   (decoding the class from the 4-bit or Huffman stream, advancing the offset-bit cursor by `width[class]`),
   then for the target block **decode one block in registers** (enumerative unrank of `(class, offset)` → the
   `T`-bit word) and popcount its low `pos%T` bits.
3. Dequant: `W[i] = (value[i] − zero) · scale_of(i)`.

Every step is O(1)/O(log) and GPU-resident; §4–§7 of `docs/chromofold.md` and the `bench_*` records give the
measured throughput and sizes.

---

## 6. Versioning & compatibility

- **Magic byte** (`MAGIC[7]`) is the *container* format version; a reader that doesn't recognise it must refuse.
- **`version` `[major, minor]`** is the *schema* version. Same major ⇒ compatible; a reader ignores unknown
  minor-version additions (new params, new sections, new object types it doesn't handle).
- **Forward-compat rule:** unknown `sections` and `params` keys are skipped, never errored. New coders/objects
  are added by registering a new `object` value and its section set — old readers simply don't load them.
- **Endianness** is fixed little-endian; producers on big-endian hosts byte-swap before packing.

---

## 7. Worked example (a compressed weight tensor)

```python
from warp_compress.weight_store import QuantizedWeightStore
from warp_compress import format as fmt

st   = QuantizedWeightStore(W, bits=4, huffman=True, group_size=128)   # quantize + entropy-code
blob = st.save()                                                       # -> one bytes object
print(fmt.summary(blob))
# chromofold v1.0  object=weight_store  pipeline={'quantize':'int4','transform':'none',
#   'code':'huffman','group_size':128}  14 sections  36.1 KB payload / 37.2 KB total

st2 = QuantizedWeightStore.load(blob)          # rebuild on the GPU
assert (st2.reconstruct() == st.reconstruct()).all()   # byte-identical
```

A whole model: `save_model(model, stores)` → one `.cfold` blob (gpt2 int8 ≈ 93 MB vs fp16 249 MB);
`apply_model(model, load_model(blob))` reloads it and the model generates.

---

## 8. Conformance checklist

A reader is conformant if it: validates the magic and container version; parses the header JSON; slices the
payload strictly by `sections[].nbytes` in order; rebuilds arrays with the exact `dtype`/`shape`; dequantizes
per §4.1; **ignores** unknown sections/params; and treats a mismatched schema major or a bad magic as a hard
error. A writer is conformant if it emits little-endian arrays, a complete `sections` list whose `nbytes` sum
equals the payload length, and a truthful `object`/`config`/`params`.
