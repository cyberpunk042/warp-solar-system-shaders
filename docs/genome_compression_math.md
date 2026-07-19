# The genome compression engine as procedural, token-addressable compression

*What the multistage engine (`warp_shaders/genome/thread.py` + the stage libs) actually is, mathematically,
and what can be extracted from it: a way to pack a token sequence into a small space while keeping O(1)
token-by-token navigation — because every operation is a closed-form ("warp procedural") map.*

---

## 1. The objects

Start from the real card. Voxel-sample its SDF (`gpu_board.board_map`) → an occupancy grid; every occupied
bit is a **token**. Two arrays come out (`genome/tokenize.py`):

- **positions** `P ∈ ℝ^{N×3}` — each token's home in card space.
- **type ids** `t ∈ {0,…,V−1}^N` — from the **merge codec** (`warp_compress.mergecube`): byte-identical
  card blocks share an id. So `t` is already a *clustering* — `V ≪ N` distinct types, each repeated.

This is the whole input: `N` ordered items, each carrying one of `V` reusable content vectors.

## 2. The read order — serialization is a bijection

`thread._scan_order` reads the card as a **boustrophedon raster** (snake: rows in z, alternating x each
row). That is a permutation

```
σ : {0,…,N−1} → {0,…,N−1}         rank r = σ(token i)  =  position along the thread
```

`σ` is the *serialization* of the 2-D/3-D card into one 1-D **thread**. The snake matters: it is
**locality-preserving** — consecutive ranks `r, r+1` are spatial neighbours on the card. Everything
downstream inherits this.

## 3. Each stage is a closed-form embedding Φ_k

A "stage" is a map from **rank** to a point in ℝ³:

```
Φ_k : {0,…,N−1} → ℝ³          Φ_k(r) = position of the r-th thread element at stage k
```

Every `Φ_k` in the engine is **analytic** — grid arithmetic + trig, no lookup tables (this is the
"all we do is warp procedural operations" point). Concretely (current `thread.py`):

| stage k        | Φ_k(r) (sketch)                                                                    | shape |
|----------------|------------------------------------------------------------------------------------|-------|
| card / tokens  | `P[σ⁻¹(r)]`                                                                         | the board |
| base pairs     | boustrophedon grid, upright rung `(col·s_x, y₀±h, row·s_z)`                         | tight ladder |
| helix          | column grid + `(cosθ, y, sinθ)`, `θ = r·2π/10.5`                                   | forest of double helices |
| nucleosome     | bead grid + `(ρcosφ, ·, ρsinφ)`, `φ = (r mod G)/G·1.75·2π`                          | beads on a string |
| 30 nm fibre    | solenoid `(Rcos ψ, y(r), Rsin ψ)` of the bead centres                              | coil |
| chromatid / X  | pinched, tilted rod; sister = mirror                                               | metaphase X |

Three properties hold for **every** stage, by construction:

1. **Order-preserving / locally continuous.** `‖Φ_k(r+1) − Φ_k(r)‖` is small and bounded — each layout is a
   snake/coil that never teleports. Neighbours in rank stay neighbours in space.
2. **Closed form, O(1).** `Φ_k(r)` is computed directly from `r` (and `k`'s constants). No table, no scan.
3. **Volume-shrinking.** `diam(Φ_{k+1}(·)) < diam(Φ_k(·))`. Each stage folds the same `N` points into a
   strictly smaller bounding volume — that is the *compression* the animation shows.

The chain is `Φ_card → Φ_pair → … → Φ_chromo`, and the engine's invariant is that stage `k+1` **starts from
stage k's exact end frame** (`thread.frame` only interpolates consecutive `Φ`s). So the whole thing is one
continuous, invertible deformation of a single ordered point set.

## 4. What is actually compressed

The naive storage of the sequence is `N` items. With this engine it factors into three tiny pieces:

```
sequence  ≡  ( content book: V vectors ,  the map Φ_k (a handful of constants) ,  the id stream t )
```

- The **positions are free** — never stored, always `Φ_k(r)` on demand.
- The **content** is `V` unique vectors, not `N` (merge-codec dedup).
- The **order** is `σ` — itself procedural (a raster), so ~free; only the per-rank **id** `t[σ⁻¹(r)]` is real
  data, and that is exactly the merge-coded stream (identical elements → one symbol + a count).

So the engine is a **positional code**: it turns "where is token r, and how do I get to r+1" from stored
data into *evaluated* data. Compression = content dedup (merge codec) + implicit, procedural geometry.

## 5. Token-by-token navigation inside the compressed form

This is the payoff and it is real:

- **Random access:** token at rank `r` → `Φ_k(r)` in O(1). No decompression of the rest.
- **Walk:** `next = r+1`. Because `Φ_k` is order-preserving, that is a **local** hop in the compressed
  space — you can traverse the chromosome token by token without ever unfolding it.
- **Inverse (decompress a region):** given a point `x` in the compact body, recover its rank by inverting
  `Φ_k` (analytic where the layout is a pure grid/coil; nearest-token otherwise). So a spatial query →
  a rank range → the token ids there.

Compare to a normal compressor (gzip, a KV-cache blob): random access requires decompressing up to that
point. Here addressing is closed-form and O(1), because the geometry *is* the index.

## 6. Extraction: a general "chromosome" for token sequences

Drop the DNA vocabulary; keep the machinery. For **any** ordered token sequence (e.g. an LLM context):

1. **Cluster** with a merge codec → `V` reusable content vectors + an id stream (dedup + run-length).
2. **Serialize** with a locality-preserving space-filling curve `σ` (boustrophedon here; Hilbert is the
   stronger choice — better neighbour locality, still closed-form and invertible).
3. **Embed** with a shrinking stack of closed-form `Φ_k` — a compression *hierarchy* (coarse chromosome →
   fine tokens), each level a smaller addressable volume.

Store `(content book, σ, {Φ_k constants}, id stream)`. To use it:

- **decompress token r:** `id = t[σ⁻¹(r)]`, `content = book[id]`, `x = Φ_k(r)`.
- **navigate:** `r ± 1`, all closed-form, no full unfold.

For a **language model** this is an *addressable compressed memory*: a long context is held as a chromosome
— `V` unique embeddings + the procedural map — and the model reads it **token by token via `Φ`/`σ`**,
paying O(1) per access instead of storing/attending over the full `N`. The multi-scale stages give a natural
**coarse-to-fine retrieval** (attend at the fibre/chromosome scale, drill to the token scale on demand).

## 7. What's solid vs. what needs proof

- **Solid:** every `Φ_k` is closed-form, order-preserving, volume-shrinking; `σ` is an invertible
  locality-preserving serialization; the merge codec is genuine content dedup with O(1) addressing.
- **To establish (the research):**
  - *Invertibility budget* — the grid/coil `Φ_k` are analytically invertible, but the pinched/mirrored
    chromatid is not globally injective (two sisters + the pinch). Quantify where inversion is exact vs.
    nearest-token, and bound the error.
  - *Rate* — measure real bits: `V·dim(content) + |RLE(t)| + |σ,Φ constants|` vs. baseline. The claim
    "clusters into smaller space" needs a rate–distortion curve.
  - *Locality of `σ`* — swap boustrophedon → Hilbert and measure `E‖Φ(r+1)−Φ(r)‖` and neighbour recall.
  - *LM utility* — does token-by-token navigation over a chromosome-compressed context match full-context
    quality at lower memory? That is the actual experiment.

## 7b. The principled realization: wavelet matrix + FM-index (built)

The flat `Φ` gives O(1) *positional* access; the principled compressed **self-index** over the alphabet is
the **wavelet matrix** (`warp_compress/wavelet.py`): `access(i)`, `rank(c,i)`, `select(c,k)` in O(bits),
index size near the H0 entropy bound. Put `rank` behind the sequence's **BWT** and you get the **FM-index**
(`warp_compress/fm_index.py`): `count(pattern)` / `locate(pattern)` by backward search in O(|pattern|·bits),
**inside the compressed sequence, never materialised**. That is exactly how genomic reads are aligned — so
the DNA metaphor is literal: the compressed, addressable, *searchable* token index is the FM-index. One
sequence, three capabilities: **compressed** (near H0), **addressable** (token_chromosome / wavelet),
**searchable** (FM-index). All tested (`tests/test_wavelet_fm.py`, `tests/test_token_chromosome.py`).

## 7c. Recursion: X/Y chromosomes, base-pair merge, super-chromosomes (built)

The single-chromosome fold compresses **one** sequence. Lifting it to a whole **cluster** is the natural
recursion (`warp_compress/super_chromosome.py`): give every chromosome a **type — X or Y** — and let an X and
a Y **merge into a new base-pair strand** (position `i` pairs `X[i]` with `Y[i]`; identical rungs deduped).
That merged strand is itself a token sequence, so it **re-enters the same fold** as a *super-chromosome*.
Recurse, pairing two at a time, until one root remains — depth `~log2(#chromosomes)`, so the transform scales
with the cluster, "relative to size and depth."

- **Lossless** — `decode()` unzips the whole tree back to the originals; reads only the root strand + the
  per-merge codebooks (interior strands are procedural, never stored).
- **O(depth) random access** — `fetch(leaf, pos)`: positional pairing keeps `pos` constant down the tree, so
  one original token is a straight descent through `log2(K)` pair-codebooks — no full unfold. gzip/LZ cannot
  do this.
- **Small-alphabet payoff** — over ACGT (V=4) there are only V×V base-pair types, so codebooks stay tiny and
  the rung strand is low-entropy. Measured on 16 related 600-bp chromosomes: at realistic population
  divergence (~0.5–1%) the super-chromosome **beats both raw and gzip** (6.0× / 1.1× at 0.5%); as edits
  scatter, gzip's LZ overtakes on pure ratio (still beating raw to ~10%), but only the recursion gives
  O(depth) addressing. Tested (`tests/test_super_chromosome.py`, 7 cases).

This is the same fold read one level up: a chromosome compresses tokens; a super-chromosome compresses
chromosomes. The X/Y typing is literal — two strands become the base pairs of the next level.

## 8. Next steps to evolve it

1. Extract a standalone `token_chromosome` module: `compress(tokens) → Chromosome`, `Chromosome.at(r)`,
   `.next(r)`, `.invert(x)`, `.decompress(range)` — no rendering, pure math over the same `Φ_k`.
2. Replace boustrophedon with a Hilbert `σ` (keeps O(1), improves locality).
3. Add exact inverses for the grid/coil stages; nearest-token fallback with an error bound elsewhere.
4. Benchmark rate–distortion vs. gzip/zstd on token streams, and O(1)-access vs. block decompression.
5. Prototype the LM memory: store a long context as a chromosome, navigate `Φ`/`σ` at inference, compare
   quality/memory to the full KV cache.

---

*The animation was the proof of concept that the whole pipeline is one continuous, invertible, procedural
deformation of an ordered token set. The compression engine is that same fact, read as math: geometry as
an O(1) index into a clustered, addressable, hierarchically-compressed sequence.*
