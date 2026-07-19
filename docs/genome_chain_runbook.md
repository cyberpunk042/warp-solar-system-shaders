# Runbook — the genome-compression chain (handoff)

A pickup guide for continuing the **genome** work on branch `claude/warp-shader-raymarching-3cohzd`
(PR #49). Read this top to bottom before touching anything; it has the goal, the current state, how to
run/watch/render (GPU + WSL), the invariants you must not break, and the open work.

---

## 1. The goal (operator intent — do not dilute)

> One very long continuous GIF of the **whole** genome-compression process — an RTX graphics card's data
> compressed *biologically* from tokenization down to a chromosome — as **one flowing transformation**.
> **No cheating, no hack, no illogism.** Each stage starts from the previous stage's real output (matter
> is conserved — nothing spawned or teleported). **Transform, don't morph-cheat.** Reach a beautiful,
> tight chromosome *properly*, through the real process. From start to end the viewer is looking at **a
> single scene, a single shape that keeps retransforming** — no camera cuts/glitches. It must **start on
> the graphics card (stage 0)** and show **tokenization first**.

The 8-rung ladder (stage 0 is the card itself):

`gpu_board` → tokens → base pairs → double helices → nucleosome beads → 30 nm fibre → telomere → chromosome.

The compression is honest: **scan-and-merge** — every token carries a merge-codec type id; identical card
blocks share an id, so 182,872 base pairs collapse into ~143 unique chromosome slots (≈1279× merge).
Same element → same chromosome slot.

---

## 2. Current state (what's done)

- **All 8 stages exist and are individually verified** (physics-correct, no self-interpenetration). Each
  genome library preserves the pair ordering, so the *same matter* flows through every stage.
- **The whole take is one continuous scene:** `warp_genome_chain` — plays each stage's own timeline and
  **cross-dissolves** every seam (stage i's settled body melts into stage i+1 re-forming from it). No cuts.
- **Stage 0 is the real tokenization source:** the chain opens on `gpu_board` (the exact board
  `tokenize_card` voxel-samples), posed to the token field's angle, so the first dissolve is that board
  becoming its **own** tokens in place — not a cut to a different card. (This was the last fix.)
- **Deliverable GIF:** `docs/engine/genome_chain.gif` (~11 MB, ~21.6 s, 302 frames @ 14 fps, 384×256),
  reproducible with `python build_genome_chain.py`. Referenced from `docs/gallery.md`.

### Key files

| Path | Role |
|---|---|
| `warp_shaders/genome/tokenize.py` | Process 1 — samples `gpu_board.board_map` → tokens (+ merge-codec ids). The **source of the whole chain**. |
| `warp_shaders/genome/{basepair,helix,nucleosome,fibre,telomere,chromatid}.py` | Stage libraries. Each preserves pair order; `chromatid.py` holds the **scan-and-merge**. |
| `warp_shaders/genome/strand.py` | `min_separation(pts, gap)` — the excluded-volume (no pass-through) check. |
| `warp_shaders/scenes/warp_{tokenize,basepair,helix,nucleosome,fibre,telomere,chromosome,chromosome_x}.py` | The dedicated per-stage scenes (the beautiful renders). |
| `warp_shaders/scenes/warp_genome_chain.py` | **The continuous take**, as a normal `Scene` (device-threaded → GPU-ready). Timeline = `_SEG` + `_DISSOLVE`; `TOTAL` seconds. |
| `warp_shaders/scenes/gpu_board.py` | Stage 0 — the real board; `board_map` is what `tokenize_card` samples. Camera is `mouse`-steerable (az=0.14+mx·0.01, el=0.78+my·0.005). |
| `build_genome_chain.py` | Renders `warp_genome_chain` → the GIF (global cross-stage palette so the purple chromosome survives quantization). |
| `watch.py` | **Live MJPEG viewer** (WSL-friendly). |

There is also a `warp_genome` scene (an earlier single-point-cloud morph) — **superseded**; the
cross-dissolved `warp_genome_chain` is the one to use. `warp_chromosome_x` is the metaphase-X finale
variant (vs the single chromatid in `warp_chromosome`).

---

## 3. Run it

Everything auto-selects **CUDA when a GPU is present, else CPU** (`--device auto`, the default). Nothing
is hardcoded to CPU anymore.

```bash
# one-time
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt   # warp-lang, numpy, pillow

# list scenes / render a single frame (GPU if available)
python render.py --list
python render.py --scene warp_genome_chain --time 10 --width 960 --height 640 -o /tmp/f.png

# render the whole take to the deliverable GIF (auto device)
python build_genome_chain.py                       # -> docs/engine/genome_chain.gif
python build_genome_chain.py --device cuda --fps 18 --width 512 --height 342

# or via the generic CLI to an mp4 (smoother than a GIF)
python render.py --scene warp_genome_chain --frames 302 --fps 14 --device auto --video /tmp/genome.mp4
```

Verify a scene is physically honest (no interpenetration) — example for the fibre:

```bash
python -c "from warp_shaders.genome.fibre import coil_fibre; from warp_shaders.genome.strand import min_separation; \
import numpy as np; fb=coil_fibre(); m=0.5*(fb.fib_a+fb.fib_b); print('min_sep', min_separation(m, 0.0))"
# expect a positive number (strands touch but never pass through)
```

---

## 4. Watch it LIVE from WSL Ubuntu on Windows 11

`watch.py` renders continuously in a background thread and serves an **MJPEG stream over HTTP**. WSL2
forwards `localhost` to Windows automatically, so **no X server / WSLg / GUI toolkit is needed** — you
just open a URL in a Windows browser.

```bash
# in WSL Ubuntu:
python watch.py                                  # warp_genome_chain, auto device, :8008
python watch.py --scene warp_helix --width 640 --height 426 --fps 24
python watch.py --speed 0.5                      # half speed to study a transition
python watch.py --device cuda --port 8008
```

Then on **Windows**, open <http://localhost:8008>. The page shows the live frame + a HUD
(`frame N · fps · t=…s`). The take **loops** at `TOTAL` seconds (any scene exposing `TOTAL`); other
scenes free-run. `Ctrl-C` stops it.

- If `localhost` ever doesn't forward, use the WSL IP: `ip addr show eth0` → open `http://<that-ip>:8008`.
- CPU does ~6 fps @ 256×170; a CUDA GPU runs it smoothly at higher res — raise `--width/--height`.
- The renderer rate is capped at `--fps` so a fast GPU doesn't spin uselessly.

---

## 5. Invariants — the "no cheating" rules (do not break)

1. **Conservation.** Every stage consumes the previous stage's real output. `n` pairs is constant
   (P=182,872 from 365,744 tokens). Nothing is spawned; nothing teleports. If you add matter, it's a bug.
2. **No interpenetration.** `strand.min_separation(midpoints, gap)` must stay **positive** for every
   stage (turns/beads touch but never pass through). Re-check after any geometry change.
3. **Registration at every seam.** A transition must show the *same matter* re-forming **in place** —
   same world/camera so it reads as one shape, not a cut. The stage-0 fix (posing `gpu_board` to the
   token angle) is the template: verify a 50/50 blend of the two adjacent renders lines up.
4. **Honest compression.** The chromosome uses the real **scan-and-merge** (`chromatid.py`): identical
   token ids share a slot (~143 unique). Don't fake the tightening.
5. **One camera, no glitches.** The take is continuous; seams are cross-dissolves, never hard cuts.
6. **Operator words are sacrosanct** — quote verbatim, log before acting; **adding ≠ discarding** (layer
   new direction onto prior, don't rewrite wholesale).

---

## 6. Open work / next steps

- **KNOWN DEFECT — stray trailing frame in `docs/engine/genome_chain.gif`.** A cross-branch merge
  (`8ab821a`) + GIF recompress (`2ab2795`, `gifsicle -O3 --lossy=60`) left the committed GIF ending on a
  single **gpu_board** frame *after* the chromosome (frame 282 = chromosome ✓, frame 283 = board ✗) — a
  one-frame flash at the loop point, exactly the kind of glitch the operator rejects. The rendered scene
  (`warp_genome_chain`) is correct and ends on the chromosome; only the encoded artifact is wrong.
  **Fix (needs `gifsicle`, which this box lacks):**
  `gifsicle docs/engine/genome_chain.gif '#0-282' -O3 --lossy=60 -o docs/engine/genome_chain.gif`
  (drops the last frame, keeps the ~5.8 MB size). Or regenerate from source
  (`python build_genome_chain.py`) — correct ordering, no stray frame — then re-shrink with the same
  `gifsicle` pass to stay small. Don't re-encode with PIL alone: it can't match the lossy size (~11 MB).
- **Audit every seam for registration (the current live thread).** Stage 0→1 (card→tokens) is fixed and
  verified in place. The operator's standing bar is *"all the scene must make sense"* — confirm each later
  seam is the **same matter re-forming in place**, not a jump: tokens→base-pairs, base-pairs→helices,
  helices→nucleosomes, …→chromosome. For each, render the two adjacent sub-scenes at the seam time and
  blend 50/50 (see §5.3); fix any that jump the way stage 0 was fixed (match world/camera/pose).
- **Finale choice.** Single chromatid (`warp_chromosome`) vs metaphase **X** (`warp_chromosome_x`). Offer
  both; the operator asked to "try both."
- **Length / pacing.** The operator wanted it *"very very very long."* Per-stage `play_secs` and
  `_DISSOLVE` live in `warp_genome_chain.py::_SEG`. Lengthen there, then re-run `build_genome_chain.py`.
- **Higher-res deliverable** once on GPU: bump `--width/--height`; consider an `.mp4` (via `render.py
  --video`) which looks smoother and has no 128/256-color palette limits.

---

## 7. Gotchas (learned the hard way)

- **GIF upload/asset cap is 30 MiB.** Keep the deliverable small: 384×256, ~14 fps, and the **global
  palette sampled across all stages** (in `build_genome_chain.py`) — a palette built from one stage
  quantizes the purple chromosome to pale pink. Prefer `.mp4` for high-res.
- **Commits must be verified:** `git config user.email noreply@anthropic.com && git config user.name
  Claude` or GitHub marks them "Unverified".
- **Shell cwd can reset between commands** in this environment — use absolute paths or
  `PYTHONPATH=/path/to/repo` when running scripts.
- **`gpu_board` first compile is slow** (~6 s) and it's a ray-marched SDF (heavier than the splat stages);
  that's why stage 0's segment is short.
- **Warp device:** `wp.get_cuda_device_count() > 0` gates GPU. Scenes thread `device` through `wp.launch`;
  don't reintroduce a hardcoded `"cpu"`.

---

## 8. Branch / PR

- Branch: `claude/warp-shader-raymarching-3cohzd` — **push here only** (`git push -u origin <branch>`,
  retry with backoff on network errors).
- PR #49 (draft). If it's already merged, start fresh from the latest default branch under the **same**
  branch name; don't stack new commits on merged history.
- The GitHub MCP tools (`mcp__github__*`) are the way to inspect/update the PR (no `gh` CLI here).
