# Research 08 — ecosystem: life at the population scale

The strand so far has grown *one* organism at a time — a plant, a mind, a wave of
its possible futures. This note is about **many** of them living together: a
meadow that changes over **years**, with seasons, competition for light, and
turnover. Source behind `warp_shaders/life/ecosystem.py` and `scenes/ecosystem.py`.

## From specimen to habitat

The `meadow` scene ([Research 04](04-lsystems.md)) placed several plants in one
mesh, swaying in one wind — but it was a *snapshot*. An **ecosystem** adds *time
at the population scale*: plants are **born, grow, bloom, senesce and die**, and
new seedlings fill the gaps, so the meadow's composition drifts year over year.
`time` is measured in **years** (1.0 = a full spring→winter cycle).

The whole thing is **deterministic from a seed**: a fixed pool of plants with
staggered birth and death times, so the same run replays identically (no
per-frame randomness to flicker). Each plant is grown through the ordinary
L-System pipeline; the ecosystem only decides *which* plants are alive, *how
grown* they are, *what colour* they wear this season, and *which way they lean*.

## Seasons

`season_phase(t)` is the fraction through the year (0 = spring, ¼ = summer, ½ =
autumn, ¾ = winter). Two things follow from it:

- **Colour** — `season_palette` interpolates the leaf palette: greens through
  spring and summer, warming to **gold** in autumn, fading to a muted **brown**
  in winter. (Only the leaf/bark entries move; a bloom keeps its colour.)
- **Vigour** — `vigor` is the canopy fullness, peaking in summer and bottoming in
  winter; it scales each plant's leaf size, so the meadow visibly **thins** in the
  cold and fills again in spring — a deciduous rhythm.

The scene also drops and warms the **sun** by season (a low, cool winter light; a
warm, lower autumn sun) so the whole frame reads the time of year.

## Competition for light

Plants do not grow in isolation — a taller neighbour **shades** a shorter one.
For each plant, `standing(t)` sums the shading from taller plants within a small
radius (nearer + taller = more shade) into an available-**light** value in
[0.2, 1]. Two consequences:

- a shaded plant grows to a **lower generation** (light gates its final size), so
  runts stay small under the canopy;
- it **leans toward the open sky** — the shading directions are summed and the
  plant's phototropism (the tropism layer from [Research 04](04-lsystems.md)
  §2.3.4) is aimed *away* from its tallest neighbours, so crowded plants bend
  toward the gaps. The more shaded, the harder the lean.

This is the same tropism mechanism the single-plant scenes use, now driven by the
plant's **neighbours** instead of a scripted light — competition as an emergent
consequence of local rules.

## Turnover

Births are staggered: an **establishment cohort** starts the meadow, and further
plants **recruit** over the following years. Each plant has a finite lifespan and
**senesces** (its growth fades) near the end of it, then dies. So over a multi-
year run the starting cohort thins while new seedlings appear — the meadow
**turns over**, its makeup never quite the same from one year to the next.

## Where this sits

This is the strand going **wide** rather than deep: after molecule → cell → plant
→ mind → wave-of-futures, the ecosystem is *life as a population* — many plants,
seasons, competition, birth and death. What remains open is giving each plant its
own **mind** (competition between deliberating agents, not just heights), genuine
**seed dispersal** (offspring near successful parents), and longer evolutionary
drift in the species mix. The engine now shows life from a strand of DNA to a
meadow living through its years.
