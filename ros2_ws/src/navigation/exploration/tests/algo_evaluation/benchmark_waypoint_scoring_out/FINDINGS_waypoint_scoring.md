# Waypoint scoring ablation — findings

Investigation of the exploration planner's waypoint **scoring** logic
(`greedy_set_cover` in `explore_costmap_map.py`). Goal: shorten the executed
path and improve coverage-per-metre without hurting coverage or plan latency.

All numbers are produced by `benchmark_waypoint_scoring.py`, driving the real
exploration session headlessly over 3 synthetic maps (`large_rectangle`,
`L_shape`, `donut` — fully known, so `frontier_gain = 0` → pure coverage +
distance scoring) and 2 asset maps (`lab_05`, `lab_ghent`). Compute-time numbers
come from a clean, sequential, idle-machine run (parallel runs contaminate
`plan_ms`; see note at the end). Metrics/images referenced here live in this same
directory. Reproduce any row with:

```
python tests/algo_evaluation/benchmark_waypoint_scoring.py --variant <name>
python tests/algo_evaluation/benchmark_waypoint_scoring.py --compare   # full table
```

---

## The scoring formula

Candidates (viewpoints) are picked greedily to maximise a utility-density score:

```
score(c) = (alpha·frontier_gain + beta·coverage_gain) · distance_factor(bfs_dist, gamma, norm)
```

`distance_factor` shrinks the score with travel distance. The whole
investigation is about the *shape* and *sourcing* of that distance term. The
factor is chosen by `distance_model` (registry `DISTANCE_MODELS`):

| model | factor | notes |
|---|---|---|
| `linear` (baseline) | `1 / (1 + gamma·d/norm)` | ×0.50 at one max-range; slow decay |
| `expdecay` | `exp(-gamma·d/norm)` | ×0.37 at one max-range; sharp decay |
| `gaussian` | `exp(-(gamma·d/norm)²)` | flat near robot, hard cutoff far |

---

## The three problems we set out to fix

- **A — myopic distance.** `dist_from_robot` is computed once from the robot's
  start. Greedy scores *every* pick against that stale distance, so only the
  first pick is truly distance-aware; the rest form a scattered set the router
  then has to untangle.
- **B — weak far-penalty.** `linear` decays too slowly: a candidate three
  max-ranges away still keeps ~25 % of its score, so far candidates leak into
  the selection and inflate path length / redundancy.
- **C — objective/metric mismatch.** The planner maximises gain-per-scored-
  distance; the run is graded on coverage-per-metre-*travelled*. C is not a
  separate code change — it is *diagnosed by* `coverage_per_metre` and closed
  when A/B make the planner honour travel cost.

---

## Variants tested

| variant | change | user knobs |
|---|---|---|
| `baseline` | `linear`, γ=1 (current production) | — |
| `A_sequential` | fix A: re-seed distance from each pick (one BFS per pick) | — |
| `B_expdecay` | fix B: `expdecay`, γ=1 | — |
| `AB_sequential_expdecay` | A + B combined | — |
| `B_expdecay_g069` | `expdecay`, γ=0.69 (×0.50 at max-range) | γ |
| `B_expdecay_g050` | `expdecay`, γ=0.50 | γ |
| `B_gaussian` | `gaussian` shape, γ=1 | — |
| `B_expdecay_floor` | `expdecay` + marginal-gain floor 0.15 (lever 4) | — |

γ and `max_waypoints` are **user-deployment inputs**, so variants that only win
by pinning them are noted as such and not treated as model improvements.

---

## Headline result

`B_expdecay` (γ=1) is the winner. On the flagship real map `lab_05` it cuts the
executed path by a third at coverage parity and near-zero extra compute; every
attempt to "improve" it (γ-tuning, gaussian shape, gain floor) made it worse.

### lab_05 — the map where scoring matters most

| variant | path m | Δpath | coverage | cov/m | plan ms (avg) | plan ms (max) | waypoints |
|---|---|---|---|---|---|---|---|
| baseline | 72.5 | — | 0.968 | 0.0134 | 2419 | 4853 | 12 |
| **B_expdecay** | **47.9** | **−34 %** | 0.957 | **0.0200 (+49 %)** | **2626 (+9 %)** | 4890 | 13 |
| A_sequential | 82.1 | +13 % ✗ | 0.956 | 0.0117 | 4004 (+66 %) | 8425 | 13 |
| AB_seq_expdecay | 74.0 | +2 % ✗ | 0.967 | 0.0131 | 4870 (+101 %) | 11221 | 14 |
| B_expdecay_g069 | 73.7 | +2 % ✗ | 0.958 | 0.0130 | 2242 | 5189 | 13 |
| B_gaussian | 78.4 | +8 % ✗ | 0.975 | 0.0124 | 2112 | 4814 | 15 |
| B_expdecay_floor | 79.5 | +10 % ✗ | 0.965 | 0.0121 | 1786 | 3599 | 13 |

### lab_ghent — the hard, coverage-ceiling-limited map

| variant | path m | Δpath | coverage | cov/m | plan ms (avg) | plan ms (max) | waypoints |
|---|---|---|---|---|---|---|---|
| baseline | 114.2 | — | 0.860 | 0.0075 | 7037 | 12834 | 20 |
| **B_expdecay** | 116.3 | +2 % (neutral) | 0.860 | 0.0074 | **5495 (−22 %)** | 12863 | 23 |
| A_sequential | 142.2 | +25 % ✗ | 0.871 | 0.0061 | 12461 (+77 %) | 22860 | 23 |
| B_expdecay_g069 | 130.8 | +15 % ✗ | 0.843 | 0.0064 | 5952 | 12765 | 19 |
| B_gaussian | 142.5 | +25 % ✗ | 0.836 | 0.0059 | 6176 | 13107 | 30 |
| B_expdecay_floor | 153.1 | +34 % ✗ | 0.869 | 0.0057 | 5261 | 8115 | 22 |

### Synthetic maps — B_expdecay vs baseline (path length)

| map | baseline | B_expdecay | Δpath | cov/m Δ |
|---|---|---|---|---|
| large_rectangle | 53.0 | 44.6 | −16 % | +22 % |
| L_shape | 32.0 | 26.9 | −16 % | +18 % |
| donut | 55.6 | 45.5 | −18 % | +21 % |

B_expdecay reduces path 16–34 % and raises coverage-per-metre 18–49 % on 4 of 5
maps, with flat coverage. lab_ghent is neutral (ceiling-limited, see below).

---

## Why A fails

`A_sequential` re-seeds the distance from each selected waypoint so selection
follows the growing tour. In principle this attacks problem A. In practice it
**loses**:

- Path **+13 % (lab_05), +25 % (lab_ghent)** — worse, not better.
- `path_efficiency_ratio` (executed / NN-lower-bound) went *up* on 4/5 maps.
- Compute **+66 % to +77 %** mean, up to **+100 %** — one full-grid BFS per pick.

**Root cause:** the existing pipeline already routes the selected set with
`nearest_neighbor_order` (a BFS nearest-neighbour TSP). A turns *selection* into
a second nearest-neighbour chain, so two NN heuristics fight: greedy-by-marginal-
gain-over-distance picks a **different, worse set**, which the router then still
has to order. The waypoint visualisations for A show *more* criss-crossing than
baseline, not less (`waypoints_lab_05_A_sequential.png`). Problem A, as framed,
was not the real bottleneck — the router was already absorbing it.

`AB_sequential_expdecay` confirms this: A **cancels** B's benefit. On lab_05, B
alone gets −34 % path; A+B gets only +2 %. Sequential selection fights the router
regardless of the distance model.

---

## Why B wins

`B_expdecay` swaps the slow `linear` decay for `exp(-gamma·d/norm)`. The sharper
penalty stops far candidates leaking into the selection, so greedy picks a
tighter, better-clustered set that the existing router turns into a clean loop
(compare `waypoints_lab_05_baseline.png`'s criss-crossing to
`waypoints_lab_05_B_expdecay.png`'s single sweep). This directly resolves
problem C: `coverage_per_metre` rises 18–49 % even though final coverage is
unchanged.

**Compute cost is near-zero.** The expdecay formula itself is free. B's only
extra cost is that it sometimes selects one more waypoint, which adds one routing
BFS (+9 % on lab_05; on lab_ghent B is actually **−22 %** because it plans in
fewer cycles). The much larger +25–58 % figures seen earlier were measurement
contamination from parallel benchmark runs, corrected by the clean sequential
re-measure.

---

## Why the "improvements" to B all fail

Every attempt to keep B's path win at lower compute made it worse. This matters:
it shows B's win is not a fragile artifact of one setting.

- **γ down (`g069`, `g050`) — the win is coupled to the cost.** Lowering γ
  softens the penalty back toward `linear`, so selection density (and compute)
  returns to baseline — **and so do the paths.** On lab_05, g069 gives +2 % path
  (worse than baseline); on large_rectangle g050 collapses to *exactly* baseline
  (53.0 m). There is no γ that keeps the path win at low compute; the sharp
  penalty *is* why B is both better and slightly costlier.
- **`gaussian` shape (lever 3) — trades, doesn't improve.** Flat-near / hard-far
  matches expdecay on synthetic maps at low compute, but on cluttered real maps
  the flat-near region causes over-selection (lab_ghent: 30 waypoints vs 20) and
  path **+8 % (lab_05), +25 % (lab_ghent)**. Worse where it matters.
- **marginal-gain floor (lever 4, `B_expdecay_floor`) — fragments routing.**
  Cutting low-gain tail picks per plan makes the robot finish fewer waypoints per
  cycle and **re-plan more often**; each re-plan re-selects scattered leftovers
  from the robot's new position, so the global tour criss-crosses. Path
  **+10 % (lab_05), +34 % (lab_ghent)** — worst of all variants. Same failure
  class as A: interfering with a selection/routing pipeline that already works.

**Conclusion:** `B_expdecay` at γ=1 is robust — no structural lever improves it,
and its path win and (small) compute cost are intrinsically coupled.

---

## Why lab_ghent resists every variant

No scoring variant meaningfully shortens lab_ghent's path or raises its 86 %
coverage. This is **not** a scoring limitation:

- **Geometric ceiling ≈ 92.6 %.** The candidate viewpoints can only see 92.6 %
  of free space; the rest is behind walls/clutter — uncoverable by any scoring.
- **Navigable space is 94 % one connected region** — not an isolation problem;
  the robot can reach almost everywhere.
- **Uncovered cells are 1 500+ tiny scattered slivers** along walls, behind
  shelf-rows, and in diagonal clutter — each below the marginal gain needed to
  earn a dedicated stop.
- **baseline and B leave the *identical* uncovered pattern**
  (`waypoints_lab_ghent_baseline.png` vs `..._B_expdecay.png`) — proof the gap is
  structural, not a scoring choice.

The real limit is the gap between the planning abstraction (360° "achievable")
and execution (87° camera FOV at a finite heading set). Closing it needs work on
**candidate generation** and/or **heading selection**, not scoring.

---

## Can B's compute time be improved? Yes — but not in the scoring

A `cProfile` of a B_expdecay lab_05 run:

| function | cumulative | calls | note |
|---|---|---|---|
| `plan_waypoints_raw` | 13.8 s | 4 | — |
| **`bfs_distance_map`** | **13.8 s** | **37** | **~100 % of plan time** |
| `nearest_neighbor_order` | 9.0 s | 3 | one routing BFS per waypoint |

`bfs_distance_map` is a pure-Python `deque` BFS over the whole 280×649 grid:
**~794 ms per call.** The scoring model contributes nothing measurable. B is
marginally slower than baseline only because it sometimes routes one more
waypoint = one more BFS.

**The real optimization is `bfs_distance_map` itself** (e.g.
`scipy.sparse.csgraph.dijkstra` on the grid graph — available, ~5–10× faster).
That speeds up **baseline and every variant equally**, so it is a planner-wide
performance task, *independent of this scoring branch* — it should not be bundled
into the scoring decision.

> **DONE (2026-07-14).** Implemented on a separate branch: `bfs_distance_map` →
> `navigable_distance_map`, a √2-weighted `scipy.sparse.csgraph.dijkstra` (cached grid
> graph). Measured ~15× faster on the big real maps (lab_ghent 247→17 ms,
> warehouse_amazon 291→19 ms). Also corrects the diagonal-cost bias (cardinal=1,
> diagonal=√2). See `docs/TODO_exploration_refactor.md` §1 and
> `tests/algo_evaluation/bfs_metric_impact.ipynb`.

---

## Recommendation

Adopt **`distance_model = expdecay`, γ = 1** as the production default; keep
`linear` selectable. It delivers a 16–34 % path reduction and 18–49 % higher
coverage-per-metre on 4/5 maps, at coverage parity and ≤ +9 % plan time
(often faster). lab_ghent is neutral and ceiling-limited regardless. `A`,
`gaussian` and the gain-floor are documented negative results, retained as
selectable variants for the record.

Separately, optimising `bfs_distance_map` is the highest-value planner-wide
performance follow-up.

---

## Artifacts (this directory)

- `metrics_<variant>.json` — per-map metrics (20 metrics each; run `--compare`
  for the full aligned table).
- `waypoints_<map>_<variant>.png` — visited waypoints in visit order per
  variant/map.

### Note on compute-time measurement

`plan_ms` is wall-clock and is inflated when multiple benchmark processes run
concurrently. All compute numbers in this report are from **sequential, idle-
machine** runs. Path / coverage / routing metrics are deterministic and
unaffected by concurrency.
