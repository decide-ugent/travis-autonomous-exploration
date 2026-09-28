# Exploration run report: lab05_slam_baseline_1

- scene: **lab05**  mode: **slam**
- run folder: `lab05_slam_baseline_1`
- time source: **sim** (sim = simulation clock, wall = laptop clock)

## Gates

| gate | measured | | threshold | result |
|---|---|:-:|---|---|
| final_coverage | 94.5% | >= | 90% | PASS |
| time_to_complete | 663s | <= | 1800s | PASS |
| nav_aborted | 11 | <= | 0 | FAIL |

## Measured KPIs

| metric | value |
|---|---|
| final_coverage | 0.9445 |
| covered_area_m2 | 145.4 |
| navigable_area_m2 | 152.56 |
| path_length_m | 61.03 |
| coverage_rate | 0.01548 |
| path_to_50pct_coverage_m | 1.5 |
| path_to_90pct_coverage_m | 52.97 |
| time_to_50pct_coverage_s | 19.2 |
| time_to_90pct_coverage_s | 590.2 |
| terminated | False |
| total_run_time_s | 663.2 |
| idle_time_s | 329.2 |
| n_planning_cycles | 0 |
| planning_time_mean_s | None |
| planning_time_std_s | None |
| total_waypoints | None |
| waypoints_per_m2 | None |

## Nav2 goal outcomes

- total: 15  succeeded: 4
- aborted: 11  canceled: 0  other: 0

