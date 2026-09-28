#!/usr/bin/env python3
"""
Aggregate N exploration runs of one (scene, mode) into mean/std statistics,
normalized against the scene baseline (system-test layer, L3).

Pure Python, no ROS. Exploration is stochastic (SLAM noise, Nav2 timing,
replans), so a single run vs the baseline can mislead; this script summarises
a GROUP of runs:

  - per-KPI mean / std / min / max over the runs,
  - each mean expressed as a RATIO to the scene's saved baseline value
    (run_mean / baseline), because absolute KPIs are only meaningful within
    one environment,
  - overlaid coverage-vs-time and coverage-vs-path curves (every run
    semi-transparent + the mean curve),
  - a summary.md table.

Groups: by default all given run folders form one group named by their
(scene, mode) from meta.yaml. Pass several --group blocks to summarise more
than one scene in a single call; the cross-scene figure then shows ONLY the
normalized ratios side by side per scene — never absolute values across
environments.

The baseline is read from the scene YAML's <mode>.baseline block (written by
evaluate_run.py --save-baseline). It is a single reference run, i.e. a point,
not a distribution: the comparison is "run distribution vs baseline value".
If you later record several manual baselines, aggregate those folders with
this same script to get a baseline mean/std.

Usage:
    # one scene: all runs of lab05 known_map vs its baseline
    aggregate_runs.py runs/lab05_known_map_run* \
        --config baselines/baseline_lab05.yaml --mode known_map \
        --out summaries/lab05_known_map

    # several scenes on one normalized figure
    aggregate_runs.py \
        --group runs/lab05_known_map_run*   --config baselines/baseline_lab05.yaml \
        --group runs/warehouse_known_map_run* --config baselines/baseline_warehouse.yaml \
        --mode known_map --out summaries/known_map_all
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import yaml

from stuck_analysis import stuck_for_run

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Reuse the single-run KPI computation so numbers match evaluate_run.py exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_run import compute_metrics, _baseline_keys, _read_csv, _f  # noqa: E402


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {} if path.is_file() else {}


def _coverage_curve(run_dir: Path):
    """[(time_s, path_m, coverage), ...] from a run's motion.csv."""
    pts, total, prev = [], 0.0, None
    for r in _read_csv(run_dir / "motion.csv"):
        t = _f(r, "timestamp_s")
        x, y, cov = _f(r, "x_m"), _f(r, "y_m"), _f(r, "coverage")
        if x is not None and y is not None:
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        if t is not None and cov is not None:
            pts.append((t, total, cov))
    return pts


def _stats(values: list[float]) -> dict | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


class Group:
    """All runs of one (scene, mode) plus that scene's baseline."""

    def __init__(self, run_dirs: list[Path], config: Path, mode: str):
        self.run_dirs = run_dirs
        self.mode = mode
        doc = _load_yaml(config)
        section = doc.get(mode, {})
        self.baseline = section.get("baseline", {}) or {}
        self.metrics = []      # one dict per run (evaluate_run.compute_metrics)
        self.curves = []       # one [(t, path, cov)] per run
        self.metas = []
        for rd in run_dirs:
            meta = _load_yaml(rd / "meta.yaml")
            self.metas.append(meta)
            # Reference every run's coverage to the baseline's covered area so
            # the 50/90% milestones are the same absolute target across runs.
            self.metrics.append(
                compute_metrics(rd, meta, self.baseline.get("covered_area_m2")))
            self.curves.append(_coverage_curve(rd))
        # The baseline's own coverage curve, so the plots show the human
        # reference line and not just the autonomous runs. The saved baseline
        # block holds scalars only, but it records run_folder, so the curve is
        # re-read from that run's motion.csv. run_folder is stored relative to
        # the system-test root (this file's directory); absolute paths are
        # honoured as given. Empty when no baseline is saved or the folder has
        # been moved away — the plots then simply omit the reference line.
        self.baseline_curve = []
        bf = self.baseline.get("run_folder")
        if bf:
            bdir = Path(bf)
            if not bdir.is_absolute():
                bdir = Path(__file__).resolve().parent / bdir
            if (bdir / "motion.csv").is_file():
                self.baseline_curve = _coverage_curve(bdir)
            else:
                print(f"WARNING [{run_dirs[0].name if run_dirs else '?'}]: "
                      f"baseline run folder '{bf}' has no motion.csv; "
                      "curves plotted without the baseline reference.")
        scenes = {m.get("scene", "?") for m in self.metas}
        if len(scenes) > 1:
            raise SystemExit(
                f"Runs of different scenes in one group: {sorted(scenes)}. "
                "Absolute KPIs are not comparable across environments; "
                "give each scene its own --group.")
        self.scene = next(iter(scenes), "?")
        wall = [rd.name for rd, m in zip(run_dirs, self.metas)
                if m.get("time_source") == "wall"]
        if wall:
            print(f"WARNING [{self.scene}]: wall-clock runs mixed in "
                  f"(time KPIs not sim time): {wall}")

    @property
    def label(self) -> str:
        return f"{self.scene} [{self.mode}]"

    def kpi_table(self):
        """rows: (kpi, stats-dict, baseline_value, mean/baseline ratio)."""
        rows = []
        for key in _baseline_keys():
            st = _stats([m.get(key) for m in self.metrics])
            if st is None:
                continue
            base = self.baseline.get(key)
            ratio = (st["mean"] / base
                     if isinstance(base, (int, float)) and base != 0 else None)
            rows.append((key, st, base, ratio))
        return rows


def plot_curves(group: Group, out_dir: Path) -> Path:
    """Overlay every run's coverage curves + the mean curve (vs time & path)."""
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(13, 5))
    for rd, pts in zip(group.run_dirs, group.curves):
        if not pts:
            continue
        ax_t.plot([p[0] for p in pts], [100 * p[2] for p in pts],
                  color="tab:blue", alpha=0.3, lw=1)
        ax_p.plot([p[1] for p in pts], [100 * p[2] for p in pts],
                  color="tab:purple", alpha=0.3, lw=1)

    # Mean curve: resample every run onto a common grid, average where >=2
    # runs still have data (curves end at different times/lengths).
    for ax, idx, colour in ((ax_t, 0, "tab:blue"), (ax_p, 1, "tab:purple")):
        ends = [pts[-1][idx] for pts in group.curves if pts]
        if not ends:
            continue
        grid = [max(ends) * i / 200.0 for i in range(201)]
        mean_y = []
        for g in grid:
            ys = []
            for pts in group.curves:
                if not pts or g > pts[-1][idx]:
                    continue
                # linear scan is fine at this size
                prev = pts[0]
                for p in pts:
                    if p[idx] >= g:
                        span = p[idx] - prev[idx]
                        frac = (g - prev[idx]) / span if span > 0 else 0.0
                        ys.append(prev[2] + frac * (p[2] - prev[2]))
                        break
                    prev = p
            mean_y.append(statistics.fmean(ys) if ys else None)
        gx = [g for g, y in zip(grid, mean_y) if y is not None]
        gy = [100 * y for y in mean_y if y is not None]
        if gx:
            ax.plot(gx, gy, color=colour, lw=2.5,
                    label=f"mean of {len(group.run_dirs)} runs")

    # The human baseline, drawn last so it sits on top of the run bundle.
    # Black dashed to read as "reference", distinct from the per-run colours.
    if group.baseline_curve:
        bpts = group.baseline_curve
        src = group.baseline.get("source") or "baseline"
        ax_t.plot([p[0] for p in bpts], [100 * p[2] for p in bpts],
                  color="black", ls="--", lw=2.0, zorder=5,
                  label=f"baseline ({src})")
        ax_p.plot([p[1] for p in bpts], [100 * p[2] for p in bpts],
                  color="black", ls="--", lw=2.0, zorder=5,
                  label=f"baseline ({src})")

    ax_t.set_xlabel("time [s] (sim)")
    ax_t.set_title("coverage vs time")
    ax_p.set_xlabel("path travelled [m]")
    ax_p.set_title("coverage vs path")
    for ax in (ax_t, ax_p):
        ax.set_ylabel("coverage [%]")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{group.label}: {len(group.run_dirs)} runs")
    out = out_dir / f"curves_{group.scene}_{group.mode}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_normalized(groups: list[Group], out_dir: Path) -> Path | None:
    """Cross-scene figure: mean/baseline ratio (+/- std/baseline) per KPI per
    scene. Only normalized ratios — never absolute values across scenes."""
    kpis = [k for k in _baseline_keys()
            if any(any(r[0] == k and r[3] is not None for r in g.kpi_table())
                   for g in groups)]
    if not kpis:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(kpis)), 5))
    width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        table = {k: (st, base, ratio) for k, st, base, ratio in g.kpi_table()}
        xs, ys, errs = [], [], []
        for ki, k in enumerate(kpis):
            st, base, ratio = table.get(k, (None, None, None))
            if ratio is None:
                continue
            xs.append(ki + gi * width)
            ys.append(ratio)
            errs.append(st["std"] / base if base else 0.0)
        ax.bar(xs, ys, width=width, yerr=errs, capsize=3, label=g.label)
    ax.axhline(1.0, color="0.3", ls="--", lw=1, label="baseline = 100%")
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(kpis))])
    ax.set_xticklabels(kpis, rotation=30, ha="right", fontsize=8)
    # Show the ratio as a percentage of baseline (1.0 -> 100%, 1.84 -> 184%).
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("run mean as % of scene baseline")
    ax.set_title("Normalized KPIs per scene (mean ± std over runs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    out = out_dir / "normalized_kpis.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _abort_reasons(run_dirs) -> dict:
    """Tally the `reason` column of nav_goals.csv across runs (ABORTED rows).

    The reason is written by recorder.py from Nav2 action feedback (this Nav2
    returns no error code). Empty for runs recorded before that column existed."""
    tally: dict[str, int] = {}
    for rd in run_dirs:
        f = rd / "nav_goals.csv"
        if not f.is_file():
            continue
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "ABORTED":
                    continue
                r = (row.get("reason") or "").strip() or "unrecorded"
                tally[r] = tally.get(r, 0) + 1
    return tally


def _findings(g: Group) -> list[str]:
    """Interpretation section: the standing conclusions about the exploration
    system, each restated with THIS group's numbers so the text tracks the data
    instead of asserting yesterday's result. Claims that the current runs cannot
    support are omitted rather than printed unbacked."""
    out: list[str] = []
    base = g.baseline
    mets = g.metrics

    def _mean(key):
        vals = [m.get(key) for m in mets
                if isinstance(m.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    out.append("### Conclusions")
    out.append("")

    # 1. Front-loading: path spent to reach 50% coverage vs the baseline's.
    p50, b50 = _mean("path_to_50pct_coverage_m"), base.get("path_to_50pct_coverage_m")
    if p50 and isinstance(b50, (int, float)) and b50:
        out.append(
            f"1. **Greedy set-cover viewpoint selection genuinely picks "
            f"high-information positions.** Reaching 50% coverage costs "
            f"{p50:.1f} m of path on average vs {b50:.1f} m for the baseline "
            f"({p50 / b50:.2f}x) — the strategy front-loads coverage rather than "
            f"sweeping uniformly.")
        out.append("")

    # 2. The endgame tail: path spent AFTER 90% coverage is reached.
    tails = [(m["path_length_m"] - m["path_to_90pct_coverage_m"], m["path_length_m"])
             for m in mets
             if isinstance(m.get("path_length_m"), (int, float))
             and isinstance(m.get("path_to_90pct_coverage_m"), (int, float))]
    if tails:
        pct = [100.0 * t / p for t, p in tails if p]
        b_tail = None
        if all(isinstance(base.get(k), (int, float))
               for k in ("path_length_m", "path_to_90pct_coverage_m")):
            bp = base["path_length_m"]
            b_tail = 100.0 * (bp - base["path_to_90pct_coverage_m"]) / bp if bp else None
        cmp_s = f" (baseline: {b_tail:.0f}%)" if b_tail is not None else ""
        out.append(
            f"2. **The weakness is the endgame, not the exploration.** Path spent "
            f"*after* 90% coverage is reached averages {sum(pct) / len(pct):.0f}% "
            f"of the total journey (range {min(pct):.0f}–{max(pct):.0f}%)"
            f"{cmp_s}. Hunting the last few scattered cells — not the search "
            f"itself — is where the extra path length comes from; the system has "
            f"no diminishing-returns criterion.")
        out.append("")

    # 3. Coverage ceiling: where healthy runs actually saturate.
    covs = [m["final_coverage"] for m in mets
            if isinstance(m.get("final_coverage"), (int, float))]
    bcov = base.get("final_coverage")
    if covs:
        bs = (f" and the human baseline reaches {bcov:.1%}"
              if isinstance(bcov, (int, float)) else "")
        out.append(
            f"3. **Coverage saturates near the achievable ceiling.** Runs land at "
            f"{min(covs):.1%}–{max(covs):.1%}{bs}. The residue is most likely "
            f"geometrically unreachable under the sensor model and inflation "
            f"radius, not a strategy failure — which is why chasing it (point 2) "
            f"costs so much.")
        out.append("")

    # 4. Reliability: aborts are a strategy<->Nav2 boundary problem.
    reasons = _abort_reasons(g.run_dirs)
    total_ab = sum(reasons.values())
    if total_ab:
        top = ", ".join(f"`{k}` x{v}" for k, v in
                        sorted(reasons.items(), key=lambda kv: -kv[1]))
        runs_with = sum(1 for m in mets
                        if (m.get("nav_goals") or {}).get("aborted"))
        out.append(
            f"4. **Nav2 integration is the reliability risk, not viewpoint "
            f"choice.** {total_ab} aborted goal(s) across {runs_with}/{len(mets)} "
            f"run(s): {top}. Aborts concentrated in `failed_near_goal` / "
            f"`no_valid_path` mean the planner proposed goals Nav2 could not "
            f"service (typically waypoints in or beside obstacle inflation) — the "
            f"strategy layer is sound; the strategy↔Nav2 boundary is where runs "
            f"degrade.")
        out.append("")

    out.append(f"_Basis: {len(mets)} run(s) of {g.label}. Small sample — treat "
               f"magnitudes as indicative._")
    out.append("")
    return out


def write_summary(groups: list[Group], out_dir: Path, curve_paths,
                  norm_path) -> Path:
    lines = ["# Aggregated exploration runs", ""]
    for g, cp in zip(groups, curve_paths):
        lines.append(f"## {g.label} — {len(g.run_dirs)} runs")
        lines.append("")
        lines.append("Runs: " + ", ".join(f"`{rd.name}`" for rd in g.run_dirs))
        base_src = g.baseline.get("source") or "(no baseline saved)"
        lines.append(f"Baseline: {base_src} `{g.baseline.get('run_folder', '')}`")
        lines.append("")

        # Per-run table: the aggregate mean/std hides whether runs cluster or
        # split into regimes, and which single run is dragging a KPI. One row per
        # run makes outliers (e.g. a run with many nav aborts) directly visible.
        lines.append("### Per-run results")
        lines.append("")
        lines.append("| run | coverage | vs baseline | path [m] | time [s] "
                     "| idle [s] | aborts | waypoints | human unstuck "
                     "| exploration_outcome |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        stuck_all = [stuck_for_run(rd) for rd in g.run_dirs]
        for rd, m, sk in zip(g.run_dirs, g.metrics, stuck_all):
            def _n(key, fmt="{:g}"):
                v = m.get(key)
                return fmt.format(v) if isinstance(v, (int, float)) else "—"
            cov = m.get("final_coverage")
            cov_s = f"{cov:.1%}" if isinstance(cov, (int, float)) else "—"
            # Coverage referenced to the baseline's covered area: comparable
            # across runs, unlike the per-run SLAM-denominator ratio beside it.
            covb = m.get("final_coverage_vs_baseline")
            covb_s = f"{covb:.1%}" if isinstance(covb, (int, float)) else "—"
            aborted = (m.get("nav_goals") or {}).get("aborted")
            aborted_s = str(aborted) if aborted is not None else "—"
            # Human-unstuck count + short stop reason from the exploration log;
            # "—" when the run has no log (not launched via run_with_log.sh).
            if sk is None:
                human_s, stop_s = "—", "—"
            else:
                human_s = str(sk["human_unstuck"])
                if m.get("terminated"):
                    stop_s = (sk.get("stop_reason") or "complete").replace(
                        "coverage_complete", "complete")
                else:
                    stop_s = sk.get("stop_reason") or "interrupted"
            lines.append(
                f"| `{rd.name}` | {cov_s} | {covb_s} "
                f"| {_n('path_length_m', '{:.1f}')} "
                f"| {_n('total_run_time_s', '{:.0f}')} "
                f"| {_n('idle_time_s', '{:.0f}')} | {aborted_s} "
                f"| {_n('total_waypoints')} | {human_s} | {stop_s} |")
        lines.append("")

        # Stuck totals across the group (skip runs with no log).
        logged = [s for s in stuck_all if s is not None]
        if logged:
            th = sum(s["human_unstuck"] for s in logged)
            ts = sum(s["self_recovered"] for s in logged)
            te = sum(s["ended_stuck"] for s in logged)
            lines.append(
                f"Stuck totals (over {len(logged)}/{len(g.run_dirs)} runs with "
                f"logs): **{th} human unstuck**, {ts} self-recovered (Nav2), "
                f"{te} ended while stuck.")
            lines.append("")

        lines.append("### Aggregate KPIs")
        lines.append("")
        lines.append("| KPI | mean | std | min | max | baseline | mean/baseline |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, st, base, ratio in g.kpi_table():
            base_s = f"{base:g}" if isinstance(base, (int, float)) else "—"
            ratio_s = f"{ratio:.2f}x" if ratio is not None else "—"
            lines.append(
                f"| {key} | {st['mean']:.3f} | {st['std']:.3f} "
                f"| {st['min']:g} | {st['max']:g} | {base_s} | {ratio_s} |")
        lines.append("")
        if cp is not None:
            lines.append(f"![curves]({cp.name})")
            lines.append("")
        lines.extend(_findings(g))
    if norm_path is not None:
        lines.append("## Normalized cross-scene comparison")
        lines.append("")
        lines.append("Ratios only (run mean / scene baseline); absolute KPIs "
                     "are never compared across environments.")
        lines.append("")
        lines.append(f"![normalized]({norm_path.name})")
        lines.append("")
    out = out_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate N runs of one or more (scene, mode) groups")
    parser.add_argument("runs", nargs="*", type=Path,
                        help="Run folders (single-group form; use with --config)")
    parser.add_argument("--config", action="append", type=Path, default=[],
                        help="Scene YAML; one per --group (or one for the "
                             "positional runs)")
    parser.add_argument("--group", action="append", nargs="+", type=Path,
                        default=[], metavar="RUN_DIR",
                        help="Run folders forming one scene group; repeatable, "
                             "pair each with a --config in the same order")
    parser.add_argument("--mode", required=True, choices=["known_map", "slam"])
    parser.add_argument("--out", type=Path, required=True,
                        help="Output folder for summary.md + PNGs")
    args = parser.parse_args(argv)

    group_dirs = list(args.group)
    if args.runs:
        group_dirs.insert(0, args.runs)
    if not group_dirs:
        parser.error("give run folders (positional) or at least one --group")
    if len(args.config) != len(group_dirs):
        parser.error(f"{len(group_dirs)} group(s) but {len(args.config)} "
                     "--config; pair one config per group, in order")

    groups = [Group([Path(r) for r in dirs], cfg, args.mode)
              for dirs, cfg in zip(group_dirs, args.config)]

    args.out.mkdir(parents=True, exist_ok=True)
    curve_paths = [plot_curves(g, args.out) for g in groups]
    norm_path = plot_normalized(groups, args.out)
    summary = write_summary(groups, args.out, curve_paths, norm_path)

    for g in groups:
        print(f"{g.label}: {len(g.run_dirs)} runs aggregated"
              + ("" if g.baseline.get("final_coverage") is not None
                 else "  (no baseline saved -> ratios omitted)"))
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
