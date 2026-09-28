"""
Post-run ideal waypoint annotation tool.

After running visual_demo_coverage.py, use this tool to define the ideal
robot path by clicking on the map. The result is saved as ideal_waypoints.csv
in the same log directory, for offline comparison with the actual plan.

Run:
    python tests/annotate_ideal.py                   # auto-detects latest log dir
    python tests/annotate_ideal.py path/to/log_dir   # explicit log directory

Controls:
    Left-click   Add next ideal waypoint (snapped to nearest navigable cell)
    Right-click  Remove last ideal waypoint
    z            Undo last waypoint
    Enter        Save and exit
    Close window Save and exit
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # tests/ — for conftest constants
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "exploration"))  # dir containing the `exploration` package

from exploration.explore_costmap_map import load_map, pixel_to_world
from conftest import REFERENCE_MAP_PGM, REFERENCE_MAP_YAML

# ── Constants ────────────────────────────────────────────────────────────────
SNAP_RADIUS_PX   = 20       # max pixel distance to snap click to navigable cell
INFLATION_M      = 0.3      # must match the demo

_TESTS_DIR = Path(__file__).parent
MAP_PGM    = REFERENCE_MAP_PGM
MAP_YAML   = REFERENCE_MAP_YAML
LOG_BASE   = _TESTS_DIR / "visual_demo_coverage"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _latest_log_dir(base: Path) -> Path | None:
    """Return the most recently modified coverage_demo_log* directory."""
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("coverage_demo_log")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _build_navigable_index(navigable_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) arrays of all navigable pixel positions."""
    rows, cols = np.where(navigable_mask)
    return rows, cols


def _snap_to_navigable(
    click_col: float,
    click_row: float,
    nav_rows: np.ndarray,
    nav_cols: np.ndarray,
) -> tuple[int, int] | None:
    """Find the nearest navigable pixel within SNAP_RADIUS_PX. Returns (col, row) or None."""
    if nav_rows.size == 0:
        return None
    dists = np.hypot(nav_cols - click_col, nav_rows - click_row)
    idx = int(np.argmin(dists))
    if dists[idx] > SNAP_RADIUS_PX:
        return None
    return int(nav_cols[idx]), int(nav_rows[idx])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Resolve log directory ─────────────────────────────────────────────
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
    else:
        log_dir = _latest_log_dir(LOG_BASE)
        if log_dir is None:
            print("No coverage_demo_log* directory found. Run visual_demo_coverage.py first.")
            sys.exit(1)
        print(f"Using log directory: {log_dir}")

    output_path = log_dir / "ideal_waypoints.csv"

    # ── Load map ──────────────────────────────────────────────────────────
    md = load_map(MAP_PGM, MAP_YAML, INFLATION_M)
    H, W = md.pgm_array.shape
    nav_rows, nav_cols = _build_navigable_index(md.navigable_mask)

    # ── Load robot trajectory and waypoints from CSVs ────────────────────
    motion_rows  = _load_csv(log_dir / "robot_motion.csv")
    waypoint_rows = _load_csv(log_dir / "waypoints.csv")

    traj_cols = [float(r["col"]) for r in motion_rows if r.get("col")]
    traj_rows = [float(r["row"]) for r in motion_rows if r.get("row")]

    # Collect actual planned waypoints (rank=0 = visited) per plan
    actual_wps = [(float(r["col"]), float(r["row"]), int(r["plan_id"]), int(r["rank"]))
                  for r in waypoint_rows if r.get("col")]

    # ── Set up figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title("Annotate Ideal Waypoints")

    ax.imshow(md.pgm_array, cmap="gray", origin="upper")

    # Navigable overlay (light blue, translucent)
    nav_overlay = np.zeros((H, W, 4), dtype=np.float32)
    nav_overlay[md.navigable_mask] = [0.4, 0.6, 1.0, 0.15]
    ax.imshow(nav_overlay, origin="upper")

    # Robot trajectory
    if traj_cols:
        ax.plot(traj_cols, traj_rows, color="0.5", linewidth=0.8,
                alpha=0.7, label="Robot trajectory")

    # Actual waypoints (small blue dots, labelled with plan_id.rank)
    for c, r, pid, rnk in actual_wps:
        ax.plot(c, r, "b.", markersize=5, alpha=0.5)
        ax.text(c + 2, r - 2, f"{pid}.{rnk}", fontsize=5, color="blue", alpha=0.6)

    ax.set_title(
        "Left-click: add ideal waypoint  |  Right-click / Z: undo  |  Enter / close: save",
        fontsize=9,
    )
    ax.axis("off")

    # ── State ─────────────────────────────────────────────────────────────
    ideal: list[tuple[int, int]] = []   # (col, row) in click order
    markers: list = []                  # matplotlib artists for each waypoint

    def _redraw() -> None:
        for m in markers:
            for artist in m:
                artist.remove()
        markers.clear()
        for i, (c, r) in enumerate(ideal, start=1):
            star, = ax.plot(c, r, "r*", markersize=12)
            lbl   = ax.text(c + 3, r - 4, str(i), fontsize=8,
                            color="red", fontweight="bold")
            markers.append((star, lbl))
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 1:          # left-click → add
            snapped = _snap_to_navigable(event.xdata, event.ydata, nav_rows, nav_cols)
            if snapped is None:
                print("Click is too far from a navigable cell — move closer to the floor.")
                return
            ideal.append(snapped)
            print(f"  Waypoint {len(ideal)}: pixel ({snapped[0]}, {snapped[1]})")
            _redraw()
        elif event.button == 3:        # right-click → undo
            _undo()

    def _undo() -> None:
        if ideal:
            removed = ideal.pop()
            print(f"  Removed waypoint at pixel {removed}")
            _redraw()

    def _on_key(event) -> None:
        if event.key in ("z", "Z"):
            _undo()
        elif event.key == "enter":
            _save_and_close()

    def _save_and_close() -> None:
        _save()
        plt.close(fig)

    def _on_close(_event) -> None:
        _save()

    def _save() -> None:
        if not ideal:
            print("No ideal waypoints defined — nothing saved.")
            return
        rows = []
        for rank, (col, row) in enumerate(ideal, start=1):
            x_m, y_m = pixel_to_world(col, row, md.resolution, md.origin_x, md.origin_y, H)
            rows.append({
                "rank":  rank,
                "col":   col,
                "row":   row,
                "x_m":   round(x_m, 3),
                "y_m":   round(y_m, 3),
            })
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(ideal)} ideal waypoints saved → {output_path}")

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)

    print("\nAnnotation tool ready.")
    print("  Left-click  → add ideal waypoint (snapped to navigable floor)")
    print("  Right-click → remove last waypoint")
    print("  Z           → undo")
    print("  Enter / close window → save and exit\n")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
