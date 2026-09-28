"""
Real-time matplotlib visualiser for the exploration planner.

Four display layers (bottom to top):
    1. Occupancy grid    — white=free, black=wall, grey=unknown
    2. covered_mask      — semi-transparent green overlay
    3. Planned waypoints — blue numbered dots connected by a dashed path
    4. Robot state       — orange trajectory line + red arrow for current heading

Usage (non-blocking, in a simulation loop):

    from visualisation.exploration_visualiser import ExplorationVisualiser
    import matplotlib.pyplot as plt

    vis = ExplorationVisualiser(map_data)
    vis.show_nonblocking()

    while not done:
        waypoints, ratio, no_frontiers = plan_waypoints(map_data, config, rx, ry)
        for wp in waypoints:
            for heading in wp.headings:
                update_covered_mask(map_data, wp.col, wp.row, heading, 87.0, max_range_px)
                trajectory.append((wp.col, wp.row))
                vis.update(waypoints, wp.col, wp.row, heading, trajectory, ratio)
                plt.pause(0.05)
            robot_x, robot_y = wp.x, wp.y

Usage (blocking, standalone inspection):

    vis = ExplorationVisualiser(map_data)
    vis.update(waypoints, robot_col, robot_row, 0.0, [])
    vis.show()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from navigation.exploration.explore_costmap_map import MapData, Waypoint


class ExplorationVisualiser:
    """Real-time exploration progress visualiser."""

    def __init__(self, map_data: MapData) -> None:
        self._map_data = map_data
        self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._fig.tight_layout()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        waypoints: list[Waypoint],
        robot_col: int,
        robot_row: int,
        robot_heading_deg: float,
        trajectory: list[tuple[int, int]],
        coverage_pct: float | None = None,
    ) -> None:
        """Refresh all layers with the current exploration state.

        Call plt.pause(dt) after this to drive the event loop when running
        non-blocking.

        Args:
            waypoints:         Current planned waypoints (blue dots).
            robot_col/row:     Robot pixel position.
            robot_heading_deg: Current camera heading (degrees, 0=East, 90=North).
            trajectory:        List of (col, row) positions visited so far.
            coverage_pct:      Coverage ratio [0, 1] shown in the title (optional).
        """
        ax = self._ax
        ax.cla()

        H, W = self._map_data.pgm_array.shape

        # -- Layer 1: occupancy grid ----------------------------------------
        display = np.full((H, W, 3), 0.65)              # grey = unknown
        display[self._map_data.free_mask] = 1.0          # white = free
        display[self._map_data.occupied_mask] = 0.0      # black = wall
        ax.imshow(display, origin="upper", zorder=0)

        # -- Layer 2: covered_mask (green overlay) --------------------------
        green = np.zeros((H, W, 4), dtype=np.float32)
        green[self._map_data.covered_mask] = [0.0, 0.75, 0.0, 0.4]
        ax.imshow(green, origin="upper", zorder=1)

        # -- Layer 3: planned waypoints (blue numbered dots + dashed path) --
        if waypoints:
            wp_cols = [wp.col for wp in waypoints]
            wp_rows = [wp.row for wp in waypoints]
            ax.plot(wp_cols, wp_rows, "--", color="steelblue", linewidth=0.8,
                    alpha=0.6, zorder=2)
            for i, wp in enumerate(waypoints):
                ax.plot(wp.col, wp.row, "o", color="steelblue", markersize=7, zorder=3)
                ax.text(
                    wp.col + 4, wp.row - 4, str(i + 1),
                    color="steelblue", fontsize=7, fontweight="bold", zorder=4,
                )

        # -- Layer 4a: trajectory (orange line) ----------------------------
        if len(trajectory) > 1:
            t_cols = [p[0] for p in trajectory]
            t_rows = [p[1] for p in trajectory]
            ax.plot(t_cols, t_rows, "-", color="darkorange", linewidth=1.5,
                    alpha=0.8, zorder=2)

        # -- Layer 4b: robot pose (red square + heading arrow) -------------
        arrow_len = max(W, H) * 0.035
        heading_rad = np.radians(robot_heading_deg)
        dx = arrow_len * np.cos(heading_rad)
        dy = -arrow_len * np.sin(heading_rad)   # -sin: image row↓, world y↑
        ax.annotate(
            "",
            xy=(robot_col + dx, robot_row + dy),
            xytext=(robot_col, robot_row),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
            zorder=5,
        )
        ax.plot(robot_col, robot_row, "s", color="red", markersize=9, zorder=5)

        # -- Title and labels ----------------------------------------------
        title = "Exploration Progress"
        if coverage_pct is not None:
            title += f"  —  Coverage: {coverage_pct:.1%}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("col (px)")
        ax.set_ylabel("row (px)")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        self._fig.canvas.draw_idle()

    def show_nonblocking(self) -> None:
        """Open the figure window without blocking the caller.

        Drive the event loop with plt.pause(dt) inside your simulation loop.
        """
        plt.ion()
        self._fig.show()

    def show(self) -> None:
        """Display the figure and block until the window is closed."""
        plt.ioff()
        plt.show()

    def save(self, path: str | Path) -> None:
        """Save the current frame as a PNG file."""
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
