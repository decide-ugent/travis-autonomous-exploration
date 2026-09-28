"""
Extended real-time visualiser for the visual exploration demos.

Renders all map masks as overlays, object markers, robot state, and
optionally a live 3D semantic map side panel.

Layers (bottom to top):
  1. Occupancy base       — white=free, black=wall, grey=unknown
  2. Navigable mask       — light-blue semi-transparent overlay
  3. Covered mask         — green semi-transparent overlay
  4. Planned waypoints    — blue numbered dots + dashed path
  5. Trajectory           — orange line
  6. Robot pose           — red square + heading arrow
  7. Object markers       — coloured stars (★ detected / grey × undetected)
  8. Legend               — label colours in plot legend

Usage:
    from visualisation.semantic_exploration_visualiser import SemanticExplorationVisualiser
    import matplotlib.pyplot as plt

    vis = SemanticExplorationVisualiser(map_data, objects, show_3d=True)
    vis.show_nonblocking()

    while not done:
        ...
        vis.update(waypoints, col, row, heading, trajectory, coverage_pct, semantic_map)
        plt.pause(0.02)
    vis.show()
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root or from visualisation/
_ROOT = Path(__file__).parent.parent.parent.parent  # /ros2_ws/src/
_EXPLORATION_PKG = _ROOT / "navigation" / "exploration"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "navigation"))
sys.path.insert(0, str(_ROOT / "travis_brain"))
sys.path.insert(0, str(_EXPLORATION_PKG))

from navigation.exploration.exploration.explore_costmap_map import MapData, Waypoint


class SemanticExplorationVisualiser:
    """Real-time exploration visualiser with object markers and optional 3D semantic panel.

    A class is used because the matplotlib figure and axes objects must persist
    across many .update() calls — re-creating them every frame would be slow and
    cause flickering. The class stores them once and redraws on demand.

    Args:
        map_data: MapData loaded from load_map().
        objects:  List of SceneObject instances to render as markers.
        show_3d:  If True, adds a 3D semantic map panel alongside the 2D map.
    """

    # Colour palette: label → matplotlib colour (tab10)
    _CMAP = plt.get_cmap("tab10")

    def __init__(
        self,
        map_data: MapData,
        objects: list,              # list[SceneObject] — imported lazily to avoid circular deps
        show_3d: bool = False,
        all_objects_visible: bool = False,
    ) -> None:
        """
        all_objects_visible: when True every object is shown as a coloured ★ from the
            start, regardless of its .detected flag. Use for Demo 1 (reference markers).
            When False (Demo 3) undetected objects are grey × and turn coloured ★ only
            when the camera has seen them.
        """
        self._map_data = map_data
        self._objects = objects
        self._show_3d = show_3d
        self._all_visible = all_objects_visible

        # Assign a consistent colour index to each unique label
        all_labels = sorted({obj.label for obj in objects}) if objects else []
        self._label_colour: dict[str, tuple] = {
            lbl: self._CMAP(i % 10) for i, lbl in enumerate(all_labels)
        }

        if show_3d:
            self._fig = plt.figure(figsize=(20, 10))
            self._ax   = self._fig.add_subplot(121)
            self._ax3d = self._fig.add_subplot(122, projection="3d")
        else:
            self._fig, self._ax = plt.subplots(figsize=(12, 8))
            self._ax3d = None

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
        semantic_map=None,
    ) -> None:
        """Redraw all layers with the current exploration state.

        Call plt.pause(dt) after this to drive the matplotlib event loop.

        Args:
            waypoints:         Current planned waypoints (blue numbered dots).
            robot_col/row:     Robot pixel position.
            robot_heading_deg: Current camera heading (0=East, 90=North, CCW+).
            trajectory:        Ordered list of (col, row) positions visited.
            coverage_pct:      Visual coverage ratio [0,1] shown in title.
            semantic_map:      SemanticMap instance — triggers 3D panel refresh
                               (only used when show_3d=True).
        """
        ax = self._ax
        ax.cla()

        md = self._map_data
        H, W = md.pgm_array.shape

        # ── Layer 1: occupancy base ────────────────────────────────────────
        display = np.full((H, W, 3), 0.65)          # grey = unknown
        display[md.free_mask]     = 1.0              # white = free
        display[md.occupied_mask] = 0.0              # black = wall
        ax.imshow(display, origin="upper", zorder=0)

        # ── Layer 2: navigable mask (light-blue overlay) ──────────────────
        nav_overlay = np.zeros((H, W, 4), dtype=np.float32)
        nav_overlay[md.navigable_mask] = [0.3, 0.6, 1.0, 0.18]
        ax.imshow(nav_overlay, origin="upper", zorder=1)

        # ── Layer 3: covered mask (green overlay) ─────────────────────────
        cov_overlay = np.zeros((H, W, 4), dtype=np.float32)
        cov_overlay[md.covered_mask] = [0.0, 0.75, 0.0, 0.45]
        ax.imshow(cov_overlay, origin="upper", zorder=2)

        # ── Layer 4: planned waypoints (blue dots + dashed path) ──────────
        if waypoints:
            wp_cols = [wp.col for wp in waypoints]
            wp_rows = [wp.row for wp in waypoints]
            ax.plot(wp_cols, wp_rows, "--", color="steelblue", linewidth=0.8,
                    alpha=0.6, zorder=3)
            for i, wp in enumerate(waypoints):
                ax.plot(wp.col, wp.row, "o", color="steelblue", markersize=6, zorder=4)
                ax.text(wp.col + 4, wp.row - 4, str(i + 1),
                        color="steelblue", fontsize=6, fontweight="bold", zorder=5)

        # ── Layer 5: trajectory (orange line) ─────────────────────────────
        if len(trajectory) > 1:
            t_cols = [p[0] for p in trajectory]
            t_rows = [p[1] for p in trajectory]
            ax.plot(t_cols, t_rows, "-", color="darkorange", linewidth=1.2,
                    alpha=0.8, zorder=3)

        # ── Layer 6: robot pose (red square + heading arrow) ──────────────
        arrow_len = max(W, H) * 0.04
        heading_rad = np.radians(robot_heading_deg)
        dx =  arrow_len * np.cos(heading_rad)
        dy = -arrow_len * np.sin(heading_rad)   # -sin: row↓ = y↑ in world
        ax.annotate(
            "",
            xy=(robot_col + dx, robot_row + dy),
            xytext=(robot_col, robot_row),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
            zorder=6,
        )
        ax.plot(robot_col, robot_row, "s", color="red", markersize=9, zorder=6)

        # ── Layer 7: object markers ────────────────────────────────────────
        legend_entries: dict[str, object] = {}
        for obj in self._objects:
            colour = self._label_colour.get(obj.label, "grey")
            if self._all_visible or obj.detected:
                marker, ms, alpha = "*", 14, 1.0
            else:
                marker, ms, alpha = "x", 9, 0.55
                colour = "grey"
            ax.plot(obj.col, obj.row, marker, color=colour,
                    markersize=ms, alpha=alpha, markeredgewidth=1.5, zorder=7)
            ax.text(obj.col + 3, obj.row - 5, obj.label,
                    fontsize=5, color=colour, alpha=alpha, zorder=7)
            if (self._all_visible or obj.detected) and obj.label not in legend_entries:
                legend_entries[obj.label] = plt.Line2D(
                    [0], [0], marker="*", color="w",
                    markerfacecolor=self._label_colour[obj.label],
                    markersize=10, label=obj.label,
                )

        if legend_entries:
            ax.legend(handles=list(legend_entries.values()),
                      loc="lower right", fontsize=7, framealpha=0.7)

        # ── Title ──────────────────────────────────────────────────────────
        title = "Visual Exploration"
        if coverage_pct is not None:
            title += f"  —  Coverage: {coverage_pct:.1%}"
        n_detected = sum(1 for o in self._objects if o.detected)
        if self._objects:
            title += f"  —  Objects detected: {n_detected}/{len(self._objects)}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("col (px)")
        ax.set_ylabel("row (px)")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # ── 3D semantic panel ──────────────────────────────────────────────
        if self._show_3d and self._ax3d is not None and semantic_map is not None:
            self._ax3d.cla()
            if len(semantic_map) > 0:
                from navigation.semantic_map.visualisation.view_semantic_map import visualise_semantic_map
                visualise_semantic_map(semantic_map, fig=self._fig, ax=self._ax3d,
                                       title="Semantic Map")

        self._fig.canvas.draw_idle()

    def show_nonblocking(self) -> None:
        """Open the figure window without blocking. Drive with plt.pause(dt)."""
        plt.ion()
        self._fig.show()

    def show(self) -> None:
        """Display the figure and block until the window is closed."""
        plt.ioff()
        plt.show()

    def save(self, path: str | Path) -> None:
        """Save current frame as PNG."""
        self._fig.savefig(str(path), dpi=150, bbox_inches="tight")
