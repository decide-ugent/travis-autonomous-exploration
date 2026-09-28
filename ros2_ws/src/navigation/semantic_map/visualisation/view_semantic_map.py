"""
3D interactive visualization of a SemanticMap.

Usage
-----
# visualise a saved map:
python travis_brain/visualisation/view_semantic_map.py path/to/semantic_map.pkl

# visualise a built-in demo map (no argument needed):
python travis_brain/visualisation/view_semantic_map.py

Controls
--------
Left-click + drag   Rotate
Right-click + drag  Zoom
Middle-click + drag Pan
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3D projection

# Allow running from repo root or from tests/
_ROOT = Path(__file__).parent.parent.parent.parent  # /ros2_ws/src/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "navigation"))

from navigation.semantic_map.semantic_map import SemanticMap, SemanticNode


# ---------------------------------------------------------------------------
# Demo map used when no pickle is supplied
# ---------------------------------------------------------------------------

def _build_demo_map() -> SemanticMap:
    m = SemanticMap()
    detections = [
        SemanticNode(label="mug",    confidence=95.0, x=1.0, y=2.0, z=0.8,  timestamp=1000.0),
        SemanticNode(label="chair",  confidence=85.0, x=3.0, y=1.0, z=0.0,  timestamp=1001.0),
        SemanticNode(label="laptop", confidence=90.0, x=2.0, y=3.0, z=1.0,  timestamp=1003.0),
        SemanticNode(label="bottle", confidence=82.0, x=5.0, y=0.5, z=0.4,  timestamp=1004.0),
        SemanticNode(label="chair",  confidence=91.0, x=6.0, y=6.0, z=0.0,  timestamp=1006.0),
        SemanticNode(label="plant",  confidence=88.0, x=0.5, y=4.0, z=1.2,  timestamp=1007.0),
        SemanticNode(label="mug",    confidence=87.0, x=1.1, y=2.1, z=0.85, timestamp=1008.0),  # duplicate
    ]
    for det in detections:
        m.add_node(det)
    return m


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualise_semantic_map(semantic_map: SemanticMap, fig:plt.figure=None, ax=None, title: str = "Semantic Map", ) -> None:
    nodes = [
        attrs["data"]
        for _, attrs in semantic_map.get_map().nodes(data=True)
    ]

    if not nodes:
        print("Map is empty — nothing to display.")
        return

    # Assign a consistent color to each unique label
    unique_labels = sorted({n.label for n in nodes})
    cmap = plt.get_cmap("tab10")
    label_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}
    
    no_fig_input = False
    if fig is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        no_fig_input  =True

    # Plot each node as a scatter point + text label
    for node in nodes:
        color = label_color[node.label]
        ax.scatter(node.x, node.y, node.z, color=color, s=80, zorder=5)
        ax.text(
            node.x, node.y, node.z,
            f"  {node.label}\n  id:{node.id}",
            fontsize=8,
            color=color,
        )

    # Legend: one entry per label
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=label_color[lbl], markersize=9, label=lbl)
        for lbl in unique_labels
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{title}  ({len(semantic_map)} nodes)")
    if no_fig_input:
        plt.tight_layout()
        plt.show()
    return fig, ax



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl_path = Path(sys.argv[1])
        if not pkl_path.exists():
            print(f"File not found: {pkl_path}")
            sys.exit(1)
        semantic_map = SemanticMap.load_semantic_map(pkl_path)
        title = f"Semantic Map — {pkl_path.name}"
    else:
        print("No pickle file given, showing demo map.")
        semantic_map = _build_demo_map()
        title = "Semantic Map, demo"

    visualise_semantic_map(semantic_map, title=title)
