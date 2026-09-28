"""
Semantic map: a NetworkX graph of detected objects in the environment.

Pure Python, no ROS2 imports. Parameters are read from the per-package
config/semantic_map_system_parameters.yaml when used as a standalone module.
When called from a ROS2 node, parameters are passed directly to __init__.
"""


import logging
import math #faster than numpy here
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "semantic_map_system_parameters.yaml"

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass #dataclass equivalent to write _init__ etc, but shorter
class SemanticNode:
    label: str
    confidence: float # 0–100
    x: float          # global position (meters)
    y: float
    z: float
    timestamp: float # seconds
    id: int = field(default=-1)# assigned by SemanticMap on insertion
    observed_from: list = field(default_factory=list)   # Iteration 2 (for later use)

class SemanticMap:
    """
    Manages a set of detected objects as a NetworkX graph (nodes only, no edges
    in Iteration 1).

    Parameters are loaded from system_parameters.yaml (semantic_map section).
    Pass a custom config_path to override the default location.
    """

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG, logger=None) -> None:
        with open(Path(config_path), "r") as f:
            cfg = yaml.safe_load(f)["semantic_map"]

        self.confidence_threshold: float = cfg["confidence_threshold"]
        self.duplicate_distance_threshold: float = cfg["duplicate_distance_threshold"]
        self.position_query_default_tolerance: float = cfg["position_query_default_tolerance"]

        self._graph: nx.Graph = nx.Graph()
        self._next_id: int = 0
        # Accept any logger with .debug()/.info()/.warning()/.error() —
        # works with Python logging and rclpy node loggers alike.
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Add node
    # ------------------------------------------------------------------

    def add_node(self, node: SemanticNode) -> int | None:
        """
        Add a detection to the map.

        The node's id field is set here — do not assign it before calling this
        method. Returns the node id (existing or new) if accepted, None if
        rejected due to low confidence.
        """
        if node.confidence < self.confidence_threshold:
            self._logger.debug(
                f"Node rejected — label='{node.label}', confidence={node.confidence:.1f} "
                f"below threshold {self.confidence_threshold:.1f}"
            )
            return None

        duplicate_id = self._find_duplicate(node)
        if duplicate_id is not None:
            self._update_node(duplicate_id, node)
            return duplicate_id

        # Assign id to the SemanticNode and use that same value as the graph key.
        node.id = self._next_id
        self._next_id += 1
        self._graph.add_node(node.id, data=node)

        # Invariant check: graph key must equal node.id
        assert self._graph.nodes[node.id]["data"].id == node.id
        self._logger.debug(
            f"Node added    — id={node.id}, label='{node.label}', "
            f"confidence={node.confidence:.1f}, pos=({node.x:.2f}, {node.y:.2f}, {node.z:.2f})"
        )
        return node.id

    def _find_duplicate(self, node: SemanticNode) -> int | None:
        """Return the id of an existing node with the same label within
        duplicate_distance_threshold, or None."""
        for node_id, attrs in self._graph.nodes(data=True):
            existing: SemanticNode = attrs["data"]
            if existing.label != node.label:
                continue
            if _distance_3d(existing, node) < self.duplicate_distance_threshold:
                return node_id
        return None

    def _update_node(self, node_id: int, new: SemanticNode) -> None:
        """Average position with existing node, refresh confidence and timestamp."""
        existing: SemanticNode = self._graph.nodes[node_id]["data"]
        existing.x = (existing.x + new.x) / 2
        existing.y = (existing.y + new.y) / 2
        existing.z = (existing.z + new.z) / 2
        existing.confidence = new.confidence
        existing.timestamp = new.timestamp
        self._logger.debug(
            f"Duplicate     — label='{new.label}' merged into id={node_id}, "
            f"new pos=({existing.x:.2f}, {existing.y:.2f}, {existing.z:.2f}), "
            f"confidence={existing.confidence:.1f}"
        )

    # ------------------------------------------------------------------
    # Query a node given variable
    # ------------------------------------------------------------------

    def get_node_by_id(self, node_id: int) -> SemanticNode | None:
        """Return the SemanticNode for the given id, or None if not found."""
        if node_id not in self._graph:
            return None
        node: SemanticNode = self._graph.nodes[node_id]["data"]
        assert node.id == node_id, f"Invariant violated: graph key {node_id} != node.id {node.id}"
        return node

    def get_nodes_by_label(self, label: str) -> list[SemanticNode]:
        """Return all nodes whose label matches exactly."""
        return [
            attrs["data"]
            for _, attrs in self._graph.nodes(data=True)
            if attrs["data"].label == label
        ]

    def get_nodes_by_position(
        self,
        x: float,
        y: float,
        z: float | None = None,
        max_distance: float | None = None,
    ) -> list[SemanticNode]:
        """
        Query nodes by position.

        Parameters
        ----------
        x, y : float
            Query coordinates (meters).
        z : float | None
            If provided, distance is computed in 3D; otherwise in 2D.
        max_distance : float | None
            If given, return all nodes whose distance to (x, y[, z]) is <=
            max_distance.
            If None, find the closest node, then return all nodes within
            position_query_default_tolerance of that closest node's position.

        Returns
        -------
        list[SemanticNode]
            Matching nodes, or an empty list if none qualify.
        """
        all_nodes: list[SemanticNode] = [
            attrs["data"] for _, attrs in self._graph.nodes(data=True)
        ]

        if not all_nodes:
            return []

        def dist_to_query(n: SemanticNode) -> float:
            if z is not None:
                return _distance_3d_coords(n.x, n.y, n.z, x, y, z)
            return _distance_2d_coords(n.x, n.y, x, y)

        if max_distance is not None:
            return [n for n in all_nodes if dist_to_query(n) <= max_distance]

        # No max_distance: find the closest node, then return all nodes within
        # position_query_default_tolerance of that closest node's position.
        closest = min(all_nodes, key=dist_to_query)

        def dist_to_closest(n: SemanticNode) -> float:
            if z is not None:
                return _distance_3d(n, closest)
            return _distance_2d(n, closest)

        return [
            n for n in all_nodes
            if dist_to_closest(n) <= self.position_query_default_tolerance
        ]

    def get_map(self) -> nx.Graph:
        """Return the underlying NetworkX graph."""
        return self._graph

    def node_exists(self, **kwargs) -> bool:
        """
        Return True if any node matches all supplied field=value pairs.

        Examples
        --------
        semantic_map.node_exists(label="mug")
        semantic_map.node_exists(label="mug", confidence=95.0)
        """
        for _, attrs in self._graph.nodes(data=True):
            node: SemanticNode = attrs["data"]
            if all(getattr(node, k, None) == v for k, v in kwargs.items()):
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_semantic_map(self, path: str | Path) -> None:
        """Serialize the map to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "graph": self._graph,
                    "next_id": self._next_id,
                    "confidence_threshold": self.confidence_threshold,
                    "duplicate_distance_threshold": self.duplicate_distance_threshold,
                    "position_query_default_tolerance": self.position_query_default_tolerance,
                },
                f,
            )
    
    @classmethod  #this is so we can call this method outside this class
    def load_semantic_map(cls, path: str | Path) -> "SemanticMap":
        """Deserialize a map from a pickle file, restoring all parameters."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)

        # Bypass __init__ so we don't re-read the yaml — all params are in the pickle.
        instance = object.__new__(cls)
        instance.confidence_threshold = data["confidence_threshold"]
        instance.duplicate_distance_threshold = data["duplicate_distance_threshold"]
        instance.position_query_default_tolerance = data["position_query_default_tolerance"]
        instance._graph = data["graph"]
        instance._next_id = data["next_id"]
        instance._logger = logging.getLogger(__name__)
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers (default)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __repr__(self) -> str:
        return f"SemanticMap({self._graph.number_of_nodes()} nodes)"


# ------------------------------------------------------------------
# Distance helpers
# ------------------------------------------------------------------

def _distance_3d(a: SemanticNode, b: SemanticNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _distance_2d(a: SemanticNode, b: SemanticNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _distance_3d_coords(x1: float, y1: float, z1: float,
                         x2: float, y2: float, z2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def _distance_2d_coords(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
