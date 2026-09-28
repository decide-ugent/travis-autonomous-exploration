"""
Tests for SemanticMap and SemanticNode.

Mock object list
----------------
label       confidence  x     y     z     timestamp  expected outcome
----------- ----------- ----- ----- ----- ---------- ----------------
mug         95.0        1.0   2.0   0.5   1000.0     accepted  (id 0)
chair       85.0        3.0   1.0   0.0   1001.0     accepted  (id 1)
box         70.0        0.5   0.5   0.2   1002.0     REJECTED  (confidence < 80)
laptop      90.0        2.0   3.0   1.0   1003.0     accepted  (id 2)
bottle      82.0        5.0   0.5   0.3   1004.0     accepted  (id 3)
mug         88.0        1.1   2.1   0.6   1005.0     DUPLICATE of id 0 → update
chair       91.0        6.0   6.0   0.0   1006.0     accepted  (id 4)  — far from first chair
"""

import time
import pytest
from pathlib import Path
from travis_brain.semantic_map import SemanticMap, SemanticNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(label: str, confidence: float, x: float, y: float, z: float,
          timestamp: float) -> SemanticNode:
    return SemanticNode(label=label, confidence=confidence,
                        x=x, y=y, z=z, timestamp=timestamp)


def _build_map() -> SemanticMap:
    """Build the standard mock map used across most tests."""
    m = SemanticMap()
    detections = [
        _node("mug",    95.0, 1.0, 2.0, 0.5, 1000.0),
        _node("chair",  85.0, 3.0, 1.0, 0.0, 1001.0),
        _node("box",    70.0, 0.5, 0.5, 0.2, 1002.0),   # rejected
        _node("laptop", 90.0, 2.0, 3.0, 1.0, 1003.0),
        _node("bottle", 82.0, 5.0, 0.5, 0.3, 1004.0),
        _node("mug",    88.0, 1.1, 2.1, 0.6, 1005.0),   # duplicate of first mug
        _node("chair",  91.0, 6.0, 6.0, 0.0, 1006.0),   # new chair, far from first
    ]
    for det in detections:
        m.add_node(det)
    return m


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:

    def test_accepted_at_threshold(self):
        m = SemanticMap()
        node_id = m.add_node(_node("mug", 80.0, 0.0, 0.0, 0.0, 0.0))
        assert node_id is not None
        assert len(m) == 1

    def test_accepted_above_threshold(self):
        m = SemanticMap()
        node_id = m.add_node(_node("mug", 95.0, 0.0, 0.0, 0.0, 0.0))
        assert node_id is not None

    def test_rejected_below_threshold(self):
        m = SemanticMap()
        node_id = m.add_node(_node("box", 70.0, 0.5, 0.5, 0.2, 0.0))
        assert node_id is None
        assert len(m) == 0

    def test_rejected_just_below_threshold(self):
        m = SemanticMap()
        node_id = m.add_node(_node("mug", 79.9, 0.0, 0.0, 0.0, 0.0))
        assert node_id is None

    def test_mock_map_has_five_nodes(self):
        # 7 detections - 1 rejected (box) - 1 duplicate (second mug) = 5
        m = _build_map()
        assert len(m) == 5


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

class TestDuplicateHandling:

    def test_duplicate_does_not_add_new_node(self):
        m = SemanticMap()
        m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        m.add_node(_node("mug", 88.0, 1.1, 2.1, 0.6, 1005.0))
        assert len(m) == 1

    def test_duplicate_returns_existing_id(self):
        m = SemanticMap()
        id_first = m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        id_second = m.add_node(_node("mug", 88.0, 1.1, 2.1, 0.6, 1005.0))
        assert id_first == id_second

    def test_duplicate_averages_position(self):
        m = SemanticMap()
        id_first = m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        m.add_node(_node("mug", 88.0, 1.1, 2.1, 0.6, 1005.0))
        node = m.get_node_by_id(id_first)
        assert node.x == pytest.approx((1.0 + 1.1) / 2)
        assert node.y == pytest.approx((2.0 + 2.1) / 2)
        assert node.z == pytest.approx((0.5 + 0.6) / 2)

    def test_duplicate_updates_confidence(self):
        m = SemanticMap()
        id_first = m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        m.add_node(_node("mug", 88.0, 1.1, 2.1, 0.6, 1005.0))
        assert m.get_node_by_id(id_first).confidence == pytest.approx(88.0)

    def test_duplicate_updates_timestamp(self):
        m = SemanticMap()
        id_first = m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        m.add_node(_node("mug", 88.0, 1.1, 2.1, 0.6, 1005.0))
        assert m.get_node_by_id(id_first).timestamp == pytest.approx(1005.0)

    def test_same_label_far_away_is_not_duplicate(self):
        m = SemanticMap()
        m.add_node(_node("chair", 85.0, 3.0, 1.0, 0.0, 1001.0))
        m.add_node(_node("chair", 91.0, 6.0, 6.0, 0.0, 1006.0))
        assert len(m) == 2

    def test_same_position_different_label_is_not_duplicate(self):
        m = SemanticMap()
        m.add_node(_node("mug",    95.0, 1.0, 2.0, 0.5, 1000.0))
        m.add_node(_node("bottle", 82.0, 1.0, 2.0, 0.5, 1004.0))
        assert len(m) == 2


# ---------------------------------------------------------------------------
# Node id invariant
# ---------------------------------------------------------------------------

class TestNodeIdInvariant:

    def test_ids_are_sequential_from_zero(self):
        m = SemanticMap()
        id0 = m.add_node(_node("mug",    95.0, 1.0, 2.0, 0.5, 1000.0))
        id1 = m.add_node(_node("chair",  85.0, 3.0, 1.0, 0.0, 1001.0))
        id2 = m.add_node(_node("laptop", 90.0, 2.0, 3.0, 1.0, 1003.0))
        assert [id0, id1, id2] == [0, 1, 2]

    def test_rejected_node_does_not_advance_counter(self):
        m = SemanticMap()
        m.add_node(_node("box", 70.0, 0.0, 0.0, 0.0, 0.0))   # rejected
        id0 = m.add_node(_node("mug", 95.0, 1.0, 2.0, 0.5, 1000.0))
        assert id0 == 0

    def test_graph_key_equals_node_id(self):
        m = _build_map()
        graph = m.get_map()
        for node_id, attrs in graph.nodes(data=True):
            assert attrs["data"].id == node_id


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_round_trip_node_count(self, tmp_path):
        m = _build_map()
        path = tmp_path / "map.pkl"
        m.save_semantic_map(path)
        loaded = SemanticMap.load_semantic_map(path)
        assert len(loaded) == len(m)

    def test_round_trip_all_attributes(self, tmp_path):
        m = _build_map()
        path = tmp_path / "map.pkl"
        m.save_semantic_map(path)
        loaded = SemanticMap.load_semantic_map(path)

        original_nodes = {n.id: n for n in
                         [m.get_node_by_id(i) for i in range(5)]}
        for node_id, orig in original_nodes.items():
            restored = loaded.get_node_by_id(node_id)
            assert restored is not None
            assert restored.label       == orig.label
            assert restored.confidence  == pytest.approx(orig.confidence)
            assert restored.x           == pytest.approx(orig.x)
            assert restored.y           == pytest.approx(orig.y)
            assert restored.z           == pytest.approx(orig.z)
            assert restored.timestamp   == pytest.approx(orig.timestamp)
            assert restored.id          == orig.id

    def test_round_trip_parameters(self, tmp_path):
        m = _build_map()
        path = tmp_path / "map.pkl"
        m.save_semantic_map(path)
        loaded = SemanticMap.load_semantic_map(path)
        assert loaded.confidence_threshold          == m.confidence_threshold
        assert loaded.duplicate_distance_threshold  == m.duplicate_distance_threshold
        assert loaded.position_query_default_tolerance == m.position_query_default_tolerance

    def test_round_trip_next_id_continues(self, tmp_path):
        m = _build_map()
        path = tmp_path / "map.pkl"
        m.save_semantic_map(path)
        loaded = SemanticMap.load_semantic_map(path)
        new_id = loaded.add_node(_node("plant", 85.0, 0.0, 0.0, 0.0, 9999.0))
        assert new_id == 5   # continues from where the original left off


# ---------------------------------------------------------------------------
# get_node_by_id
# ---------------------------------------------------------------------------

class TestGetNodeById:

    def test_returns_correct_node(self):
        m = _build_map()
        node = m.get_node_by_id(0)
        assert node is not None
        assert node.label == "mug"

    def test_returns_none_for_missing_id(self):
        m = _build_map()
        assert m.get_node_by_id(999) is None


# ---------------------------------------------------------------------------
# get_nodes_by_label
# ---------------------------------------------------------------------------

class TestGetNodesByLabel:

    def test_single_result(self):
        m = _build_map()
        nodes = m.get_nodes_by_label("laptop")
        assert len(nodes) == 1
        assert nodes[0].label == "laptop"

    def test_multiple_results(self):
        m = _build_map()
        nodes = m.get_nodes_by_label("chair")
        assert len(nodes) == 2
        assert all(n.label == "chair" for n in nodes)

    def test_unknown_label_returns_empty(self):
        m = _build_map()
        assert m.get_nodes_by_label("unicorn") == []

    def test_rejected_label_not_in_map(self):
        m = _build_map()
        assert m.get_nodes_by_label("box") == []


# ---------------------------------------------------------------------------
# get_nodes_by_position
# ---------------------------------------------------------------------------

class TestGetNodesByPosition:

    def test_with_max_distance_returns_all_within_range(self):
        m = _build_map()
        # mug is at approx (1.05, 2.05, 0.55) after averaging with duplicate
        results = m.get_nodes_by_position(1.05, 2.05, max_distance=0.2)
        labels = [n.label for n in results]
        assert "mug" in labels

    def test_with_max_distance_excludes_far_nodes(self):
        m = _build_map()
        results = m.get_nodes_by_position(1.0, 2.0, max_distance=0.1)
        labels = [n.label for n in results]
        assert "bottle" not in labels   # bottle is at (5.0, 0.5)

    def test_without_max_distance_returns_closest(self):
        m = _build_map()
        results = m.get_nodes_by_position(3.0, 1.0)    # exact position of chair id 1
        assert len(results) >= 1
        assert any(n.label == "chair" for n in results)

    def test_without_max_distance_always_returns_closest(self):
        m = _build_map()
        # Without max_distance, always returns the closest node — no matter how far.
        # Use max_distance if you need a "nothing nearby" guard.
        results = m.get_nodes_by_position(50.0, 50.0)
        assert len(results) >= 1

    def test_max_distance_returns_empty_when_far(self):
        m = _build_map()
        results = m.get_nodes_by_position(50.0, 50.0, max_distance=1.0)
        assert results == []

    def test_2d_query_returns_nodes_at_same_xy_different_z(self):
        m = SemanticMap()
        m.add_node(_node("cup",   90.0, 1.0, 1.0, 0.0, 0.0))
        m.add_node(_node("plate", 90.0, 1.0, 1.0, 1.5, 0.0))
        # 2D query (no z) — both share same x,y → both are closest, both returned
        results = m.get_nodes_by_position(1.0, 1.0)
        assert len(results) == 2

    def test_3d_query_with_z_distinguishes_height(self):
        m = SemanticMap()
        m.add_node(_node("cup",   90.0, 1.0, 1.0, 0.0, 0.0))
        m.add_node(_node("plate", 90.0, 1.0, 1.0, 1.5, 0.0))
        # 3D query close to cup — plate is 1.5m above, outside 0.15m tolerance
        results = m.get_nodes_by_position(1.0, 1.0, z=0.0)
        assert len(results) == 1
        assert results[0].label == "cup"

    def test_max_distance_3d(self):
        m = _build_map()
        results = m.get_nodes_by_position(1.05, 2.05, z=0.55, max_distance=0.1)
        assert any(n.label == "mug" for n in results)

    def test_empty_map_returns_empty(self):
        m = SemanticMap()
        assert m.get_nodes_by_position(0.0, 0.0) == []


# ---------------------------------------------------------------------------
# node_exists
# ---------------------------------------------------------------------------

class TestNodeExists:

    def test_exists_by_label(self):
        m = _build_map()
        assert m.node_exists(label="mug") is True

    def test_not_exists_by_label(self):
        m = _build_map()
        assert m.node_exists(label="unicorn") is False

    def test_rejected_node_does_not_exist(self):
        m = _build_map()
        assert m.node_exists(label="box") is False

    def test_exists_by_id(self):
        m = _build_map()
        assert m.node_exists(id=0) is True

    def test_not_exists_by_id(self):
        m = _build_map()
        assert m.node_exists(id=999) is False

    def test_exists_multi_field(self):
        m = _build_map()
        assert m.node_exists(label="laptop", id=2) is True

    def test_not_exists_multi_field_mismatch(self):
        m = _build_map()
        # label and id both exist, but not on the same node
        assert m.node_exists(label="mug", id=2) is False


# ---------------------------------------------------------------------------
# get_map
# ---------------------------------------------------------------------------

class TestGetMap:

    def test_returns_networkx_graph(self):
        import networkx as nx
        m = _build_map()
        assert isinstance(m.get_map(), nx.Graph)

    def test_graph_node_count_matches_len(self):
        m = _build_map()
        assert m.get_map().number_of_nodes() == len(m)
