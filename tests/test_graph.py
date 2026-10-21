import pytest

from tracknet.graph import CameraGraph


def test_levels_forward_and_backward():
    g = CameraGraph()
    fwd = g.levels("c017", "forward", 2)
    assert fwd["c017"] == 0 and fwd["c018"] == 1 and fwd["c021"] == 1 and fwd["c025"] == 2
    assert "c026" not in fwd or fwd["c026"] == 2
    back = g.levels("c021", "backward", 1)
    assert set(back) == {"c021", "c017", "c018"}


def test_bidirectional_levels_has_both_sides():
    out = CameraGraph().bidirectional_levels("c021", 1)
    assert set(out["forward"]) == {"c021", "c025", "c026"}
    assert set(out["backward"]) == {"c021", "c017", "c018"}


def test_shortest_path_prefers_fewest_hops():
    g = CameraGraph()
    assert g.shortest_path("c017", "c021") == ["c017", "c021"]          # direct skip edge
    assert g.shortest_path("c017", "c026") == ["c017", "c021", "c026"]
    assert g.shortest_path("c017", "c017") == ["c017"]


def test_no_path_against_direction_of_travel():
    g = CameraGraph()
    assert g.shortest_path("c026", "c017") is None
    assert g.transit_window("c026", "c017") is None
    assert g.hops("c018", "c017") is None


def test_unknown_cameras_return_none():
    assert CameraGraph().shortest_path("c017", "zzz") is None


def test_transit_window_sums_edges():
    g = CameraGraph()
    assert g.transit_window("c017", "c021") == (8.0, 30.0)
    assert g.transit_window("c017", "c026") == (16.0, 60.0)


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        CameraGraph(edges={("a", "b"): (10.0, 2.0)}, cameras={"a": {}, "b": {}})


def test_bidirectional_search_on_a_longer_chain():
    edges = {(f"n{i}", f"n{i+1}"): (1.0, 5.0) for i in range(9)}
    g = CameraGraph(edges=edges, cameras={f"n{i}": {} for i in range(10)})
    assert g.shortest_path("n0", "n9") == [f"n{i}" for i in range(10)]
    assert g.hops("n2", "n7") == 5
