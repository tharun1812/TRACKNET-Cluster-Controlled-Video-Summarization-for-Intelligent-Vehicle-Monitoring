from tracknet.graph import CameraGraph
from tracknet.matcher import color_score, find_candidates, score_pair, temporal_score
from tests.conftest import make_track

G = CameraGraph()


def test_temporal_score_shape():
    w = (3.0, 20.0)
    assert temporal_score(11.5, w) == 1.0
    assert 0.7 <= temporal_score(3.0, w) < 1.0
    assert 0 < temporal_score(1.0, w) < 0.7        # slightly early: allowed with a penalty
    assert temporal_score(-20, w) is None           # far too early
    assert temporal_score(60, w) is None            # far too late


def test_color_score():
    assert color_score("red", "red") == 1.0
    assert color_score("white", "silver") == 0.5
    assert color_score("red", "blue") == 0.0
    assert color_score("unknown", "blue") == 0.5


def test_true_match_scores_high_and_wrong_color_lower():
    src = make_track("c017-T001", "c017", 10, 16)
    good = make_track("c018-T001", "c018", 24, 30)
    bad = make_track("c018-T002", "c018", 24, 30, color="blue")
    s_good = score_pair(src, good, G, "forward")
    s_bad = score_pair(src, bad, G, "forward")
    assert s_good.total > 0.9
    assert s_good.total > s_bad.total + 0.3


def test_class_mismatch_is_penalised():
    src = make_track("c017-T001", "c017", 10, 16)
    truck = make_track("c018-T001", "c018", 24, 30, cls="truck")
    car = make_track("c018-T002", "c018", 24, 30)
    assert score_pair(src, truck, G, "forward").total < 0.4 * score_pair(src, car, G, "forward").total + 1e-9


def test_impossible_pairs_return_none():
    src = make_track("c017-T001", "c017", 10, 16)
    assert score_pair(src, make_track("x", "c017", 30, 36), G, "forward") is None      # same camera
    assert score_pair(src, make_track("x", "c018", 5, 9), G, "forward") is None        # arrives before leaving
    assert score_pair(src, make_track("x", "c018", 200, 206), G, "forward") is None    # far too late
    back_cam = make_track("x", "c026", 30, 36)
    assert score_pair(src, back_cam, G, "backward") is None                             # no road back from c026 to c017's past


def test_backward_direction_finds_earlier_track():
    later = make_track("c018-T001", "c018", 24, 30)
    earlier = make_track("c017-T001", "c017", 10, 16)
    s = score_pair(later, earlier, G, "backward")
    assert s is not None and s.total > 0.9


def test_find_candidates_ranks_and_respects_exclusions(store):
    src = make_track("c017-T001", "c017", 10, 16)
    store.add_tracks([
        src,
        make_track("c018-T001", "c018", 24, 30),                    # true match
        make_track("c018-T002", "c018", 22, 28, color="blue"),      # distractor
        make_track("c021-T001", "c021", 30, 36),                    # same colour, two hops of slack
    ])
    cands = find_candidates(src, store, G, "forward")
    assert cands[0].candidate_id == "c018-T001"
    assert {c.candidate_id for c in cands} >= {"c018-T001", "c018-T002", "c021-T001"}
    assert "c017-T001" not in {c.candidate_id for c in cands}
    ex = find_candidates(src, store, G, "forward", exclude_ids={"c018-T001"})
    assert "c018-T001" not in {c.candidate_id for c in ex}
    ex_cam = find_candidates(src, store, G, "forward", exclude_cameras={"c018"})
    assert all(c.camera_id != "c018" for c in ex_cam)
