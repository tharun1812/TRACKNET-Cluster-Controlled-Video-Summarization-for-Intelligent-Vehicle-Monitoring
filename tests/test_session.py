import pytest

from tracknet.graph import CameraGraph
from tracknet.session import TrackerSession
from tests.conftest import make_track


@pytest.fixture()
def populated(store):
    store.add_tracks([
        make_track("c017-T001", "c017", 10, 16),
        make_track("c018-T001", "c018", 24, 30),
        make_track("c018-T002", "c018", 25, 31, color="blue"),
        make_track("c021-T001", "c021", 40, 46),
        make_track("c025-T001", "c025", 55, 61),
    ])
    return store


def test_unknown_track_raises(store):
    with pytest.raises(KeyError):
        TrackerSession(store, CameraGraph(), "nope")


def test_confirm_extends_chain_and_moves_frontier(populated):
    s = TrackerSession(populated, CameraGraph(), "c017-T001")
    top = s.candidates("forward")[0]
    assert top.candidate_id == "c018-T001"
    s.confirm("forward", "c018-T001")
    assert s.chain == ["c017-T001", "c018-T001"]
    assert s.frontier["forward"] == "c018-T001"
    assert s.candidates("forward")[0].candidate_id == "c021-T001"


def test_skip_removes_candidate_and_invalid_confirm_fails(populated):
    s = TrackerSession(populated, CameraGraph(), "c017-T001")
    s.reject("forward", "c018-T001")
    assert "c018-T001" not in [c.candidate_id for c in s.candidates("forward")]
    with pytest.raises(ValueError):
        s.confirm("forward", "c018-T001")
    with pytest.raises(ValueError):
        s.confirm("sideways", "c018-T002")


def test_backward_confirm_prepends(populated):
    s = TrackerSession(populated, CameraGraph(), "c021-T001")
    top = s.candidates("backward")[0]
    s.confirm("backward", top.candidate_id)
    assert s.chain[-1] == "c021-T001" and s.chain[0] == top.candidate_id


def test_auto_trace_builds_full_chain_in_time_order(populated):
    s = TrackerSession(populated, CameraGraph(), "c017-T001")
    s.auto_trace(min_score=0.6)
    assert s.chain == ["c017-T001", "c018-T001", "c021-T001", "c025-T001"]
    d = s.to_dict()
    assert d["visited_cameras"] == ["c017", "c018", "c021", "c025"]
    assert len(d["log"]) == 3


def test_auto_trace_from_the_middle_goes_both_ways(populated):
    s = TrackerSession(populated, CameraGraph(), "c021-T001")
    s.auto_trace(min_score=0.6)
    assert s.chain[0].startswith("c017") or s.chain[0].startswith("c018")
    assert s.chain[-1] == "c025-T001"
