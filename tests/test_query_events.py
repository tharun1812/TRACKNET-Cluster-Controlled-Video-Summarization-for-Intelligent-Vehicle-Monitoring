from tracknet.events import congestion_events, detect_events, stationary_events
from tracknet.query import parse_query, search
from tests.conftest import make_track


def test_parse_query_extracts_attributes():
    q = parse_query("Show me the WHITE truck on c021 going left", ["c017", "c021"])
    assert (q.color, q.vehicle_class, q.camera, q.direction) == ("white", "truck", "c021", "left")
    assert parse_query("a grey sedan").color == "silver"
    assert parse_query("blue jeep").vehicle_class == "car"
    assert parse_query("hello there").attributes() == {}
    assert parse_query("").attributes() == {}


def test_unknown_camera_is_ignored():
    assert parse_query("red car c999", ["c017"]).camera is None


def test_search_ranks_full_matches_first(store):
    store.add_tracks([
        make_track("c017-T001", "c017", 1, 5, color="red", cls="car"),
        make_track("c017-T002", "c017", 6, 12, color="red", cls="truck"),
        make_track("c018-T001", "c018", 1, 5, color="blue", cls="car"),
    ])
    res = search(store, parse_query("red car"))
    assert res[0][0].track_id == "c017-T001" and res[0][1] == 1.0
    assert {t.track_id for t, _ in res} == {"c017-T001", "c017-T002", "c018-T001"}   # partial matches follow
    only_cam = search(store, parse_query("red car"), camera="c018")
    assert [t.track_id for t, _ in only_cam] == ["c018-T001"]
    assert search(store, parse_query("nonsense")) == []


def test_congestion_detected_when_enough_vehicles_overlap():
    tracks = [make_track(f"c017-T{i}", "c017", 10 + i, 20 + i) for i in range(5)] + \
             [make_track("c017-T9", "c017", 100, 104)]
    events = congestion_events(tracks, "c017", threshold=4)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "congestion" and ev.detail["peak_vehicles"] == 5
    assert ev.start_ts >= 10 and ev.end_ts <= 26


def test_no_congestion_when_traffic_is_light():
    tracks = [make_track(f"c017-T{i}", "c017", i * 20, i * 20 + 5) for i in range(5)]
    assert congestion_events(tracks, "c017", threshold=3) == []
    assert congestion_events([], "c017") == []


def test_stationary_vehicle_event(store):
    parked = make_track("c017-T001", "c017", 0, 30, direction="static")
    moving = make_track("c017-T002", "c017", 0, 30)
    evs = stationary_events([parked, moving], "c017")
    assert len(evs) == 1 and evs[0].detail["track_id"] == "c017-T001"
    store.add_tracks([parked, moving])
    assert any(e.kind == "stationary_vehicle" for e in detect_events(store, "c017"))
