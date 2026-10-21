import pytest

from tracknet.api import create_app
from tracknet.db import Store
from tests.conftest import make_track


@pytest.fixture()
def client(tmp_path):
    db = tmp_path / "t.db"
    seed = Store(db)
    seed.add_tracks([
        make_track("c017-T001", "c017", 10, 16),
        make_track("c018-T001", "c018", 24, 30),
        make_track("c018-T002", "c018", 24, 30, color="blue", cls="truck"),
        make_track("c021-T001", "c021", 40, 46),
    ])
    seed.close()
    app = create_app(db, tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_health_and_listings(client):
    assert client.get("/api/health").get_json() == {"status": "ok", "tracks": 4}
    cams = client.get("/api/cameras").get_json()
    assert [c["camera_id"] for c in cams] == ["c017", "c018", "c021", "c025", "c026"]
    assert {c["cluster"] for c in client.get("/api/clusters").get_json()} == {"A", "B"}
    assert len(client.get("/api/tracks?camera=c018").get_json()) == 2
    assert client.get("/api/tracks?limit=abc").status_code == 400


def test_track_detail_and_missing_image(client):
    t = client.get("/api/tracks/c017-T001").get_json()
    assert t["trajectory"] and "embedding" not in t
    assert client.get("/api/tracks/nope").status_code == 404
    assert client.get("/api/tracks/c017-T001/image").status_code == 404


def test_search_validation_and_results(client):
    assert client.post("/api/search", json={}).status_code == 400
    assert client.post("/api/search", json={"query": "hello"}).status_code == 400
    assert client.post("/api/search", json={"query": "red car", "camera": "zzz"}).status_code == 400
    res = client.post("/api/search", json={"query": "blue truck"}).get_json()
    assert res["parsed"]["color"] == "blue" and res["results"][0]["track_id"] == "c018-T002"


def test_full_tracing_flow(client):
    assert client.post("/api/sessions", json={}).status_code == 400
    assert client.post("/api/sessions", json={"track_id": "nope"}).status_code == 404
    sid = client.post("/api/sessions", json={"track_id": "c017-T001"}).get_json()["session_id"]

    cands = client.get(f"/api/sessions/{sid}/candidates?direction=forward").get_json()["candidates"]
    assert cands[0]["candidate_id"] == "c018-T001" and cands[0]["track"]["camera_id"] == "c018"
    assert client.get(f"/api/sessions/{sid}/candidates?direction=up").status_code == 400

    bad = client.post(f"/api/sessions/{sid}/confirm", json={"direction": "forward", "candidate_id": "zzz"})
    assert bad.status_code == 400
    ok = client.post(f"/api/sessions/{sid}/confirm", json={"direction": "forward", "candidate_id": "c018-T001"})
    assert [t["track_id"] for t in ok.get_json()["session"]["chain"]] == ["c017-T001", "c018-T001"]

    skip = client.post(f"/api/sessions/{sid}/skip", json={"direction": "forward", "candidate_id": "c021-T001"})
    assert skip.status_code == 200
    auto = client.post(f"/api/sessions/{sid}/auto", json={"min_score": 0.6}).get_json()
    assert auto["added"] == []
    assert client.get("/api/sessions/unknown").status_code == 404
    # tracks have no video attached, so summarising gives a clear 422 instead of crashing
    assert client.post(f"/api/sessions/{sid}/summary").status_code == 422


def test_labels(client):
    assert client.post("/api/labels", json={"track_chain": ["c017-T001"], "label_type": "correct"}).status_code == 400
    assert client.post("/api/labels", json={"track_chain": ["a", "b"], "label_type": "maybe"}).status_code == 400
    assert client.post("/api/labels", json={"track_chain": ["a", "b"], "label_type": "correct"}).status_code == 404
    ok = client.post("/api/labels", json={"track_chain": ["c017-T001", "c018-T001"], "label_type": "correct",
                                          "annotated_by": "tester"})
    assert ok.status_code == 201
    assert client.get("/api/labels").get_json()[0]["annotated_by"] == "tester"


def test_analytics_events_and_summary_404(client):
    a = client.get("/api/analytics").get_json()
    assert a["total_tracks"] == 4 and a["by_color"]["red"] == 3
    assert client.get("/api/events").status_code == 200
    assert client.get("/api/events?camera=zzz").status_code == 400
    assert client.get("/api/summaries/missing.mp4").status_code == 404
    assert client.post("/api/clusters/Z/summary").status_code == 404


def test_serves_built_dashboard_when_present(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>dashboard</html>")
    (dist / "app.js").write_text("console.log(1)")
    app = create_app(tmp_path / "x.db", tmp_path, static_dir=dist)
    c = app.test_client()
    assert b"dashboard" in c.get("/").data
    assert b"console.log" in c.get("/app.js").data
    assert b"dashboard" in c.get("/some/client/route").data          # SPA fallback
    assert c.get("/api/does-not-exist").status_code == 404
    assert c.get("/api/health").status_code == 200
