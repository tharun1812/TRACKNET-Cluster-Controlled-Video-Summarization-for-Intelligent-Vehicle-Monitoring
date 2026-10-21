"""Flask REST API for search, cross-camera tracing, analytics and summaries."""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from .db import Store
from .events import detect_events
from .graph import CameraGraph
from .query import parse_query, search
from .session import TrackerSession
from .summarizer import intervals_for_cluster, intervals_for_tracks, render_summary


def create_app(db_path: str | Path = "data/tracknet.db", data_dir: str | Path = "data",
               graph: CameraGraph | None = None, static_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    store = Store(db_path)
    graph = graph or CameraGraph()
    data_dir = Path(data_dir)
    summaries_dir = data_dir / "summaries"
    sessions: dict[str, TrackerSession] = {}
    lock = threading.Lock()
    app.extensions["tracknet_store"] = store

    def error(message: str, status: int = 400):
        return jsonify({"error": message}), status

    def get_session(sid: str) -> TrackerSession | None:
        with lock:
            return sessions.get(sid)

    # ---- system ----------------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "tracks": store.count_tracks()})

    @app.get("/api/cameras")
    def cameras():
        rows = []
        for cam_id, meta in graph.cameras.items():
            rows.append({"camera_id": cam_id, **{k: v for k, v in meta.items() if k != "time_offset"},
                         "tracks": store.count_tracks(cam_id),
                         "next": graph.neighbors(cam_id, "forward"), "previous": graph.neighbors(cam_id, "backward")})
        return jsonify(sorted(rows, key=lambda r: r["camera_id"]))

    @app.get("/api/clusters")
    def clusters():
        grouped: dict[str, list[str]] = {}
        for cam_id, meta in graph.cameras.items():
            grouped.setdefault(meta.get("cluster", "default"), []).append(cam_id)
        return jsonify([{"cluster": c, "cameras": sorted(cams), "tracks": sum(store.count_tracks(x) for x in cams)}
                        for c, cams in sorted(grouped.items())])

    # ---- tracks ----------------------------------------------------------------------------
    @app.get("/api/tracks")
    def tracks():
        try:
            limit = min(int(request.args.get("limit", 100)), 500)
        except ValueError:
            return error("limit must be an integer")
        rows = store.list_tracks(request.args.get("camera"), request.args.get("vehicle_class"),
                                 request.args.get("color"), limit)
        return jsonify([t.to_public() for t in rows])

    @app.get("/api/tracks/<track_id>")
    def track_detail(track_id: str):
        t = store.get_track(track_id)
        if t is None:
            return error("Track not found", 404)
        return jsonify({**t.to_public(), "trajectory": t.trajectory})

    @app.get("/api/tracks/<track_id>/image")
    def track_image(track_id: str):
        t = store.get_track(track_id)
        if t is None or not t.image_path or not Path(t.image_path).exists():
            return error("Image not found", 404)
        return send_file(Path(t.image_path).resolve(), mimetype="image/jpeg")

    # ---- search ----------------------------------------------------------------------------
    @app.post("/api/search")
    def do_search():
        body = request.get_json(silent=True) or {}
        text = str(body.get("query", "")).strip()
        if not text:
            return error("Provide a non-empty 'query', for example 'red car'.")
        camera = body.get("camera") or None
        if camera and camera not in graph.cameras:
            return error(f"Unknown camera: {camera}")
        parsed = parse_query(text, list(graph.cameras))
        if not parsed.attributes():
            return error("No colour, vehicle type or direction found. Try 'white truck' or 'red car'.")
        results = search(store, parsed, camera=camera, limit=int(body.get("limit", 20)))
        return jsonify({"parsed": parsed.to_dict(),
                        "results": [{**t.to_public(), "match": round(score, 2)} for t, score in results]})

    # ---- cross-camera tracing ---------------------------------------------------------------
    @app.post("/api/sessions")
    def start_session():
        body = request.get_json(silent=True) or {}
        track_id = body.get("track_id")
        if not track_id:
            return error("'track_id' is required")
        try:
            s = TrackerSession(store, graph, track_id)
        except KeyError:
            return error("Track not found", 404)
        with lock:
            sessions[s.session_id] = s
        return jsonify(s.to_dict()), 201

    @app.get("/api/sessions/<sid>")
    def session_detail(sid: str):
        s = get_session(sid)
        return jsonify(s.to_dict()) if s else error("Session not found", 404)

    @app.get("/api/sessions/<sid>/candidates")
    def session_candidates(sid: str):
        s = get_session(sid)
        if not s:
            return error("Session not found", 404)
        direction = request.args.get("direction", "forward")
        try:
            cands = s.candidates(direction, top_k=int(request.args.get("limit", 8)))
        except ValueError as exc:
            return error(str(exc))
        out = []
        for c in cands:
            t = store.get_track(c.candidate_id)
            out.append({**c.to_dict(), "track": t.to_public() if t else None})
        return jsonify({"direction": direction, "frontier": s.frontier[direction], "candidates": out})

    @app.post("/api/sessions/<sid>/confirm")
    def session_confirm(sid: str):
        s = get_session(sid)
        if not s:
            return error("Session not found", 404)
        body = request.get_json(silent=True) or {}
        try:
            match = s.confirm(body.get("direction", ""), body.get("candidate_id", ""))
        except ValueError as exc:
            return error(str(exc))
        return jsonify({"match": match.to_dict(), "session": s.to_dict()})

    @app.post("/api/sessions/<sid>/skip")
    def session_skip(sid: str):
        s = get_session(sid)
        if not s:
            return error("Session not found", 404)
        body = request.get_json(silent=True) or {}
        try:
            s.reject(body.get("direction", ""), body.get("candidate_id", ""))
        except ValueError as exc:
            return error(str(exc))
        return jsonify(s.to_dict())

    @app.post("/api/sessions/<sid>/auto")
    def session_auto(sid: str):
        s = get_session(sid)
        if not s:
            return error("Session not found", 404)
        body = request.get_json(silent=True) or {}
        added = s.auto_trace(min_score=float(body.get("min_score", 0.55)))
        return jsonify({"added": added, "session": s.to_dict()})

    @app.post("/api/sessions/<sid>/summary")
    def session_summary(sid: str):
        s = get_session(sid)
        if not s:
            return error("Session not found", 404)
        intervals = intervals_for_tracks(s.chain_tracks())
        out = summaries_dir / f"vehicle_{s.session_id}.mp4"
        try:
            stats = render_summary(intervals, out)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return error(str(exc), 422)
        store.add_summary("vehicle", s.session_id, str(out), stats["intervals"])
        return jsonify({**stats, "url": f"/api/summaries/{out.name}"})

    @app.post("/api/clusters/<cluster>/summary")
    def cluster_summary(cluster: str):
        cams = sorted(c for c, m in graph.cameras.items() if m.get("cluster") == cluster)
        if not cams:
            return error("Unknown cluster", 404)
        body = request.get_json(silent=True) or {}
        intervals = intervals_for_cluster(store, cams, congestion_threshold=int(body.get("threshold", 3)))
        out = summaries_dir / f"cluster_{cluster}.mp4"
        try:
            stats = render_summary(intervals, out)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return error(str(exc), 422)
        store.add_summary("cluster", cluster, str(out), stats["intervals"])
        return jsonify({**stats, "url": f"/api/summaries/{out.name}"})

    @app.get("/api/summaries/<name>")
    def summary_file(name: str):
        path = (summaries_dir / Path(name).name).resolve()
        if not path.exists() or path.parent != summaries_dir.resolve():
            return error("Summary not found", 404)
        return send_file(path, mimetype="video/mp4", conditional=True)

    # ---- analytics, events, labels ----------------------------------------------------------
    @app.get("/api/analytics")
    def analytics():
        all_tracks = store.list_tracks()
        return jsonify({
            "total_tracks": len(all_tracks),
            "by_camera": dict(Counter(t.camera_id for t in all_tracks)),
            "by_class": dict(Counter(t.vehicle_class for t in all_tracks)),
            "by_color": dict(Counter(t.color for t in all_tracks)),
            "by_direction": dict(Counter(t.direction for t in all_tracks)),
        })

    @app.get("/api/events")
    def events():
        camera = request.args.get("camera")
        cams = [camera] if camera else sorted(graph.cameras)
        if camera and camera not in graph.cameras:
            return error(f"Unknown camera: {camera}")
        threshold = int(request.args.get("threshold", 3))
        out = [e.to_dict() for c in cams for e in detect_events(store, c, threshold)]
        return jsonify(sorted(out, key=lambda e: e["start_ts"]))

    @app.post("/api/labels")
    def add_label():
        body = request.get_json(silent=True) or {}
        chain = body.get("track_chain")
        label_type = body.get("label_type")
        if not isinstance(chain, list) or len(chain) < 2 or not all(isinstance(x, str) for x in chain):
            return error("'track_chain' must be a list of at least two track ids")
        if label_type not in {"correct", "incorrect", "uncertain"}:
            return error("'label_type' must be 'correct', 'incorrect' or 'uncertain'")
        missing = [x for x in chain if store.get_track(x) is None]
        if missing:
            return error(f"Unknown tracks: {', '.join(missing)}", 404)
        label_id = store.add_label(chain, label_type, str(body.get("annotated_by", "anonymous")))
        return jsonify({"label_id": label_id}), 201

    @app.get("/api/labels")
    def list_labels():
        return jsonify(store.list_labels())

    # ---- built dashboard (frontend/dist) ----------------------------------------------------
    dist = Path(static_dir) if static_dir else Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if (dist / "index.html").exists():
        @app.get("/")
        def index():
            return send_from_directory(dist, "index.html")

        @app.get("/<path:path>")
        def assets(path: str):
            if path.startswith("api/"):
                return error("Not found", 404)
            if (dist / path).is_file():
                return send_from_directory(dist, path)
            return send_from_directory(dist, "index.html")

    return app
