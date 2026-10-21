"""Synthetic multi-camera scenario: renders simple videos with known vehicle identities.

It lets the whole system (detection -> tracking -> metadata -> cross-camera matching -> summarization)
run end to end without any dataset, and gives ground truth for measuring matching quality.
It is a test harness, NOT real traffic footage: numbers from it show the pipeline works, not how it
performs on real CCTV.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from .db import Store, TrackRecord
from .detector import MotionDetector
from .graph import CameraGraph
from .pipeline import process_video
from .session import TrackerSession

FRAME_SIZE = (640, 360)
SIZES = {"motorcycle": (40, 30), "car": (90, 45), "truck": (130, 60), "bus": (170, 65)}
BODY_COLORS = {            # BGR
    "red": (30, 30, 200), "white": (235, 235, 235), "blue": (200, 80, 30), "black": (30, 30, 30),
    "silver": (160, 160, 160), "green": (50, 160, 50), "yellow": (40, 200, 220),
    "orange": (30, 120, 240), "purple": (160, 50, 130),
}
ACCENTS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 200, 0)]
LANES = [95, 180, 265]      # lane centre y
CLASS_WEIGHTS = (("car", 0.62), ("truck", 0.18), ("bus", 0.08), ("motorcycle", 0.12))


@dataclass
class Appearance:
    camera_id: str
    start: float
    end: float
    lane: int


@dataclass
class SyntheticVehicle:
    vid: int
    vehicle_class: str
    color: str
    accent: tuple
    direction: str
    appearances: list[Appearance] = field(default_factory=list)


def generate_scenario(n_vehicles: int = 30, seed: int = 7, graph: CameraGraph | None = None) -> list[SyntheticVehicle]:
    rng = random.Random(seed)
    graph = graph or CameraGraph()
    busy: dict[str, list[tuple[int, float, float]]] = {c: [] for c in graph.nodes}
    sources = [c for c in graph.nodes if not graph.neighbors(c, "backward")] or graph.nodes[:1]

    def schedule(cam: str, start: float, dur: float) -> tuple[float, int]:
        for _ in range(60):
            free = [i for i, _y in enumerate(LANES)
                    if all(not (l == i and start < e + 1.0 and start + dur > s - 1.0) for l, s, e in busy[cam])]
            if free:
                lane = rng.choice(free)
                busy[cam].append((lane, start, start + dur))
                return start, lane
            start += 1.0
        raise RuntimeError("Could not schedule vehicle; lower n_vehicles")

    vehicles: list[SyntheticVehicle] = []
    clock = 4.0
    classes, weights = zip(*CLASS_WEIGHTS)
    for vid in range(n_vehicles):
        clock += rng.uniform(3.0, 6.0)
        v = SyntheticVehicle(
            vid=vid, vehicle_class=rng.choices(classes, weights)[0], color=rng.choice(list(BODY_COLORS)),
            accent=rng.choice(ACCENTS), direction="right")
        cam, start = rng.choice(sources), clock
        while True:
            dur = rng.uniform(4.5, 7.0)
            start, lane = schedule(cam, start, dur)
            v.appearances.append(Appearance(cam, round(start, 2), round(start + dur, 2), lane))
            options = graph.neighbors(cam, "forward")
            if not options or (len(v.appearances) >= 2 and rng.random() < 0.25):
                break
            nxt = rng.choice(options)
            lo, hi = graph.forward[cam][nxt]
            start = start + dur + rng.uniform(lo + 1.0, max(lo + 1.5, hi - 6.0))
            cam = nxt
        vehicles.append(v)
    return vehicles


def _draw_vehicle(frame: np.ndarray, cx: float, cy: float, v: SyntheticVehicle, gain: float) -> None:
    w, h = SIZES[v.vehicle_class]
    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    body = tuple(int(np.clip(c * gain, 0, 255)) for c in BODY_COLORS[v.color])
    cv2.rectangle(frame, (x1, y1), (x2, y2), body, -1)
    ax1, ax2 = int(x1 + 0.35 * w), int(x1 + 0.65 * w)
    ay1, ay2 = int(y1 + 0.10 * h), int(y1 + 0.35 * h)
    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), v.accent, -1)
    for wx in (int(x1 + 0.2 * w), int(x1 + 0.8 * w)):
        cv2.circle(frame, (wx, y2), max(3, h // 8), (15, 15, 15), -1)


def render_camera_video(path: str | Path, camera_id: str, vehicles: list[SyntheticVehicle], duration: float,
                        fps: int = 10, size: tuple[int, int] = FRAME_SIZE, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    gain = float(rng.uniform(0.92, 1.08))        # per-camera exposure difference
    W, H = size
    bg = np.full((H, W, 3), 105, dtype=np.uint8)
    for y in (140, 225):
        cv2.line(bg, (0, y), (W, y), (190, 190, 190), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    items = [(v, a) for v in vehicles for a in v.appearances if a.camera_id == camera_id]
    # a small pool of pre-noised backgrounds is much faster than generating noise for every frame
    pool = [np.clip(bg.astype(np.int16) + rng.normal(0, 2, bg.shape).astype(np.int16), 0, 255).astype(np.uint8)
            for _ in range(12)]
    for i in range(int(duration * fps)):
        t = i / fps
        frame = pool[int(rng.integers(len(pool)))].copy()
        for v, a in items:
            if a.start <= t <= a.end:
                p = (t - a.start) / (a.end - a.start)
                w = SIZES[v.vehicle_class][0]
                cx = -w / 2 + p * (W + w)
                _draw_vehicle(frame, cx, LANES[a.lane], v, gain)
        cv2.putText(frame, f"{camera_id}  {t:6.1f}s", (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def build_demo(out_dir: str | Path, store: Store, n_vehicles: int = 30, seed: int = 7, fps: int = 10) -> dict:
    """Render videos, run the real detection+tracking pipeline on them, fill the database."""
    out_dir = Path(out_dir)
    graph = CameraGraph()
    vehicles = generate_scenario(n_vehicles, seed, graph)
    duration = max(a.end for v in vehicles for a in v.appearances) + 4.0

    store.clear_tracks()
    stats = {}
    for cam in graph.nodes:
        video = out_dir / "videos" / f"{cam}.mp4"
        render_camera_video(video, cam, vehicles, duration, fps=fps, seed=seed + int(cam[1:]))
        res = process_video(video, cam, store, MotionDetector(), images_dir=out_dir / "crops")
        stats[cam] = {"tracks": len(res["tracks"]), "processing_fps": round(res["processing_fps"], 1)}

    truth = {"seed": seed, "fps": fps, "duration": duration,
             "vehicles": [{"vid": v.vid, "vehicle_class": v.vehicle_class, "color": v.color,
                           "appearances": [asdict(a) for a in v.appearances]} for v in vehicles]}
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return {"ground_truth": truth, "stats": stats}


# ---- evaluation -------------------------------------------------------------------------------
def _temporal_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def map_tracks_to_truth(store: Store, truth: dict, min_iou: float = 0.5) -> dict[str, int]:
    """track_id -> ground-truth vehicle id, by temporal overlap on the same camera."""
    mapping: dict[str, int] = {}
    for t in store.list_tracks():
        best, best_iou = None, min_iou
        for v in truth["vehicles"]:
            for a in v["appearances"]:
                if a["camera_id"] != t.camera_id:
                    continue
                iou = _temporal_iou(t.start_ts, t.end_ts, a["start"], a["end"])
                if iou > best_iou:
                    best, best_iou = v["vid"], iou
        if best is not None:
            mapping[t.track_id] = best
    return mapping


def evaluate_pipeline(store: Store, truth: dict, graph: CameraGraph | None = None, min_score: float = 0.55) -> dict:
    graph = graph or CameraGraph()
    mapping = map_tracks_to_truth(store, truth)
    tracks = store.list_tracks()
    n_true_app = sum(len(v["appearances"]) for v in truth["vehicles"])

    detected = len({(mapping[t.track_id], t.camera_id) for t in tracks if t.track_id in mapping})
    by_vid_cam: dict[tuple[int, str], list[TrackRecord]] = {}
    for t in tracks:
        if t.track_id in mapping:
            by_vid_cam.setdefault((mapping[t.track_id], t.camera_id), []).append(t)

    # cross-camera linking: start from each vehicle's first track and auto-trace forward
    pred_links, true_links = set(), set()
    per_query_ms: list[float] = []
    for v in truth["vehicles"]:
        apps = sorted(v["appearances"], key=lambda a: a["start"])
        first = by_vid_cam.get((v["vid"], apps[0]["camera_id"]))
        for a, b in zip(apps, apps[1:]):
            ta, tb = by_vid_cam.get((v["vid"], a["camera_id"])), by_vid_cam.get((v["vid"], b["camera_id"]))
            if ta and tb:
                true_links.add((ta[0].track_id, tb[0].track_id))
        if not first:
            continue
        session = TrackerSession(store, graph, first[0].track_id)
        t0 = time.perf_counter()
        session.auto_trace(min_score=min_score)
        per_query_ms.append((time.perf_counter() - t0) * 1000)
        chain = session.chain
        pred_links.update(zip(chain, chain[1:]))

    correct = pred_links & true_links
    precision = len(correct) / len(pred_links) if pred_links else 0.0
    recall = len(correct) / len(true_links) if true_links else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "vehicles": len(truth["vehicles"]),
        "true_appearances": n_true_app,
        "tracks": len(tracks),
        "appearances_detected": detected,
        "detection_recall": round(detected / n_true_app, 3) if n_true_app else 0.0,
        "fragmentation": round(len(mapping) / max(detected, 1), 3),
        "link_precision": round(precision, 3),
        "link_recall": round(recall, 3),
        "link_f1": round(f1, 3),
        "mean_trace_ms": round(float(np.mean(per_query_ms)), 2) if per_query_ms else None,
    }


def scalability_benchmark(sizes=(100, 500, 1000, 2000), seed: int = 1) -> list[dict]:
    """Time one candidate search against databases of growing size (random synthetic tracks)."""
    from .matcher import find_candidates
    rng = np.random.default_rng(seed)
    graph = CameraGraph()
    cams = graph.nodes
    rows = []
    for n in sizes:
        store = Store(":memory:")
        for i in range(n):
            cam = cams[i % len(cams)]
            start = float(rng.uniform(0, n * 0.5))
            emb = rng.random(config_dim()).astype(np.float32)
            emb /= np.linalg.norm(emb)
            store.add_track(TrackRecord(
                track_id=f"{cam}-T{i:05d}", camera_id=cam, start_ts=start, end_ts=start + 5, video_start=start,
                video_end=start + 5, vehicle_class="car", color=str(rng.choice(list(BODY_COLORS))),
                direction="right", embedding=[float(x) for x in emb], trajectory=[]))
        src = store.list_tracks(camera_id=cams[0], limit=1)[0]
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            find_candidates(src, store, graph, "forward")
        rows.append({"tracks": n, "search_ms": round((time.perf_counter() - t0) / reps * 1000, 2)})
        store.close()
    return rows


def config_dim() -> int:
    from .features import EMBEDDING_DIM
    return EMBEDDING_DIM
