"""Video -> detections -> tracks -> metadata stored in the database."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config
from .db import Store, TrackRecord
from .features import color_histogram, direction_from_trajectory, dominant_color_name
from .tracker import DeepSortTracker, Track


def _clip_box(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def process_video(video_path: str | Path, camera_id: str, store: Store, detector,
                  images_dir: str | Path = "data/crops", tracker: DeepSortTracker | None = None,
                  embedder: Callable[[np.ndarray], np.ndarray] = color_histogram,
                  stride: int = 1, max_frames: int | None = None, min_track_frames: int = 5,
                  time_offset: float | None = None, replace: bool = True,
                  progress: Callable[[int, int], None] | None = None) -> dict:
    """Run detection + tracking over one video and store one TrackRecord per vehicle track.

    `embedder` is pluggable: swap in a ReID network that maps a BGR crop to a feature vector.
    Returns a dict with the created records and timing statistics.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    offset = config.CAMERAS.get(camera_id, {}).get("time_offset", 0.0) if time_offset is None else time_offset
    tracker = tracker or DeepSortTracker(max_age=max(5, int(fps / stride)), n_init=3)

    best: dict[int, tuple[float, np.ndarray]] = {}   # internal track id -> (area, best crop)
    frame_idx, processed = -1, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % stride:
            continue
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        for det in detections:
            x1, y1, x2, y2 = _clip_box(det.bbox, w, h)
            det.embedding = embedder(frame[y1:y2, x1:x2])
        for track in tracker.update(frame_idx, detections):
            x1, y1, x2, y2 = _clip_box(track.last_detection.bbox, w, h)
            area = float((x2 - x1) * (y2 - y1))
            if track.track_id not in best or area > best[track.track_id][0]:
                best[track.track_id] = (area, frame[y1:y2, x1:x2].copy())
        processed += 1
        if progress and total:
            progress(frame_idx + 1, total)
    cap.release()
    elapsed = time.perf_counter() - t0

    finished = [t for t in tracker.flush() if len(t.history) >= min_track_frames]
    finished.sort(key=lambda t: t.history[0][0])

    if replace:
        store.clear_tracks(camera_id)
    out_dir = Path(images_dir) / camera_id
    out_dir.mkdir(parents=True, exist_ok=True)
    start_n = store.next_track_number(camera_id)

    records: list[TrackRecord] = []
    for i, tr in enumerate(finished):
        records.append(_to_record(tr, camera_id, start_n + i, fps, offset, best, out_dir, embedder, video_path))
    store.add_tracks(records)

    return {
        "camera_id": camera_id, "video_path": video_path, "fps": fps, "frames_processed": processed,
        "tracks": records, "seconds": elapsed, "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
    }


def _to_record(tr: Track, camera_id: str, number: int, fps: float, offset: float,
               best: dict, out_dir: Path, embedder, video_path: str) -> TrackRecord:
    trajectory = []
    for frame_idx, bbox, _conf, _cls in tr.history:
        trajectory.append([round(frame_idx / fps, 3), float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)])
    classes = Counter(cls for *_rest, cls in tr.history)
    track_id = f"{camera_id}-T{number:03d}"

    crop = best.get(tr.track_id, (0, None))[1]
    image_path = None
    if crop is not None and crop.size:
        image_path = str(out_dir / f"{track_id}.jpg")
        cv2.imwrite(image_path, crop)
    embedding = tr.mean_embedding
    if embedding is None and crop is not None:
        embedding = embedder(crop)
    video_start, video_end = trajectory[0][0], trajectory[-1][0]
    return TrackRecord(
        track_id=track_id, camera_id=camera_id,
        start_ts=round(offset + video_start, 3), end_ts=round(offset + video_end, 3),
        video_start=video_start, video_end=video_end,
        vehicle_class=classes.most_common(1)[0][0],
        color=dominant_color_name(crop) if crop is not None else "unknown",
        direction=direction_from_trajectory(trajectory),
        embedding=[round(float(x), 5) for x in (embedding if embedding is not None else [])],
        trajectory=trajectory, image_path=image_path, video_path=video_path,
    )
