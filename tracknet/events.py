"""Temporal event detection on stored tracks: congestion periods and stationary vehicles."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .db import Store, TrackRecord


@dataclass
class Event:
    kind: str                 # "congestion" | "stationary_vehicle"
    camera_id: str
    start_ts: float
    end_ts: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def congestion_events(tracks: list[TrackRecord], camera_id: str, threshold: int = 4,
                      min_duration: float = 3.0, step: float = 1.0) -> list[Event]:
    """Periods where at least `threshold` vehicles are on screen at once for `min_duration` seconds."""
    if not tracks:
        return []
    t0 = min(t.start_ts for t in tracks)
    t1 = max(t.end_ts for t in tracks)
    n = int(np.ceil((t1 - t0) / step)) + 1
    counts = np.zeros(n, dtype=int)
    for t in tracks:
        a = int((t.start_ts - t0) // step)
        b = int((t.end_ts - t0) // step)
        counts[a:b + 1] += 1

    events: list[Event] = []
    i = 0
    while i < n:
        if counts[i] >= threshold:
            j = i
            while j + 1 < n and counts[j + 1] >= threshold:
                j += 1
            start, end = t0 + i * step, t0 + (j + 1) * step
            if end - start >= min_duration:
                events.append(Event("congestion", camera_id, round(start, 2), round(end, 2),
                                    {"peak_vehicles": int(counts[i:j + 1].max())}))
            i = j + 1
        else:
            i += 1
    return events


def stationary_events(tracks: list[TrackRecord], camera_id: str, min_duration: float = 10.0) -> list[Event]:
    """Vehicles that stay in view without moving (possible breakdown or illegal parking)."""
    return [Event("stationary_vehicle", camera_id, t.start_ts, t.end_ts,
                  {"track_id": t.track_id, "vehicle_class": t.vehicle_class, "color": t.color})
            for t in tracks if t.direction == "static" and t.duration >= min_duration]


def detect_events(store: Store, camera_id: str, congestion_threshold: int = 4) -> list[Event]:
    tracks = store.list_tracks(camera_id=camera_id)
    events = congestion_events(tracks, camera_id, threshold=congestion_threshold)
    events += stationary_events(tracks, camera_id)
    return sorted(events, key=lambda e: e.start_ts)
