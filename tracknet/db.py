"""SQLite storage for track metadata, manual labels and summary records."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    video_start REAL NOT NULL,
    video_end REAL NOT NULL,
    vehicle_class TEXT NOT NULL,
    color TEXT NOT NULL,
    direction TEXT NOT NULL,
    embedding TEXT NOT NULL,
    trajectory TEXT NOT NULL,
    image_path TEXT,
    video_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera_id, start_ts);
CREATE TABLE IF NOT EXISTS labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_chain TEXT NOT NULL,
    label_type TEXT NOT NULL,
    annotated_by TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    reference TEXT NOT NULL,
    output_path TEXT NOT NULL,
    intervals TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass
class TrackRecord:
    track_id: str
    camera_id: str
    start_ts: float            # shared clock, seconds
    end_ts: float
    video_start: float         # seconds inside the camera's own video
    video_end: float
    vehicle_class: str
    color: str
    direction: str
    embedding: list[float] = field(default_factory=list)
    trajectory: list[list[float]] = field(default_factory=list)   # [[t, cx, cy], ...]
    image_path: str | None = None
    video_path: str | None = None

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    def embedding_array(self) -> np.ndarray:
        return np.asarray(self.embedding, dtype=np.float32)

    def to_public(self) -> dict:
        """JSON-friendly view for the API (embedding and trajectory omitted for size)."""
        d = asdict(self)
        d.pop("embedding")
        d.pop("trajectory")
        d["duration"] = round(self.duration, 2)
        return d


class Store:
    """Thread-safe wrapper around one SQLite file."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---- tracks -------------------------------------------------------------------------------
    def add_track(self, t: TrackRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t.track_id, t.camera_id, t.start_ts, t.end_ts, t.video_start, t.video_end,
                 t.vehicle_class, t.color, t.direction, json.dumps(t.embedding),
                 json.dumps(t.trajectory), t.image_path, t.video_path),
            )
            self._conn.commit()

    def add_tracks(self, tracks: Iterable[TrackRecord]) -> None:
        for t in tracks:
            self.add_track(t)

    @staticmethod
    def _row_to_track(r: sqlite3.Row) -> TrackRecord:
        return TrackRecord(
            track_id=r["track_id"], camera_id=r["camera_id"], start_ts=r["start_ts"], end_ts=r["end_ts"],
            video_start=r["video_start"], video_end=r["video_end"], vehicle_class=r["vehicle_class"],
            color=r["color"], direction=r["direction"], embedding=json.loads(r["embedding"]),
            trajectory=json.loads(r["trajectory"]), image_path=r["image_path"], video_path=r["video_path"],
        )

    def get_track(self, track_id: str) -> TrackRecord | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM tracks WHERE track_id=?", (track_id,)).fetchone()
        return self._row_to_track(r) if r else None

    def list_tracks(self, camera_id: str | None = None, vehicle_class: str | None = None,
                    color: str | None = None, limit: int | None = None) -> list[TrackRecord]:
        sql, args = "SELECT * FROM tracks WHERE 1=1", []
        if camera_id:
            sql += " AND camera_id=?"
            args.append(camera_id)
        if vehicle_class:
            sql += " AND vehicle_class=?"
            args.append(vehicle_class)
        if color:
            sql += " AND color=?"
            args.append(color)
        sql += " ORDER BY start_ts, track_id"
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_track(r) for r in rows]

    def count_tracks(self, camera_id: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) FROM tracks", []
        if camera_id:
            sql += " WHERE camera_id=?"
            args.append(camera_id)
        with self._lock:
            return int(self._conn.execute(sql, args).fetchone()[0])

    def clear_tracks(self, camera_id: str | None = None) -> None:
        with self._lock:
            if camera_id:
                self._conn.execute("DELETE FROM tracks WHERE camera_id=?", (camera_id,))
            else:
                self._conn.execute("DELETE FROM tracks")
            self._conn.commit()

    def next_track_number(self, camera_id: str) -> int:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM tracks WHERE camera_id=?", (camera_id,)).fetchone()[0]
        return int(n) + 1

    def camera_video(self, camera_id: str) -> tuple[str, float] | None:
        """(video_path, clock offset) for a camera, derived from any of its tracks."""
        with self._lock:
            r = self._conn.execute(
                "SELECT video_path, start_ts, video_start FROM tracks "
                "WHERE camera_id=? AND video_path IS NOT NULL LIMIT 1", (camera_id,)).fetchone()
        if not r:
            return None
        return r["video_path"], float(r["start_ts"] - r["video_start"])

    # ---- labels -------------------------------------------------------------------------------
    def add_label(self, track_chain: list[str], label_type: str, annotated_by: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO labels (track_chain,label_type,annotated_by,created_at) VALUES (?,?,?,?)",
                (json.dumps(track_chain), label_type, annotated_by, time.time()))
            self._conn.commit()
            return int(cur.lastrowid)

    def list_labels(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM labels ORDER BY label_id DESC").fetchall()
        return [{"label_id": r["label_id"], "track_chain": json.loads(r["track_chain"]),
                 "label_type": r["label_type"], "annotated_by": r["annotated_by"],
                 "created_at": r["created_at"]} for r in rows]

    # ---- summaries ----------------------------------------------------------------------------
    def add_summary(self, kind: str, reference: str, output_path: str, intervals: list[dict]) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO summaries (kind,reference,output_path,intervals,created_at) VALUES (?,?,?,?,?)",
                (kind, reference, output_path, json.dumps(intervals), time.time()))
            self._conn.commit()
            return int(cur.lastrowid)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
