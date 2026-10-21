"""Video summarization: cut only the relevant segments and join them into one short clip.

* Vehicle-specific summary: the segments where each track of a confirmed chain is visible.
* Cluster-wise summary: the segments around detected events (congestion, stationary vehicles) in a cluster.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

from .db import Store, TrackRecord
from .events import detect_events


@dataclass
class Interval:
    camera_id: str
    video_path: str
    start: float          # seconds inside the video
    end: float
    label: str
    order_ts: float       # shared-clock time, used to order segments chronologically

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def merge_intervals(intervals: list[Interval], gap: float = 0.5) -> list[Interval]:
    """Merge overlapping or nearly touching segments of the same video."""
    by_video: dict[str, list[Interval]] = {}
    for iv in intervals:
        by_video.setdefault(iv.video_path, []).append(iv)
    merged: list[Interval] = []
    for video, items in by_video.items():
        items.sort(key=lambda i: i.start)
        cur = Interval(**asdict(items[0]))
        for iv in items[1:]:
            if iv.start <= cur.end + gap:
                cur.end = max(cur.end, iv.end)
                if iv.label not in cur.label:
                    cur.label = f"{cur.label}, {iv.label}"
            else:
                merged.append(cur)
                cur = Interval(**asdict(iv))
        merged.append(cur)
    return sorted(merged, key=lambda i: (i.order_ts, i.camera_id))


def intervals_for_tracks(tracks: list[TrackRecord], padding: float = 1.0) -> list[Interval]:
    out = []
    for t in tracks:
        if not t.video_path:
            continue
        out.append(Interval(t.camera_id, t.video_path, max(0.0, t.video_start - padding),
                            t.video_end + padding, t.track_id, t.start_ts - padding))
    return merge_intervals(out)


def intervals_for_cluster(store: Store, camera_ids: list[str], padding: float = 2.0,
                          congestion_threshold: int = 4) -> list[Interval]:
    out = []
    for cam in camera_ids:
        video = store.camera_video(cam)
        if video is None:
            continue
        path, offset = video
        for ev in detect_events(store, cam, congestion_threshold):
            out.append(Interval(cam, path, max(0.0, ev.start_ts - offset - padding), ev.end_ts - offset + padding,
                                ev.kind.replace("_", " "), ev.start_ts))
    return merge_intervals(out)


def _video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return frames / fps if fps else 0.0


def _transcode_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so browsers can play the clip. Returns False if no ffmpeg is available."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists() and dst.stat().st_size > 0


def render_summary(intervals: list[Interval], out_path: str | Path, size: tuple[int, int] = (640, 360),
                   fps: float | None = None) -> dict:
    """Write the joined clip to `out_path` and return statistics about the summary."""
    if not intervals:
        raise ValueError("No footage to summarize: the selected tracks have no video attached.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tracknet_"))
    raw = tmp_dir / "raw.mp4"

    first = cv2.VideoCapture(intervals[0].video_path)
    out_fps = fps or (first.get(cv2.CAP_PROP_FPS) or 10.0)
    first.release()
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, size)
    if not writer.isOpened():
        raise RuntimeError("Could not open the video writer")

    frames_written = 0
    for iv in intervals:
        cap = cv2.VideoCapture(iv.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {iv.video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or out_fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(iv.start * src_fps))
        pos = iv.start
        while pos <= iv.end:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, size)
            cv2.rectangle(frame, (0, 0), (size[0], 26), (20, 20, 20), -1)
            cv2.putText(frame, f"{iv.camera_id}  {iv.label}  t={pos:5.1f}s", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            frames_written += 1
            pos += 1.0 / src_fps
        cap.release()
    writer.release()

    if frames_written == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("The selected segments contain no readable frames.")

    if not _transcode_h264(raw, out_path):
        shutil.copyfile(raw, out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary_seconds = frames_written / out_fps
    source_seconds = sum(_video_duration(p) for p in {iv.video_path for iv in intervals})
    return {
        "output_path": str(out_path),
        "segments": len(intervals),
        "summary_seconds": round(summary_seconds, 2),
        "source_seconds": round(source_seconds, 2),
        "reduction": round(1 - summary_seconds / source_seconds, 4) if source_seconds else None,
        "intervals": [iv.to_dict() for iv in intervals],
    }
