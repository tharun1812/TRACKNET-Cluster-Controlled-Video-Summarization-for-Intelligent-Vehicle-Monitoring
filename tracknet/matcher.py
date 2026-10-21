"""Cross-camera candidate matching: appearance, colour, time, direction and graph distance."""
from __future__ import annotations

from dataclasses import dataclass, asdict

from . import config
from .db import Store, TrackRecord
from .features import cosine_similarity, direction_compatibility
from .graph import CameraGraph

NEUTRALS = {"white", "silver"}
ARRIVAL_TOLERANCE = 1.0   # seconds a track may start "before" the previous one ended (overlapping views)


@dataclass
class MatchScore:
    candidate_id: str
    camera_id: str
    total: float
    appearance: float
    color: float
    temporal: float
    direction: float
    graph: float
    hops: int
    gap: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def color_score(a: str, b: str) -> float:
    if a == b and a != "unknown":
        return 1.0
    if "unknown" in (a, b):
        return 0.5
    if a in NEUTRALS and b in NEUTRALS:
        return 0.5   # white vs silver is easily confused under changing light
    return 0.0


def temporal_score(gap: float, window: tuple[float, float]) -> float | None:
    """1.0 near the middle of the expected window, decaying outside it. None = physically implausible."""
    lo, hi = window
    span = max(hi - lo, 1e-6)
    if lo <= gap <= hi:
        centre = (lo + hi) / 2
        return 1.0 - 0.3 * abs(gap - centre) / (span / 2)
    slack = config.TEMPORAL_SLACK * span
    dist = (lo - gap) if gap < lo else (gap - hi)
    if dist >= slack:
        return None
    return 0.7 * (1.0 - dist / slack)


def score_pair(src: TrackRecord, cand: TrackRecord, graph: CameraGraph, direction: str,
               weights: dict[str, float] | None = None) -> MatchScore | None:
    """Score `cand` as the same vehicle as `src`. direction='forward' means cand appears later than src."""
    weights = weights or config.MATCH_WEIGHTS
    earlier, later = (src, cand) if direction == "forward" else (cand, src)
    if earlier.camera_id == later.camera_id:
        return None
    window = graph.transit_window(earlier.camera_id, later.camera_id)
    if window is None:
        return None
    gap = later.start_ts - earlier.end_ts
    if gap < -ARRIVAL_TOLERANCE:
        return None
    t = temporal_score(max(gap, 0.0), window)
    if t is None:
        return None

    hops = graph.hops(earlier.camera_id, later.camera_id) or 1
    parts = {
        "appearance": cosine_similarity(src.embedding_array(), cand.embedding_array()),
        "color": color_score(src.color, cand.color),
        "temporal": t,
        "direction": direction_compatibility(src.direction, cand.direction),
        "graph": 1.0 / hops,
    }
    total = sum(weights[k] * parts[k] for k in weights)
    if src.vehicle_class != cand.vehicle_class:
        total *= config.CLASS_MISMATCH_PENALTY
    return MatchScore(candidate_id=cand.track_id, camera_id=cand.camera_id, total=total, hops=hops, gap=gap, **parts)


def find_candidates(src: TrackRecord, store: Store, graph: CameraGraph, direction: str,
                    max_hops: int = config.MAX_HOPS, exclude_ids: set[str] | None = None,
                    exclude_cameras: set[str] | None = None, top_k: int | None = 10,
                    min_score: float = 0.0) -> list[MatchScore]:
    """Expand the camera graph from src's camera (BFS) and rank tracks seen there."""
    exclude_ids = exclude_ids or set()
    exclude_cameras = exclude_cameras or set()
    cameras = graph.levels(src.camera_id, direction, max_hops)
    scored: list[MatchScore] = []
    for cam in cameras:
        if cam == src.camera_id or cam in exclude_cameras:
            continue
        for cand in store.list_tracks(camera_id=cam):
            if cand.track_id in exclude_ids:
                continue
            s = score_pair(src, cand, graph, direction)
            if s is not None and s.total >= min_score:
                scored.append(s)
    scored.sort(key=lambda s: s.total, reverse=True)
    return scored if top_k is None else scored[:top_k]
