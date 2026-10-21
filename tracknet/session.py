"""Interactive cross-camera tracing session: confirm or skip candidates, growing a vehicle chain."""
from __future__ import annotations

import time
import uuid

from . import config
from .db import Store, TrackRecord
from .graph import CameraGraph
from .matcher import MatchScore, find_candidates

DIRECTIONS = ("forward", "backward")


class TrackerSession:
    def __init__(self, store: Store, graph: CameraGraph, query_track_id: str,
                 max_hops: int = config.MAX_HOPS, session_id: str | None = None):
        query = store.get_track(query_track_id)
        if query is None:
            raise KeyError(f"Unknown track: {query_track_id}")
        self.store, self.graph, self.max_hops = store, graph, max_hops
        self.session_id = session_id or uuid.uuid4().hex[:10]
        self.query_track_id = query_track_id
        self.chain: list[str] = [query_track_id]                       # time-ordered
        self.frontier = {"forward": query_track_id, "backward": query_track_id}
        self.rejected: dict[str, set[str]] = {d: set() for d in DIRECTIONS}
        self.log: list[dict] = []
        self.created_at = time.time()

    # -- state ----------------------------------------------------------------------------------
    def chain_tracks(self) -> list[TrackRecord]:
        return [t for t in (self.store.get_track(i) for i in self.chain) if t is not None]

    @property
    def visited_cameras(self) -> set[str]:
        return {t.camera_id for t in self.chain_tracks()}

    def candidates(self, direction: str, top_k: int | None = 8) -> list[MatchScore]:
        self._check_direction(direction)
        src = self.store.get_track(self.frontier[direction])
        if src is None:
            return []
        return find_candidates(
            src, self.store, self.graph, direction, max_hops=self.max_hops,
            exclude_ids=set(self.chain) | self.rejected[direction],
            exclude_cameras=self.visited_cameras, top_k=top_k)

    # -- actions --------------------------------------------------------------------------------
    def confirm(self, direction: str, candidate_id: str) -> MatchScore:
        self._check_direction(direction)
        match = next((c for c in self.candidates(direction, top_k=None) if c.candidate_id == candidate_id), None)
        if match is None:
            raise ValueError(f"{candidate_id} is not a valid {direction} candidate")
        if direction == "forward":
            self.chain.append(candidate_id)
        else:
            self.chain.insert(0, candidate_id)
        self.frontier[direction] = candidate_id
        self.log.append({"action": "confirm", "direction": direction, "candidate": candidate_id,
                         "score": round(match.total, 4), "at": time.time()})
        return match

    def reject(self, direction: str, candidate_id: str) -> None:
        self._check_direction(direction)
        self.rejected[direction].add(candidate_id)
        self.log.append({"action": "skip", "direction": direction, "candidate": candidate_id, "at": time.time()})

    def auto_trace(self, min_score: float = 0.55, max_steps: int = 10) -> list[str]:
        """Repeatedly accept the best candidate (both directions) while its score >= min_score."""
        added: list[str] = []
        for direction in DIRECTIONS:
            for _ in range(max_steps):
                cands = self.candidates(direction, top_k=1)
                if not cands or cands[0].total < min_score:
                    break
                self.confirm(direction, cands[0].candidate_id)
                added.append(cands[0].candidate_id)
        return added

    # -- views ----------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        tracks = self.chain_tracks()
        return {
            "session_id": self.session_id,
            "query_track_id": self.query_track_id,
            "chain": [t.to_public() for t in tracks],
            "visited_cameras": sorted({t.camera_id for t in tracks}),
            "log": self.log,
        }

    @staticmethod
    def _check_direction(direction: str) -> None:
        if direction not in DIRECTIONS:
            raise ValueError("direction must be 'forward' or 'backward'")
