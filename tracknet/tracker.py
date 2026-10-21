"""DeepSORT-style multi-object tracker.

Constant-velocity Kalman filter on (cx, cy, aspect, height) plus appearance embeddings.
Association: confirmed tracks are matched with a combined IoU + appearance cost (Hungarian algorithm);
leftover tracks are then matched by IoU only. Tracks are confirmed after `n_init` hits and dropped
after `max_age` missed frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    bbox: np.ndarray                 # [x1, y1, x2, y2]
    confidence: float = 1.0
    cls: str = "car"
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBoxFilter:
    """8-D constant-velocity Kalman filter (x, y, a, h, vx, vy, va, vh)."""

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)
        self._pos_w, self._vel_w = 1.0 / 20, 1.0 / 160

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._pos_w * h, 2 * self._pos_w * h, 1e-2, 2 * self._pos_w * h,
               10 * self._vel_w * h, 10 * self._vel_w * h, 1e-5, 10 * self._vel_w * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._pos_w * h, self._pos_w * h, 1e-2, self._pos_w * h,
               self._vel_w * h, self._vel_w * h, 1e-5, self._vel_w * h]
        q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([self._pos_w * h, self._pos_w * h, 1e-1, self._pos_w * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ (z - self.H @ mean)
        cov = (np.eye(8) - k @ self.H) @ cov
        return mean, cov


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    first_frame: int
    n_init: int
    hits: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    last_detection: Detection | None = None
    last_frame: int = 0
    history: list = field(default_factory=list)   # (frame, bbox, confidence, cls)
    emb_sum: np.ndarray | None = None
    emb_count: int = 0
    gallery: np.ndarray | None = None             # smoothed appearance used for association

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if self.emb_sum is None or self.emb_count == 0:
            return None
        v = self.emb_sum / self.emb_count
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def add_embedding(self, e: np.ndarray | None) -> None:
        if e is None:
            return
        e = np.asarray(e, dtype=np.float64)
        self.emb_sum = e.copy() if self.emb_sum is None else self.emb_sum + e
        self.emb_count += 1
        self.gallery = e.copy() if self.gallery is None else 0.9 * self.gallery + 0.1 * e


class DeepSortTracker:
    def __init__(self, max_age: int = 15, n_init: int = 3, iou_threshold: float = 0.2,
                 max_cost: float = 0.75, appearance_weight: float = 0.4) -> None:
        self.max_age, self.n_init = max_age, n_init
        self.iou_threshold, self.max_cost, self.appearance_weight = iou_threshold, max_cost, appearance_weight
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    # -- helpers --------------------------------------------------------------------------------
    def _cost(self, tracks: list[Track], dets: list[Detection], use_appearance: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix(np.array([t.bbox for t in tracks]), np.array([d.bbox for d in dets]))
        cost = 1.0 - iou
        if use_appearance:
            app = np.ones_like(cost)
            for i, t in enumerate(tracks):
                if t.gallery is None:
                    continue
                for j, d in enumerate(dets):
                    if d.embedding is not None:
                        denom = np.linalg.norm(t.gallery) * np.linalg.norm(d.embedding) + 1e-9
                        app[i, j] = 1.0 - float(np.clip(np.dot(t.gallery, d.embedding) / denom, 0, 1))
            cost = (1 - self.appearance_weight) * cost + self.appearance_weight * app
        cost[iou < 1e-6] = 1e5   # never match boxes that do not overlap at all (gate)
        return cost

    @staticmethod
    def _assign(cost: np.ndarray, limit: float):
        n_t, n_d = cost.shape
        if n_t == 0 or n_d == 0:
            return [], list(range(n_t)), list(range(n_d))
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= limit]
        matched_r, matched_c = {r for r, _ in matches}, {c for _, c in matches}
        return (matches,
                [i for i in range(n_t) if i not in matched_r],
                [j for j in range(n_d) if j not in matched_c])

    def _apply_update(self, track: Track, det: Detection, frame_idx: int) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(det.bbox))
        track.hits += 1
        track.time_since_update = 0
        track.last_detection, track.last_frame = det, frame_idx
        track.history.append((frame_idx, det.bbox.copy(), det.confidence, det.cls))
        track.add_embedding(det.embedding)
        if track.hits >= self.n_init:
            track.confirmed = True

    # -- public API -----------------------------------------------------------------------------
    def update(self, frame_idx: int, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns the confirmed tracks that were matched in this frame."""
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.time_since_update += 1

        c_tracks = [t for t in self.tracks if t.confirmed]
        u_tracks = [t for t in self.tracks if not t.confirmed]

        # 1) confirmed tracks: IoU + appearance
        matches, un_c, un_d = self._assign(self._cost(c_tracks, detections, True), self.max_cost)
        for ti, di in matches:
            self._apply_update(c_tracks[ti], detections[di], frame_idx)

        # 2) unconfirmed tracks and just-lost confirmed tracks: IoU only
        leftover = u_tracks + [c_tracks[i] for i in un_c if c_tracks[i].time_since_update == 1]
        left_dets = [detections[j] for j in un_d]
        matches2, _, un_d2 = self._assign(self._cost(leftover, left_dets, False), 1.0 - self.iou_threshold)
        for ti, di in matches2:
            self._apply_update(leftover[ti], left_dets[di], frame_idx)

        # 3) new tracks from still-unmatched detections
        for j in un_d2:
            det = left_dets[j]
            mean, cov = self.kf.initiate(xyxy_to_xyah(det.bbox))
            t = Track(self._next_id, mean, cov, frame_idx, self.n_init, last_detection=det, last_frame=frame_idx)
            t.history.append((frame_idx, det.bbox.copy(), det.confidence, det.cls))
            t.add_embedding(det.embedding)
            if self.n_init <= 1:
                t.confirmed = True
            self._next_id += 1
            self.tracks.append(t)

        # 4) retire dead tracks
        alive = []
        for t in self.tracks:
            limit = self.max_age if t.confirmed else min(3, self.max_age)
            if t.time_since_update > limit:
                if t.confirmed:
                    self.finished.append(t)
            else:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def flush(self) -> list[Track]:
        """End of stream: finish all still-active confirmed tracks and return every finished track."""
        for t in self.tracks:
            if t.confirmed:
                self.finished.append(t)
        self.tracks = []
        return self.finished
