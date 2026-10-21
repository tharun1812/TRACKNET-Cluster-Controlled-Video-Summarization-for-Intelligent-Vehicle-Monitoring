"""Appearance features: colour histogram embedding, colour name and travel direction."""
from __future__ import annotations

import cv2
import numpy as np

HIST_BINS = (8, 4, 4)  # H, S, V
EMBEDDING_DIM = int(np.prod(HIST_BINS))


def color_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    """Hellinger-normalised HSV histogram (unit L2 norm, so cosine similarity = Bhattacharyya coefficient)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    small = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).flatten()
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.sqrt(hist / total).astype(np.float32)


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def dominant_color_name(crop_bgr: np.ndarray) -> str:
    """Rule-based colour name for the central region of a vehicle crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    y0, y1 = int(h * 0.25), max(int(h * 0.75), int(h * 0.25) + 1)
    x0, x1 = int(w * 0.25), max(int(w * 0.75), int(w * 0.25) + 1)
    centre = crop_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    chromatic = (sat >= 70) & (val >= 60)
    if chromatic.mean() >= 0.35:
        mode_hue = int(np.bincount(hue[chromatic], minlength=180).argmax())
        if mode_hue < 8 or mode_hue >= 170:
            return "red"
        if mode_hue < 20:
            return "orange"
        if mode_hue < 35:
            return "yellow"
        if mode_hue < 85:
            return "green"
        if mode_hue < 135:
            return "blue"
        return "purple"

    v = float(np.median(val))
    if v < 70:
        return "black"
    return "white" if v > 190 else "silver"


def direction_from_trajectory(trajectory: list[list[float]], min_move: float = 25.0) -> str:
    """Coarse travel direction in image space from [[t, cx, cy], ...]."""
    if len(trajectory) < 2:
        return "static"
    dx = trajectory[-1][1] - trajectory[0][1]
    dy = trajectory[-1][2] - trajectory[0][2]
    if max(abs(dx), abs(dy)) < min_move:
        return "static"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "approaching" if dy > 0 else "receding"


OPPOSITE = {"left": "right", "right": "left", "approaching": "receding", "receding": "approaching"}


def direction_compatibility(a: str, b: str) -> float:
    """1.0 same direction, 0.5 if either is static, 0.0 if opposite, 0.2 otherwise."""
    if a == b:
        return 1.0
    if "static" in (a, b):
        return 0.5
    if OPPOSITE.get(a) == b:
        return 0.0
    return 0.2
