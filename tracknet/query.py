"""Descriptive vehicle search ("red car", "white truck on c021").

This is a rule-based parser over the stored metadata (colour, class, camera, direction). It is not a
trained text-to-vehicle model; swapping in a vision-language model only needs a new `search` function.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .db import Store, TrackRecord

COLORS = {
    "red": "red", "maroon": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "blue": "blue", "navy": "blue", "purple": "purple", "violet": "purple",
    "white": "white", "silver": "silver", "grey": "silver", "gray": "silver", "black": "black",
}
CLASSES = {
    "car": "car", "sedan": "car", "suv": "car", "hatchback": "car", "jeep": "car", "van": "car",
    "truck": "truck", "lorry": "truck", "pickup": "truck",
    "bus": "bus", "coach": "bus",
    "bike": "motorcycle", "motorbike": "motorcycle", "motorcycle": "motorcycle", "scooter": "motorcycle",
}
DIRECTIONS = {
    "left": "left", "right": "right", "towards": "approaching", "approaching": "approaching",
    "away": "receding", "receding": "receding",
}


@dataclass
class ParsedQuery:
    raw: str
    color: str | None = None
    vehicle_class: str | None = None
    camera: str | None = None
    direction: str | None = None

    def attributes(self) -> dict[str, str]:
        return {k: v for k, v in (("color", self.color), ("vehicle_class", self.vehicle_class),
                                  ("direction", self.direction)) if v}

    def to_dict(self) -> dict:
        return asdict(self)


def parse_query(text: str, known_cameras: list[str] | None = None) -> ParsedQuery:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    q = ParsedQuery(raw=text or "")
    for w in words:
        if w in COLORS and not q.color:
            q.color = COLORS[w]
        elif w in CLASSES and not q.vehicle_class:
            q.vehicle_class = CLASSES[w]
        elif w in DIRECTIONS and not q.direction:
            q.direction = DIRECTIONS[w]
        elif re.fullmatch(r"c\d{3}", w) and (known_cameras is None or w in known_cameras):
            q.camera = w
    return q


def search(store: Store, query: ParsedQuery, camera: str | None = None, limit: int = 20) -> list[tuple[TrackRecord, float]]:
    """Rank tracks by the fraction of requested attributes they match (best first, longer tracks win ties)."""
    wanted = query.attributes()
    if not wanted:
        return []
    cam = camera or query.camera
    results: list[tuple[TrackRecord, float]] = []
    for t in store.list_tracks(camera_id=cam):
        hits = sum(1 for k, v in wanted.items() if getattr(t, k) == v)
        if hits:
            results.append((t, hits / len(wanted)))
    results.sort(key=lambda r: (-r[1], -r[0].duration, r[0].track_id))
    return results[:limit]
