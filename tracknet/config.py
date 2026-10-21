"""Camera clusters, the camera graph and matching weights.

The camera network mirrors the project report: cameras c017, c018, c021, c025 and c026 with
directional transitions and travel-time windows. Edit CAMERAS / EDGES to describe another site.
"""
from __future__ import annotations

# camera_id -> metadata. `cluster` groups nearby cameras; `time_offset` aligns a camera's video clock
# to the shared clock (seconds) when prerecorded videos do not start at the same moment.
CAMERAS: dict[str, dict] = {
    "c017": {"cluster": "A", "name": "Junction entry", "view": "front", "time_offset": 0.0},
    "c018": {"cluster": "A", "name": "Junction side road", "view": "side", "time_offset": 0.0},
    "c021": {"cluster": "A", "name": "Main road", "view": "rear", "time_offset": 0.0},
    "c025": {"cluster": "B", "name": "Flyover ramp", "view": "side", "time_offset": 0.0},
    "c026": {"cluster": "B", "name": "Exit road", "view": "front", "time_offset": 0.0},
}

# Directed edges (source -> destination) in the direction of travel, with the (min, max) number of
# seconds a vehicle needs to get from the source camera's view to the destination camera's view.
EDGES: dict[tuple[str, str], tuple[float, float]] = {
    ("c017", "c018"): (3.0, 20.0),
    ("c018", "c021"): (3.0, 20.0),
    ("c021", "c025"): (3.0, 20.0),
    ("c025", "c026"): (3.0, 20.0),
    ("c017", "c021"): (8.0, 30.0),
    ("c021", "c026"): (8.0, 30.0),
}

# Weights for combining the similarity cues into one match score (they sum to 1).
MATCH_WEIGHTS: dict[str, float] = {
    "appearance": 0.40,
    "color": 0.20,
    "temporal": 0.20,
    "direction": 0.10,
    "graph": 0.10,
}

CLASS_MISMATCH_PENALTY = 0.3      # score multiplier when vehicle classes differ
TEMPORAL_SLACK = 0.5              # fraction of the window width tolerated outside the window
MAX_HOPS = 2                      # how many camera hops the BFS expands from the current track
