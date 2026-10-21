import numpy as np
import pytest

from tracknet.db import Store, TrackRecord
from tracknet.features import color_histogram
from tracknet.demo import BODY_COLORS


def patch(color: str, size: int = 40) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = BODY_COLORS[color]
    return img


def make_track(track_id, camera, start, end, color="red", cls="car", direction="right", video=None):
    emb = color_histogram(patch(color))
    return TrackRecord(
        track_id=track_id, camera_id=camera, start_ts=start, end_ts=end, video_start=start, video_end=end,
        vehicle_class=cls, color=color, direction=direction, embedding=[float(x) for x in emb],
        trajectory=[[start, 10.0, 100.0], [end, 600.0, 100.0]], image_path=None, video_path=video)


@pytest.fixture()
def store():
    s = Store(":memory:")
    yield s
    s.close()
