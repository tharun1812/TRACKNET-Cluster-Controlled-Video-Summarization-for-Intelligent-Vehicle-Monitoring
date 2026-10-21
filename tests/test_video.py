"""Tests that run the real video pipeline on a tiny rendered clip."""
import cv2
import pytest

from tracknet.db import Store
from tracknet.demo import Appearance, SyntheticVehicle, render_camera_video
from tracknet.detector import MotionDetector
from tracknet.pipeline import process_video
from tracknet.summarizer import Interval, intervals_for_tracks, merge_intervals, render_summary


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("clip")
    vehicles = [
        SyntheticVehicle(0, "car", "red", (0, 255, 255), "right", [Appearance("c017", 3.0, 9.0, 0)]),
        SyntheticVehicle(1, "truck", "blue", (255, 0, 255), "right", [Appearance("c017", 12.0, 18.0, 2)]),
    ]
    video = d / "c017.mp4"
    render_camera_video(video, "c017", vehicles, duration=22.0, fps=10, seed=1)
    return d, video


@pytest.fixture(scope="module")
def processed(clip):
    d, video = clip
    store = Store(":memory:")
    result = process_video(video, "c017", store, MotionDetector(), images_dir=d / "crops")
    return d, video, store, result


def test_pipeline_finds_both_vehicles_with_correct_attributes(processed):
    _d, _video, store, result = processed
    tracks = store.list_tracks("c017")
    assert len(tracks) == 2 and result["frames_processed"] == 220
    red, blue = tracks
    assert (red.color, red.vehicle_class, red.direction) == ("red", "car", "right")
    assert (blue.color, blue.vehicle_class) == ("blue", "truck")
    assert abs(red.start_ts - 3.0) < 1.0 and abs(red.end_ts - 9.0) < 1.0
    assert red.image_path and cv2.imread(red.image_path) is not None
    assert len(red.embedding) == 128


def test_reprocessing_replaces_instead_of_duplicating(processed):
    d, video, store, _ = processed
    process_video(video, "c017", store, MotionDetector(), images_dir=d / "crops")
    assert store.count_tracks("c017") == 2


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_video(tmp_path / "nope.mp4", "c017", Store(":memory:"), MotionDetector())


def test_merge_intervals_joins_overlaps_and_keeps_order():
    a = Interval("c017", "v.mp4", 0, 5, "T1", 0)
    b = Interval("c017", "v.mp4", 5.2, 8, "T2", 5)
    c = Interval("c017", "v.mp4", 20, 25, "T3", 20)
    out = merge_intervals([c, b, a], gap=0.5)
    assert [(i.start, i.end) for i in out] == [(0, 8), (20, 25)]
    assert out[0].label == "T1, T2"


def test_vehicle_summary_is_shorter_than_source(processed, tmp_path):
    _d, video, store, _ = processed
    first = store.list_tracks("c017")[:1]
    stats = render_summary(intervals_for_tracks(first), tmp_path / "s.mp4")
    out = cv2.VideoCapture(stats["output_path"])
    assert out.isOpened() and out.get(cv2.CAP_PROP_FRAME_COUNT) > 20
    out.release()
    assert stats["segments"] == 1
    assert 0 < stats["summary_seconds"] < stats["source_seconds"]
    assert stats["reduction"] > 0.4


def test_summary_without_footage_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError):
        render_summary([], tmp_path / "x.mp4")
