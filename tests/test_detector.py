"""YoloDetector is tested against a stand-in for `ultralytics` (the real package is optional and heavy)."""
import sys
import types

import numpy as np
import pytest

from tracknet.detector import MotionDetector


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.kwargs = None

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        boxes = types.SimpleNamespace(
            xyxy=_Arr([[10, 20, 110, 80], [200, 50, 330, 140]]),
            conf=_Arr([0.9, 0.5]),
            cls=_Arr([2, 7]),
        )
        return [types.SimpleNamespace(boxes=boxes)]


def test_yolo_detector_maps_coco_classes(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    from tracknet.detector import YoloDetector
    det = YoloDetector("yolov8n.pt", conf=0.4)
    out = det.detect(np.zeros((360, 640, 3), dtype=np.uint8))
    assert [d.cls for d in out] == ["car", "truck"]
    assert out[0].bbox.tolist() == [10, 20, 110, 80] and out[0].confidence == pytest.approx(0.9)
    assert det.model.kwargs["classes"] == [2, 3, 5, 7] and det.model.kwargs["conf"] == 0.4


def test_yolo_detector_gives_helpful_error_without_ultralytics(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    from tracknet.detector import YoloDetector
    with pytest.raises(ImportError, match="requirements-yolo"):
        YoloDetector()


def test_motion_detector_ignores_an_empty_scene_and_finds_a_moving_box():
    det = MotionDetector()
    bg = np.full((360, 640, 3), 100, dtype=np.uint8)
    for _ in range(8):   # the first frames are warm-up, the rest are a static scene
        assert det.detect(bg.copy()) == []
    frame = bg.copy()
    frame[100:150, 200:300] = (30, 30, 200)
    found = det.detect(frame)
    assert len(found) == 1
    x1, y1, x2, y2 = found[0].bbox
    assert abs(x1 - 200) <= 3 and abs(x2 - 300) <= 3 and abs(y1 - 100) <= 3 and abs(y2 - 150) <= 3
