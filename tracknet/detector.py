"""Vehicle detectors.

* YoloDetector   - YOLOv8n via the `ultralytics` package (the detector used for real traffic footage).
* MotionDetector - background-subtraction fallback. It needs no model download and is used for the
                   synthetic demo videos. Its class labels are a crude size heuristic (demo only).
"""
from __future__ import annotations

import cv2
import numpy as np

from .tracker import Detection

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("YoloDetector needs ultralytics: pip install -r requirements-yolo.txt") from exc
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=list(COCO_VEHICLES),
                                    device=self.device, verbose=False)[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy()):
            detections.append(Detection(bbox=xyxy, confidence=float(conf), cls=COCO_VEHICLES[int(cls)]))
        return detections


class MotionDetector:
    """Detects moving blobs with MOG2 background subtraction."""

    def __init__(self, min_area: int = 500, history: int = 120, var_threshold: float = 40.0, warmup: int = 5):
        self.min_area = min_area
        self.warmup = warmup          # MOG2 marks the whole first frames as foreground while it learns the scene
        self._seen = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)
        self.open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    @staticmethod
    def _label(w: float, h: float) -> str:
        area = w * h
        if area < 2500:
            return "motorcycle"
        if area < 6000:
            return "car"
        if area < 9500:
            return "truck"
        return "bus"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self.bg.apply(frame)
        self._seen += 1
        if self._seen <= self.warmup:
            return []
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area:
                continue
            out.append(Detection(bbox=[x, y, x + w, y + h], confidence=0.9, cls=self._label(w, h)))
        return out
