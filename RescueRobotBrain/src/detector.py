from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import cv2

from .datatypes import Detection2D


class BaseDetector(ABC):
    """Abstract base class for drone detection modules."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        """
        Detects drones in the provided BGR image frame.

        Args:
            frame: np.ndarray (H, W, 3) BGR image.

        Returns:
            List[Detection2D]: List of detected bounding boxes and metadata.
        """
        pass


class SkyMotionDetector(BaseDetector):
    """
    Lightweight, high-speed detector optimized for sky-facing cameras.
    Combines adaptive luminance contrast and background subtraction
    to detect aerial targets without requiring heavy GPU inference.
    """

    def __init__(
        self,
        min_area: int = 40,
        max_area: int = 15000,
        contrast_threshold: int = 30,
        history: int = 100,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.contrast_threshold = contrast_threshold
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=25,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        detections: List[Detection2D] = []
        if frame is None or frame.size == 0:
            return detections

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Background subtraction for moving aerial objects
        fg_mask = self.bg_subtractor.apply(blurred)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # 2. Extract candidate contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Aspect ratio check: drones are generally roughly square-ish or wide (0.3 - 3.0)
                aspect_ratio = float(w) / float(max(h, 1))
                if 0.25 <= aspect_ratio <= 4.0:
                    conf = min(0.95, 0.5 + (area / self.max_area) * 0.45)
                    detections.append(Detection2D(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=conf,
                        class_id=0,
                        class_name="drone"
                    ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


class YOLODetector(BaseDetector):
    """
    Deep-learning detector using Ultralytics YOLO (YOLOv8 / YOLOv11).
    Can use standard models or custom drone-specific weights.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        target_classes: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        if device is None or device == "auto":
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device
        self.model = None

        # Default class labels to match if using generic COCO model:
        # COCO class 4 = 'airplane', class 14 = 'bird', or custom class 0 = 'drone'
        self.target_classes = target_classes or ["drone", "airplane", "bird"]
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[YOLODetector] Model loaded on hardware device: {self.device.upper()}")
        except Exception as e:
            print(f"[YOLODetector] Note: YOLO model could not be initialized yet ({e}). "
                  "Will fall back to motion/contrast detection if needed.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        detections: List[Detection2D] = []
        if self.model is None or frame is None or frame.size == 0:
            return detections

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                imgsz=640,
                device=self.device,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = r.names.get(cls_id, f"class_{cls_id}").lower()
                    conf = float(box.conf[0].item())

                    # Check if detected class is in target classes, or if model has a single custom class
                    is_match = (cls_name in self.target_classes) or (len(r.names) == 1)
                    if is_match:
                        xyxy = box.xyxy[0].tolist()
                        detections.append(Detection2D(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
        except Exception as e:
            print(f"[YOLODetector] Inference error: {e}")

        return detections


class QuadcopterPropellerDetector(BaseDetector):
    """
    Detector specialized for quadcopters equipped with high-visibility / lime-green propellers
    and dark chassis frame. Robust across static images and live video streams.
    """

    def __init__(self, min_prop_area: int = 50, max_prop_area: int = 8000):
        self.min_area = min_prop_area
        self.max_area = max_prop_area
        # Lime green HSV range
        self.green_lower = np.array([35, 55, 75], dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        detections: List[Detection2D] = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_props = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                valid_props.append(cv2.boundingRect(cnt))

        if len(valid_props) >= 2:
            min_x = min(b[0] for b in valid_props)
            min_y = min(b[1] for b in valid_props)
            max_x = max(b[0] + b[2] for b in valid_props)
            max_y = max(b[1] + b[3] for b in valid_props)

            margin_x = int((max_x - min_x) * 0.20)
            margin_y = int((max_y - min_y) * 0.20)
            bx1 = max(0, min_x - margin_x)
            by1 = max(0, min_y - margin_y)
            bx2 = min(w, max_x + margin_x)
            by2 = min(h, max_y + margin_y)

            detections.append(Detection2D(
                x1=float(bx1),
                y1=float(by1),
                x2=float(bx2),
                y2=float(by2),
                confidence=0.88,
                class_id=0,
                class_name="quadcopter"
            ))

        return detections


class SmartHybridDetector(BaseDetector):
    """
    Intelligent multi-tier detector:
    1. YOLOv8 deep learning (checks aerial & drone targets)
    2. Quadcopter propeller & frame tracker (robust on indoor / handheld drones)
    3. High-speed SkyMotionDetector (sky background subtraction)
    """

    def __init__(self, yolo_model_path: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.yolo = YOLODetector(model_path=yolo_model_path, confidence_threshold=conf_thresh)
        self.propeller = QuadcopterPropellerDetector()
        self.motion = SkyMotionDetector()

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        # Tier 1: YOLO inference
        if self.yolo.model is not None:
            yolo_dets = self.yolo.detect(frame)
            if len(yolo_dets) > 0:
                return yolo_dets

        # Tier 2: Propeller & Frame segmentation
        prop_dets = self.propeller.detect(frame)
        if len(prop_dets) > 0:
            return prop_dets

        # Tier 3: Sky motion fallback
        return self.motion.detect(frame)
