"""End-to-end pipeline: detect → crop → classify → annotate."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .. import config
from .classifier import EmotionClassifier
from .detector import FaceBox, FaceDetector


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    emotion: str
    score: float
    all_scores: dict[str, float]


class Pipeline:
    """Combines face detection and emotion classification.

    Stateless w.r.t. callers — both sub-models are long-lived singletons
    owned by the application lifespan.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        classifier: EmotionClassifier | None = None,
        skip_every_n: int = 1,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.classifier = classifier or EmotionClassifier()
        # For video: optionally classify every Nth frame to keep FPS up.
        # 1 = classify every frame; 3 = classify ~10 fps if video is 30fps.
        self.skip_every_n = max(1, int(skip_every_n))
        self._frame_idx = 0
        self._last_results: list[DetectionResult] = []

    def ensure_loaded(self) -> None:
        self.detector.load()
        self.classifier.load()

    # -------------------------------------------------------------- core api
    def detect_and_draw(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, list[DetectionResult]]:
        """Run detection (and classification when due), then draw on the frame.

        Returns the annotated frame and a list of detection dicts.
        """
        if not self.detector.is_loaded or not self.classifier.is_loaded:
            self.ensure_loaded()

        boxes = self.detector.detect(frame_bgr)
        results: list[DetectionResult] = []
        do_classify = (
            self._frame_idx % self.skip_every_n == 0 or not self._last_results
        )

        for box in boxes:
            crop = self._safe_crop(frame_bgr, box)
            if crop is None:
                continue
            try:
                label, score, all_scores = self.classifier.classify(crop)
            except Exception:
                # If classifier fails for one face, skip it but keep others
                continue
            results.append(
                DetectionResult(
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    emotion=label,
                    score=score,
                    all_scores=all_scores,
                )
            )

        if not do_classify and not results:
            # Reuse last results to avoid flicker when skipping classification
            results = self._last_results

        self._last_results = results
        self._frame_idx += 1

        annotated = self._draw(frame_bgr, results)
        return annotated, results

    def detect_only_json(
        self, frame_bgr: np.ndarray
    ) -> tuple[list[DetectionResult], float]:
        """For the JSON API: detect + classify and return timing (ms)."""
        t0 = time.perf_counter()
        if not self.detector.is_loaded or not self.classifier.is_loaded:
            self.ensure_loaded()
        boxes = self.detector.detect(frame_bgr)
        results: list[DetectionResult] = []
        for box in boxes:
            crop = self._safe_crop(frame_bgr, box)
            if crop is None:
                continue
            label, score, all_scores = self.classifier.classify(crop)
            results.append(
                DetectionResult(
                    bbox=(box.x1, box.y1, box.x2, box.y2),
                    emotion=label,
                    score=score,
                    all_scores=all_scores,
                )
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, elapsed_ms

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _safe_crop(frame: np.ndarray, box: FaceBox) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1 = max(0, min(w, box.x1))
        y1 = max(0, min(h, box.y1))
        x2 = max(0, min(w, box.x2))
        y2 = max(0, min(h, box.y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _draw(frame: np.ndarray, results: list[DetectionResult]) -> np.ndarray:
        out = frame
        for r in results:
            color = config.EMOTION_COLORS_BGR.get(r.emotion, (0, 255, 0))
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{r.emotion} {r.score * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            ty1 = max(0, y1 - th - baseline - 6)
            ty2 = ty1 + th + baseline + 6
            cv2.rectangle(out, (x1, ty1), (x1 + tw + 10, ty2), color, -1)
            cv2.putText(
                out,
                label,
                (x1 + 5, ty2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out
