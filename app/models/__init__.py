"""Detection + classification models package."""
from .detector import FaceDetector, FaceBox
from .classifier import EmotionClassifier
from .pipeline import Pipeline, DetectionResult

__all__ = [
    "FaceDetector",
    "FaceBox",
    "EmotionClassifier",
    "Pipeline",
    "DetectionResult",
]
