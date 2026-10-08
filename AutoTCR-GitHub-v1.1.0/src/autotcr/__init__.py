"""Public Python API for AutoTCR inference."""

from .aggregation import aggregate_probabilities
from .backend import LegacyTorchBackend, PredictionBackend, TorchBackend
from .config import InferenceSettings
from .hub import download_model
from .predictor import AutoTCRPredictor, RepertoirePrediction

__all__ = [
    "AutoTCRPredictor",
    "InferenceSettings",
    "LegacyTorchBackend",
    "TorchBackend",
    "PredictionBackend",
    "RepertoirePrediction",
    "aggregate_probabilities",
    "download_model",
]

__version__ = "1.1.0"

