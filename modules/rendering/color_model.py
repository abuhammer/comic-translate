from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


_LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class BubbleColorPrediction:
    """Container describing the desired text/outline colours."""

    use_light_text: bool
    confidence: float


class BubbleColorClassifier:
    """Lightweight logistic classifier trained on bubble background swatches.

    The classifier is intentionally simple: a hand-curated palette of labelled
    manhwa bubble backgrounds is transformed into statistics (mean/variance
    across RGB channels, luminance coverage, etc.) and a logistic regression is
    fitted using gradient descent. At runtime each sampled background region is
    reduced to the same feature vector before being scored by the learned model.
    """

    _instance: Optional["BubbleColorClassifier"] = None

    def __init__(self) -> None:
        dataset = self._load_dataset()
        features = np.stack([self._colour_to_features(sample.rgb) for sample in dataset])
        labels = np.array([1.0 if sample.label == "light_text" else 0.0 for sample in dataset])

        (
            self._feature_mean,
            self._feature_std,
            self._weights,
            self._bias,
        ) = self._train_logistic(features, labels)

    @dataclass
    class _Sample:
        rgb: np.ndarray
        label: str

    @classmethod
    def get(cls) -> "BubbleColorClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_from_region(self, region: np.ndarray) -> Optional[BubbleColorPrediction]:
        if region.size == 0:
            return None

        if region.ndim == 2:
            region = np.repeat(region[:, :, None], 3, axis=2)
        elif region.shape[2] > 3:
            region = region[:, :, :3]

        pixels = region.reshape(-1, 3).astype(np.float32) / 255.0
        if pixels.size == 0:
            return None

        features = self._region_to_features(pixels)
        score = self._score(features)
        probability = 1.0 / (1.0 + math.exp(-score))
        return BubbleColorPrediction(use_light_text=probability >= 0.5, confidence=probability)

    def _score(self, features: np.ndarray) -> float:
        normalised = (features - self._feature_mean) / self._feature_std
        return float(normalised.dot(self._weights) + self._bias)

    def _region_to_features(self, pixels: np.ndarray) -> np.ndarray:
        """Derive statistics describing a sampled bubble region."""

        if pixels.shape[0] > 4096:
            # Subsample large regions for speed while keeping determinism.
            step = math.ceil(pixels.shape[0] / 4096)
            pixels = pixels[::step]

        mean_rgb = pixels.mean(axis=0)
        std_rgb = pixels.std(axis=0)

        luminance = pixels.dot(_LUMINANCE_WEIGHTS)
        mean_luminance = float(luminance.mean())
        std_luminance = float(luminance.std())
        dark_fraction = float((luminance < 0.35).mean())
        light_fraction = float((luminance > 0.75).mean())
        luminance_range = float(luminance.max() - luminance.min())

        contrast_to_white = float(((np.array([1.0, 1.0, 1.0]) - pixels) ** 2).mean())
        contrast_to_black = float((pixels ** 2).mean())

        feature_vector = np.array(
            [
                mean_rgb[0],
                mean_rgb[1],
                mean_rgb[2],
                std_rgb[0],
                std_rgb[1],
                std_rgb[2],
                mean_luminance,
                std_luminance,
                dark_fraction,
                light_fraction,
                luminance_range,
                contrast_to_white,
                contrast_to_black,
            ],
            dtype=np.float32,
        )
        return feature_vector

    def _colour_to_features(self, rgb: np.ndarray) -> np.ndarray:
        pixels = np.tile(rgb[None, :], (64, 1))
        jitter = (np.random.default_rng(42).random(pixels.shape) - 0.5) * 0.02
        pixels = np.clip(pixels + jitter, 0.0, 1.0)
        return self._region_to_features(pixels)

    def _train_logistic(
        self, features: np.ndarray, labels: np.ndarray, epochs: int = 2000, lr: float = 0.15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std[feature_std == 0] = 1.0

        normalised = (features - feature_mean) / feature_std
        weights = np.zeros(normalised.shape[1], dtype=np.float32)
        bias = 0.0

        for _ in range(epochs):
            logits = normalised.dot(weights) + bias
            preds = 1.0 / (1.0 + np.exp(-logits))
            error = preds - labels
            grad_w = normalised.T.dot(error) / normalised.shape[0]
            grad_b = float(error.mean())
            weights -= lr * grad_w
            bias -= lr * grad_b

        return feature_mean, feature_std, weights, bias

    def _load_dataset(self) -> List["BubbleColorClassifier._Sample"]:
        dataset_path = self._dataset_path()
        if not dataset_path.exists():
            return self._default_dataset()

        with dataset_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        samples: List[BubbleColorClassifier._Sample] = []
        for entry in raw:
            rgb_values = np.array(entry.get("rgb", [0, 0, 0]), dtype=np.float32) / 255.0
            label = entry.get("label", "light_text")
            samples.append(self._Sample(rgb=rgb_values, label=label))
        return samples

    def _dataset_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "resources" / "data" / "bubble_color_dataset.json"

    def _default_dataset(self) -> List["BubbleColorClassifier._Sample"]:
        presets = [
            ([20, 30, 60], "light_text"),
            ([235, 240, 245], "dark_text"),
            ([65, 75, 85], "light_text"),
            ([250, 250, 250], "dark_text"),
        ]
        return [
            self._Sample(rgb=np.array(rgb, dtype=np.float32) / 255.0, label=label)
            for rgb, label in presets
        ]


def predict_bubble_text_color(region: np.ndarray) -> Optional[BubbleColorPrediction]:
    """Convenience wrapper that returns a ML-based colour prediction."""

    classifier = BubbleColorClassifier.get()
    return classifier.predict_from_region(region)
