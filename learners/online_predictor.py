"""
Online predictor using river for AI Trading System V3.
Implements online machine learning for trade prediction.
"""
from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from collections import deque as rolling_deque
from river import compose, linear_model, metrics, optim, preprocessing, tree

from core.logger import get_logger
from core.state import MarketContext, PredictionRecord, MarketRegime
from learners.experience_buffer import ExperienceBuffer

log = get_logger("online_predictor")


class OnlinePredictor:
    """
    Online machine learning predictor using river.

    Features:
    - Multiple model ensemble (logistic regression, Hoeffding tree)
    - Online learning from trade outcomes
    - Prediction confidence calibration
    - Model persistence
    - Performance tracking by regime
    """

    # Minimum samples before predictions are trusted
    MIN_SAMPLES_FOR_PREDICTION = 50

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        model_type: str = "ensemble",
        learning_rate: float = 0.01,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize online predictor.

        Args:
            experience_buffer: Buffer for trade experiences
            model_type: "logistic", "tree", or "ensemble"
            learning_rate: Learning rate for gradient-based models
            persist_path: Path for model persistence
        """
        self.experience_buffer = experience_buffer
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.persist_path = Path(persist_path) if persist_path else None

        # Initialize models
        self._init_models()

        # Prediction tracking
        self._predictions: deque[PredictionRecord] = deque(maxlen=1000)
        self._prediction_count: int = 0

        # Performance metrics
        self._accuracy = metrics.Accuracy()
        self._accuracy_by_regime: dict[MarketRegime, metrics.Accuracy] = {
            regime: metrics.Accuracy() for regime in MarketRegime
        }
        # Custom rolling window for accuracy (river.metrics.Rolling not available)
        self._rolling_window: rolling_deque[bool] = rolling_deque(maxlen=50)

        # Confidence calibration
        self._calibration_bins: dict[str, list[tuple[float, bool]]] = {}

        # Training stats
        self._samples_trained: int = 0

        log.info(f"[OnlinePredictor] Initialized with model_type={model_type}")

    def _init_models(self) -> None:
        """Initialize online learning models."""
        # Feature preprocessing pipeline
        self._preprocessor = compose.Pipeline(
            preprocessing.StandardScaler(),
        )

        if self.model_type == "logistic":
            self._model = compose.Pipeline(
                self._preprocessor,
                linear_model.LogisticRegression(
                    optimizer=optim.Adam(lr=self.learning_rate),
                )
            )
        elif self.model_type == "tree":
            self._model = compose.Pipeline(
                self._preprocessor,
                tree.HoeffdingTreeClassifier(
                    grace_period=100,
                    max_depth=10,
                )
            )
        elif self.model_type == "ensemble":
            # Ensemble of logistic regression and decision tree
            self._logistic = compose.Pipeline(
                preprocessing.StandardScaler(),
                linear_model.LogisticRegression(
                    optimizer=optim.Adam(lr=self.learning_rate),
                )
            )
            self._tree = compose.Pipeline(
                preprocessing.StandardScaler(),
                tree.HoeffdingTreeClassifier(
                    grace_period=50,
                    max_depth=8,
                )
            )
            self._model = None  # Will use both models
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, context: MarketContext, side: str) -> PredictionRecord:
        """
        Make prediction for a potential trade.

        Args:
            context: Current market context
            side: Proposed trade side ("long" or "short")

        Returns:
            PredictionRecord with prediction details
        """
        # Extract features
        features = self._extract_features(context, side)

        # Make prediction
        if self._samples_trained < self.MIN_SAMPLES_FOR_PREDICTION:
            # Not enough data - return neutral prediction
            predicted_direction = "neutral"
            confidence = 0.0
            predicted_magnitude = 0.0
        else:
            if self.model_type == "ensemble":
                # Ensemble prediction
                prob_logistic = self._get_probability(self._logistic, features)
                prob_tree = self._get_probability(self._tree, features)

                # Weighted average (logistic weighted higher early, tree weighted higher later)
                tree_weight = min(self._samples_trained / 500, 0.5)
                logistic_weight = 1.0 - tree_weight
                prob = prob_logistic * logistic_weight + prob_tree * tree_weight
            else:
                prob = self._get_probability(self._model, features)

            # Convert probability to direction
            if prob > 0.55:
                predicted_direction = "up"
            elif prob < 0.45:
                predicted_direction = "down"
            else:
                predicted_direction = "neutral"

            # Calculate confidence (distance from 0.5)
            confidence = abs(prob - 0.5) * 2
            confidence = self._calibrate_confidence(confidence, context.regime.regime)

            # Estimate magnitude based on volatility
            vol_magnitude = {"low": 0.5, "medium": 1.0, "high": 2.0, "extreme": 3.0}
            predicted_magnitude = vol_magnitude.get(context.regime.volatility_level, 1.0) * confidence

        # Create prediction record
        prediction = PredictionRecord(
            prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
            symbol=context.symbol,
            predicted_direction=predicted_direction,
            predicted_magnitude=predicted_magnitude,
            confidence=confidence,
            features=features,
            prediction_time=datetime.now(timezone.utc),
        )

        self._predictions.append(prediction)
        self._prediction_count += 1

        log.debug(
            f"[OnlinePredictor] Prediction for {context.symbol}: "
            f"{predicted_direction} (conf: {confidence:.2f})"
        )

        return prediction

    def _get_probability(self, model: Any, features: dict[str, float]) -> float:
        """Get probability of positive outcome from model."""
        try:
            proba = model.predict_proba_one(features)
            if proba and True in proba:
                return proba[True]
            return 0.5
        except Exception:
            return 0.5

    def _extract_features(self, context: MarketContext, side: str) -> dict[str, float]:
        """
        Extract features from market context.

        Args:
            context: Market context
            side: Trade side

        Returns:
            Feature dictionary
        """
        # Get indicator values
        ind_1h = context.price_feed.indicators.get("1h")
        rsi = ind_1h.rsi_14 if ind_1h and ind_1h.rsi_14 else 50.0
        macd_hist = ind_1h.macd_histogram if ind_1h and ind_1h.macd_histogram else 0.0

        # Volatility encoding
        vol_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "extreme": 1.0}

        features = {
            # Market state
            "sentiment": context.sentiment.sentiment_score,
            "rsi_normalized": (rsi - 50) / 50,
            "macd_histogram": np.clip(macd_hist / 0.01, -1, 1),
            "volume_relative": np.clip((context.volume.relative_volume - 1) / 2, -1, 1),
            "funding_rate": np.clip(context.funding.current_rate / 0.001, -1, 1),
            "volatility": vol_map.get(context.regime.volatility_level, 0.5),

            # Derived signals
            "momentum_score": context.momentum_score,
            "trend_aligned": 1.0 if context.trend_aligned else 0.0,
            "regime_confidence": context.regime.confidence,
            "trend_strength": context.regime.trend_strength,

            # Trade side
            "side_long": 1.0 if side == "long" else 0.0,

            # Fear & Greed
            "fear_greed": (context.sentiment.fear_greed_index - 50) / 50,
            "long_short_ratio": np.clip((context.sentiment.long_short_ratio - 1) * 2, -1, 1),
        }

        # Add regime one-hot encoding
        for regime in MarketRegime:
            features[f"regime_{regime.value}"] = 1.0 if context.regime.regime == regime else 0.0

        return features

    def learn_from_outcome(
        self,
        prediction: PredictionRecord,
        actual_pnl_percent: float,
    ) -> None:
        """
        Learn from trade outcome.

        Args:
            prediction: Original prediction
            actual_pnl_percent: Actual PnL percentage
        """
        # Determine actual direction
        if actual_pnl_percent > 0.5:
            actual_direction = "up"
            outcome = True
        elif actual_pnl_percent < -0.5:
            actual_direction = "down"
            outcome = False
        else:
            actual_direction = "neutral"
            outcome = None  # Don't learn from neutral outcomes

        # Update prediction record
        prediction.actual_direction = actual_direction
        prediction.actual_magnitude = abs(actual_pnl_percent)
        prediction.was_correct = (
            (prediction.predicted_direction == "up" and actual_direction == "up") or
            (prediction.predicted_direction == "down" and actual_direction == "down")
        )
        prediction.resolution_time = datetime.now(timezone.utc)

        if outcome is None:
            return  # Skip neutral outcomes

        # Learn from outcome
        features = prediction.features
        y = outcome

        if self.model_type == "ensemble":
            self._logistic.learn_one(features, y)
            self._tree.learn_one(features, y)
        else:
            self._model.learn_one(features, y)

        self._samples_trained += 1

        # Update metrics
        predicted = prediction.predicted_direction == "up"
        self._accuracy.update(outcome, predicted)
        # Update rolling window
        was_correct = (outcome == predicted)
        self._rolling_window.append(was_correct)

        # Update regime-specific accuracy
        regime_str = features.get("regime_strong_uptrend")  # Check which regime feature is set
        for regime in MarketRegime:
            if features.get(f"regime_{regime.value}", 0) == 1.0:
                self._accuracy_by_regime[regime].update(outcome, predicted)
                break

        # Update calibration
        self._update_calibration(prediction.confidence, prediction.was_correct or False)

        log.debug(
            f"[OnlinePredictor] Learned from outcome: "
            f"predicted={prediction.predicted_direction}, actual={actual_direction}, "
            f"correct={prediction.was_correct}"
        )

    def learn_from_buffer(self, n_samples: int = 100) -> int:
        """
        Learn from experiences in buffer.

        Args:
            n_samples: Number of samples to learn from

        Returns:
            Number of samples learned
        """
        features_list, rewards = self.experience_buffer.get_feature_matrix(n_samples, method="uniform_regime")

        if not features_list:
            return 0

        learned = 0
        for features, reward in zip(features_list, rewards):
            # Convert reward to binary outcome
            outcome = reward > 0

            if self.model_type == "ensemble":
                self._logistic.learn_one(features, outcome)
                self._tree.learn_one(features, outcome)
            else:
                self._model.learn_one(features, outcome)

            self._samples_trained += 1
            learned += 1

        log.info(f"[OnlinePredictor] Learned from {learned} buffered experiences")
        return learned

    def _calibrate_confidence(self, raw_confidence: float, regime: MarketRegime) -> float:
        """
        Calibrate confidence based on historical accuracy.

        Args:
            raw_confidence: Raw model confidence
            regime: Current market regime

        Returns:
            Calibrated confidence
        """
        # Get regime-specific accuracy
        regime_accuracy = self._accuracy_by_regime[regime]
        if regime_accuracy.cm.total_weight >= 20:
            accuracy = regime_accuracy.get()
            # Scale confidence by how well model performs in this regime
            calibration_factor = accuracy * 2  # 0.5 accuracy = 1x, 0.75 accuracy = 1.5x
            calibration_factor = np.clip(calibration_factor, 0.5, 1.5)
            return raw_confidence * calibration_factor

        return raw_confidence

    def _update_calibration(self, confidence: float, was_correct: bool) -> None:
        """Update confidence calibration bins."""
        # Bin confidence into deciles
        bin_idx = min(int(confidence * 10), 9)
        bin_key = f"bin_{bin_idx}"

        if bin_key not in self._calibration_bins:
            self._calibration_bins[bin_key] = []

        self._calibration_bins[bin_key].append((confidence, was_correct))

        # Keep only recent samples
        if len(self._calibration_bins[bin_key]) > 100:
            self._calibration_bins[bin_key] = self._calibration_bins[bin_key][-100:]

    def get_stats(self) -> dict[str, Any]:
        """Get predictor statistics."""
        # Calculate rolling accuracy from window
        rolling_accuracy = 0.0
        if len(self._rolling_window) > 0:
            rolling_accuracy = sum(self._rolling_window) / len(self._rolling_window)

        return {
            'samples_trained': self._samples_trained,
            'predictions_made': self._prediction_count,
            'overall_accuracy': self._accuracy.get() if self._accuracy.cm.total_weight > 0 else 0.0,
            'rolling_accuracy_50': rolling_accuracy,
            'accuracy_by_regime': {
                regime.value: (
                    self._accuracy_by_regime[regime].get()
                    if self._accuracy_by_regime[regime].cm.total_weight >= 10 else None
                )
                for regime in MarketRegime
            },
            'model_type': self.model_type,
            'min_samples_reached': self._samples_trained >= self.MIN_SAMPLES_FOR_PREDICTION,
        }

    def get_feature_importances(self) -> dict[str, float]:
        """
        Get feature importances (for interpretable models).

        Returns:
            Dict of feature -> importance score
        """
        importances = {}

        if self.model_type == "logistic" or self.model_type == "ensemble":
            model = self._logistic if self.model_type == "ensemble" else self._model

            # Get logistic regression weights
            try:
                lr_model = model.steps[-1]
                if hasattr(lr_model, 'weights'):
                    weights = lr_model.weights
                    total_weight = sum(abs(w) for w in weights.values())
                    if total_weight > 0:
                        importances = {
                            k: abs(v) / total_weight
                            for k, v in weights.items()
                        }
            except Exception:
                pass

        return importances

    def save(self) -> None:
        """Save model to disk."""
        if not self.persist_path:
            return

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Save stats and predictions (models are harder to serialize)
            data = {
                'samples_trained': self._samples_trained,
                'prediction_count': self._prediction_count,
                'accuracy': self._accuracy.get() if self._accuracy.cm.total_weight > 0 else 0.0,
                'accuracy_n': int(self._accuracy.cm.total_weight),
                'calibration_bins': self._calibration_bins,
                'model_type': self.model_type,
            }

            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            log.info(f"[OnlinePredictor] Saved to {self.persist_path}")

        except Exception as e:
            log.error(f"[OnlinePredictor] Failed to save: {e}")

    def load(self) -> None:
        """Load model from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            self._samples_trained = data.get('samples_trained', 0)
            self._prediction_count = data.get('prediction_count', 0)
            self._calibration_bins = data.get('calibration_bins', {})

            log.info(
                f"[OnlinePredictor] Loaded state: "
                f"samples={self._samples_trained}, predictions={self._prediction_count}"
            )

        except Exception as e:
            log.error(f"[OnlinePredictor] Failed to load: {e}")

    def reset(self) -> None:
        """Reset model and statistics."""
        self._init_models()
        self._predictions.clear()
        self._prediction_count = 0
        self._samples_trained = 0
        self._accuracy = metrics.Accuracy()
        self._accuracy_by_regime = {regime: metrics.Accuracy() for regime in MarketRegime}
        self._rolling_window.clear()
        self._calibration_bins.clear()

        log.info("[OnlinePredictor] Reset complete")

    def check_health_and_rotate(
        self,
        min_samples: int = 50,
        accuracy_threshold: float = 0.40,
        degradation_window: int = 20,
    ) -> dict[str, Any]:
        """
        Check model health and auto-rotate if performance degrades.

        Args:
            min_samples: Minimum samples before checking accuracy
            accuracy_threshold: Reset if accuracy falls below this (0.40 = 40%)
            degradation_window: Check last N predictions for degradation

        Returns:
            Health report with status and any actions taken
        """
        report = {
            "samples_trained": self._samples_trained,
            "predictions_made": self._prediction_count,
            "overall_accuracy": 0.0,
            "rolling_accuracy": 0.0,
            "status": "healthy",
            "action": None,
        }

        # Get current accuracy
        if self._accuracy.cm.total_weight > 0:
            report["overall_accuracy"] = self._accuracy.get()

        # Get rolling accuracy from recent predictions
        if len(self._rolling_window) >= degradation_window:
            recent = list(self._rolling_window)[-degradation_window:]
            report["rolling_accuracy"] = sum(recent) / len(recent)

        # Check if we have enough samples to evaluate
        if self._samples_trained < min_samples:
            report["status"] = "warming_up"
            log.debug(
                f"[OnlinePredictor] Warming up: {self._samples_trained}/{min_samples} samples"
            )
            return report

        # Check if accuracy metrics have actual data (not just restored samples_trained)
        # After restart, _samples_trained is restored but _accuracy is reset to 0
        if self._accuracy.cm.total_weight < 10:
            report["status"] = "warming_up"
            log.debug(
                f"[OnlinePredictor] Accuracy metrics warming up: "
                f"{int(self._accuracy.cm.total_weight)} samples in accuracy tracker"
            )
            return report

        # Check for degradation in recent predictions
        if len(self._rolling_window) >= degradation_window:
            recent_accuracy = report["rolling_accuracy"]

            if recent_accuracy < accuracy_threshold:
                # Model is underperforming - reset
                log.warning(
                    f"[OnlinePredictor] Model degradation detected! "
                    f"Recent accuracy: {recent_accuracy:.1%} < {accuracy_threshold:.1%} threshold"
                )

                # Save stats before reset for logging
                old_samples = self._samples_trained
                old_predictions = self._prediction_count

                self.reset()

                report["status"] = "rotated"
                report["action"] = (
                    f"Model reset due to low accuracy ({recent_accuracy:.1%}). "
                    f"Previous: {old_samples} samples, {old_predictions} predictions"
                )

                log.info(
                    f"[OnlinePredictor] Auto-rotation complete. "
                    f"Was: {old_samples} samples, {old_predictions} predictions, "
                    f"accuracy {recent_accuracy:.1%}"
                )

        # Check for consistently wrong predictions (worse than random)
        # Only if we have actual accuracy data (not just 0 from no data)
        if (
            self._prediction_count >= 100 and
            report["overall_accuracy"] < 0.35 and
            self._accuracy.cm.total_weight >= 20  # Ensure we have real data
        ):
            log.warning(
                f"[OnlinePredictor] Model performing worse than random! "
                f"Overall accuracy: {report['overall_accuracy']:.1%}"
            )
            report["status"] = "degraded"

        return report
