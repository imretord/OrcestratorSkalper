"""
Meta learner for AI Trading System V3.
Adapts feature weights based on performance in different regimes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.logger import get_logger
from core.state import (
    FeatureImportance,
    LearnerState,
    MarketContext,
    MarketRegime,
    TradeExperience,
)
from learners.experience_buffer import ExperienceBuffer
from learners.online_predictor import OnlinePredictor

log = get_logger("meta_learner")


class MetaLearner:
    """
    Meta learner that adapts feature weights across market regimes.

    Responsibilities:
    - Track feature performance by regime
    - Adapt feature weights based on recent accuracy
    - Provide regime-specific trading recommendations
    - Combine signals from predictor and experience buffer
    - Generate learning insights
    """

    # Exponential moving average decay for feature tracking
    EMA_DECAY = 0.1

    # Minimum experiences for regime-specific recommendations
    MIN_REGIME_EXPERIENCES = 20

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        online_predictor: OnlinePredictor,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize meta learner.

        Args:
            experience_buffer: Buffer of trade experiences
            online_predictor: Online ML predictor
            persist_path: Path for state persistence
        """
        self.experience_buffer = experience_buffer
        self.predictor = online_predictor
        self.persist_path = Path(persist_path) if persist_path else None

        # Feature importance tracking
        self._feature_scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {"importance": 0.5, "win_correlation": 0.0, "samples": 0}
        )

        # Regime-specific feature weights
        self._regime_feature_weights: dict[MarketRegime, dict[str, float]] = {
            regime: {} for regime in MarketRegime
        }

        # Performance tracking
        self._regime_performance: dict[MarketRegime, dict[str, Any]] = {
            regime: {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "best_features": [],
            }
            for regime in MarketRegime
        }

        # Recommendations history
        self._recommendation_history: list[dict[str, Any]] = []

        # Load state if available
        if self.persist_path and self.persist_path.exists():
            self._load_state()

        log.info("[MetaLearner] Initialized")

    def update_from_experience(self, experience: TradeExperience) -> None:
        """
        Update meta learner from new trade experience.

        Args:
            experience: Completed trade experience
        """
        regime = experience.regime_at_entry
        features = self.experience_buffer.extract_features(experience)
        success = experience.success

        # Update feature correlations with win/loss
        for feature_name, feature_value in features.items():
            self._update_feature_correlation(feature_name, feature_value, success, regime)

        # Update regime performance
        perf = self._regime_performance[regime]
        perf["trades"] += 1
        if success:
            perf["wins"] += 1
        perf["total_pnl"] += experience.pnl_percent
        perf["avg_pnl"] = perf["total_pnl"] / perf["trades"]

        # Update regime-specific feature weights periodically
        if perf["trades"] % 10 == 0:
            self._recalculate_regime_weights(regime)

        log.debug(
            f"[MetaLearner] Updated from experience: regime={regime.value}, "
            f"success={success}, pnl={experience.pnl_percent:.2f}%"
        )

    def _update_feature_correlation(
        self,
        feature_name: str,
        feature_value: float,
        success: bool,
        regime: MarketRegime,
    ) -> None:
        """
        Update feature correlation with trade success.

        Args:
            feature_name: Name of feature
            feature_value: Feature value
            success: Whether trade was successful
            regime: Market regime at trade
        """
        score_dict = self._feature_scores[feature_name]

        # Update sample count
        score_dict["samples"] += 1
        n = score_dict["samples"]

        # Update win correlation using EMA
        outcome = 1.0 if success else -1.0
        current_corr = score_dict["win_correlation"]

        # Weight by feature magnitude (higher values should have more impact)
        weighted_outcome = outcome * abs(feature_value)

        # EMA update
        alpha = min(self.EMA_DECAY, 1 / n)  # Adaptive alpha for early samples
        new_corr = (1 - alpha) * current_corr + alpha * weighted_outcome
        score_dict["win_correlation"] = np.clip(new_corr, -1, 1)

        # Update importance based on correlation strength
        score_dict["importance"] = (1 + abs(score_dict["win_correlation"])) / 2

    def _recalculate_regime_weights(self, regime: MarketRegime) -> None:
        """
        Recalculate feature weights for a specific regime.

        Args:
            regime: Market regime to recalculate
        """
        regime_experiences = self.experience_buffer.get_by_regime(regime)

        if len(regime_experiences) < self.MIN_REGIME_EXPERIENCES:
            return

        # Calculate feature-outcome correlations for this regime
        feature_correlations: dict[str, list[tuple[float, bool]]] = defaultdict(list)

        for exp in regime_experiences:
            features = self.experience_buffer.extract_features(exp)
            for fname, fvalue in features.items():
                feature_correlations[fname].append((fvalue, exp.success))

        # Calculate correlation coefficients
        weights = {}
        for fname, values in feature_correlations.items():
            if len(values) < 10:
                continue

            feature_vals = np.array([v[0] for v in values])
            outcomes = np.array([1.0 if v[1] else 0.0 for v in values])

            # Simple correlation
            if np.std(feature_vals) > 0 and np.std(outcomes) > 0:
                corr = np.corrcoef(feature_vals, outcomes)[0, 1]
                if not np.isnan(corr):
                    weights[fname] = corr

        self._regime_feature_weights[regime] = weights

        # Update best features
        sorted_features = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        self._regime_performance[regime]["best_features"] = [
            {"name": f, "correlation": c} for f, c in sorted_features[:5]
        ]

        log.debug(f"[MetaLearner] Recalculated weights for {regime.value}")

    def get_regime_recommendation(
        self,
        context: MarketContext,
    ) -> dict[str, Any]:
        """
        Get regime-specific trading recommendation.

        Args:
            context: Current market context

        Returns:
            Recommendation dict with bias, confidence, and reasoning
        """
        regime = context.regime.regime
        perf = self._regime_performance[regime]

        # Check if we have enough data for this regime
        if perf["trades"] < self.MIN_REGIME_EXPERIENCES:
            return {
                "has_data": False,
                "regime": regime.value,
                "message": f"Insufficient data for {regime.value} (need {self.MIN_REGIME_EXPERIENCES}, have {perf['trades']})",
                "suggested_action": "use_default",
            }

        # Calculate regime-specific confidence
        win_rate = perf["wins"] / perf["trades"]
        avg_pnl = perf["avg_pnl"]

        # Get feature weights for this regime
        weights = self._regime_feature_weights.get(regime, {})

        # Extract current features
        features = self.predictor._extract_features(context, context.suggested_bias)

        # Calculate weighted feature score
        weighted_score = 0.0
        weight_sum = 0.0

        for fname, fvalue in features.items():
            if fname in weights:
                weighted_score += fvalue * weights[fname]
                weight_sum += abs(weights[fname])

        if weight_sum > 0:
            normalized_score = weighted_score / weight_sum
        else:
            normalized_score = 0.0

        # Determine recommendation
        if normalized_score > 0.1:
            regime_bias = "long"
        elif normalized_score < -0.1:
            regime_bias = "short"
        else:
            regime_bias = "neutral"

        # Calculate confidence
        confidence = min(abs(normalized_score) * win_rate * 2, 0.95)

        recommendation = {
            "has_data": True,
            "regime": regime.value,
            "regime_bias": regime_bias,
            "confidence": confidence,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sample_size": perf["trades"],
            "feature_score": normalized_score,
            "top_features": perf["best_features"][:3],
        }

        # Save to history
        self._recommendation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": context.symbol,
            **recommendation,
        })

        # Keep history limited
        if len(self._recommendation_history) > 500:
            self._recommendation_history = self._recommendation_history[-500:]

        return recommendation

    def get_combined_signal(
        self,
        context: MarketContext,
    ) -> dict[str, Any]:
        """
        Get combined signal from all learner components.

        Args:
            context: Current market context

        Returns:
            Combined signal with multiple perspectives
        """
        # Get predictor signal
        predictor_pred = self.predictor.predict(context, context.suggested_bias)

        # Get regime recommendation
        regime_rec = self.get_regime_recommendation(context)

        # Get context signal
        context_signal = {
            "bias": context.suggested_bias,
            "confidence": context.confidence,
            "momentum": context.momentum_score,
        }

        # Combine signals
        signals = []
        weights = []

        # Context signal (base)
        if context.suggested_bias != "neutral":
            context_value = 1.0 if context.suggested_bias == "long" else -1.0
            signals.append(context_value * context.confidence)
            weights.append(0.4)

        # Predictor signal
        if predictor_pred.predicted_direction != "neutral":
            pred_value = 1.0 if predictor_pred.predicted_direction == "up" else -1.0
            signals.append(pred_value * predictor_pred.confidence)
            weights.append(0.3 if self.predictor._samples_trained >= 50 else 0.1)

        # Regime signal
        if regime_rec.get("has_data") and regime_rec.get("regime_bias") != "neutral":
            regime_value = 1.0 if regime_rec["regime_bias"] == "long" else -1.0
            signals.append(regime_value * regime_rec["confidence"])
            weights.append(0.3)

        # Calculate combined score
        if signals and weights:
            total_weight = sum(weights)
            combined_score = sum(s * w for s, w in zip(signals, weights)) / total_weight
        else:
            combined_score = 0.0

        # Determine final bias
        if combined_score > 0.15:
            final_bias = "long"
        elif combined_score < -0.15:
            final_bias = "short"
        else:
            final_bias = "neutral"

        # Agreement check
        biases = [context.suggested_bias, predictor_pred.predicted_direction]
        if regime_rec.get("has_data"):
            biases.append(regime_rec.get("regime_bias", "neutral"))

        non_neutral = [b for b in biases if b not in ["neutral", None]]
        if non_neutral:
            agreement = len(set(non_neutral)) == 1
        else:
            agreement = True

        return {
            "symbol": context.symbol,
            "final_bias": final_bias,
            "combined_score": combined_score,
            "final_confidence": abs(combined_score),
            "agreement": agreement,
            "components": {
                "context": context_signal,
                "predictor": {
                    "direction": predictor_pred.predicted_direction,
                    "confidence": predictor_pred.confidence,
                },
                "regime": regime_rec,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_feature_importances(self) -> list[FeatureImportance]:
        """
        Get ranked feature importances.

        Returns:
            List of FeatureImportance objects
        """
        importances = []

        for fname, scores in self._feature_scores.items():
            if scores["samples"] < 10:
                continue

            # Determine trend (compare to previous)
            trend = "stable"  # Simplified - would need history tracking for real trend

            # Get regime-specific scores
            regime_specific = {}
            for regime, weights in self._regime_feature_weights.items():
                if fname in weights:
                    regime_specific[regime.value] = abs(weights[fname])

            importances.append(FeatureImportance(
                feature_name=fname,
                importance_score=scores["importance"],
                trend_24h=trend,
                regime_specific=regime_specific,
            ))

        # Sort by importance
        importances.sort(key=lambda x: x.importance_score, reverse=True)

        return importances

    def get_learner_state(self) -> LearnerState:
        """
        Get complete learner state.

        Returns:
            LearnerState object
        """
        # Get predictor stats
        pred_stats = self.predictor.get_stats()

        # Calculate accuracy by regime
        accuracy_by_regime = {}
        for regime, perf in self._regime_performance.items():
            if perf["trades"] >= 10:
                accuracy_by_regime[regime.value] = perf["wins"] / perf["trades"]

        # Get recent accuracy
        recent_10 = 0.0
        recent_50 = 0.0

        if len(self.experience_buffer) >= 10:
            recent = list(self.experience_buffer)[-10:]
            recent_10 = sum(1 for e in recent if e.success) / len(recent)

        if len(self.experience_buffer) >= 50:
            recent = list(self.experience_buffer)[-50:]
            recent_50 = sum(1 for e in recent if e.success) / len(recent)

        return LearnerState(
            experiences_count=len(self.experience_buffer),
            predictions_made=pred_stats["predictions_made"],
            predictions_correct=int(pred_stats["overall_accuracy"] * pred_stats["predictions_made"]),
            accuracy_rate=pred_stats["overall_accuracy"],
            accuracy_by_regime=accuracy_by_regime,
            feature_importances=self.get_feature_importances()[:10],
            recent_accuracy_10=recent_10,
            recent_accuracy_50=recent_50,
            last_update=datetime.now(timezone.utc),
        )

    def get_insights(self) -> list[str]:
        """
        Generate human-readable learning insights.

        Returns:
            List of insight strings
        """
        insights = []

        # Overall performance
        buffer_stats = self.experience_buffer.get_stats()
        if buffer_stats["total_trades"] >= 10:
            insights.append(
                f"Overall win rate: {buffer_stats['win_rate']:.1f}% "
                f"across {buffer_stats['total_trades']} trades"
            )

        # Best/worst regime
        best_regime = None
        worst_regime = None
        best_win_rate = 0.0
        worst_win_rate = 1.0

        for regime, perf in self._regime_performance.items():
            if perf["trades"] >= 10:
                win_rate = perf["wins"] / perf["trades"]
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_regime = regime
                if win_rate < worst_win_rate:
                    worst_win_rate = win_rate
                    worst_regime = regime

        if best_regime:
            insights.append(
                f"Best regime: {best_regime.value} ({best_win_rate:.0%} win rate)"
            )

        if worst_regime and worst_regime != best_regime:
            insights.append(
                f"Avoid trading in: {worst_regime.value} ({worst_win_rate:.0%} win rate)"
            )

        # Top features
        top_features = self.get_feature_importances()[:3]
        if top_features:
            feature_names = [f.feature_name for f in top_features]
            insights.append(f"Most predictive features: {', '.join(feature_names)}")

        # Predictor status
        pred_stats = self.predictor.get_stats()
        if pred_stats["samples_trained"] < 50:
            insights.append(
                f"Predictor warming up: {pred_stats['samples_trained']}/50 samples"
            )
        elif pred_stats["rolling_accuracy_50"] > 0.55:
            insights.append(
                f"Predictor performing well: {pred_stats['rolling_accuracy_50']:.0%} recent accuracy"
            )

        return insights

    def _save_state(self) -> None:
        """Save state to disk."""
        if not self.persist_path:
            return

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "feature_scores": dict(self._feature_scores),
                "regime_feature_weights": {
                    r.value: w for r, w in self._regime_feature_weights.items()
                },
                "regime_performance": {
                    r.value: p for r, p in self._regime_performance.items()
                },
            }

            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            log.debug(f"[MetaLearner] Saved state to {self.persist_path}")

        except Exception as e:
            log.error(f"[MetaLearner] Failed to save: {e}")

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            self._feature_scores = defaultdict(
                lambda: {"importance": 0.5, "win_correlation": 0.0, "samples": 0},
                data.get("feature_scores", {}),
            )

            for regime_str, weights in data.get("regime_feature_weights", {}).items():
                try:
                    regime = MarketRegime(regime_str)
                    self._regime_feature_weights[regime] = weights
                except ValueError:
                    pass

            for regime_str, perf in data.get("regime_performance", {}).items():
                try:
                    regime = MarketRegime(regime_str)
                    self._regime_performance[regime].update(perf)
                except ValueError:
                    pass

            log.info(f"[MetaLearner] Loaded state from {self.persist_path}")

        except Exception as e:
            log.error(f"[MetaLearner] Failed to load: {e}")

    def save(self) -> None:
        """Save all learner state."""
        self._save_state()
        self.predictor.save()

    def reset(self) -> None:
        """Reset all learning."""
        self._feature_scores.clear()
        self._regime_feature_weights = {regime: {} for regime in MarketRegime}
        self._regime_performance = {
            regime: {
                "trades": 0,
                "wins": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "best_features": [],
            }
            for regime in MarketRegime
        }
        self._recommendation_history.clear()

        self.predictor.reset()

        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

        log.info("[MetaLearner] Reset complete")
