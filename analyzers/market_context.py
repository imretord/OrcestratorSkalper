"""
Market context builder for AI Trading System V3.
Combines all analyses into a complete trading context.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from analyzers.base_analyzer import BaseAnalyzer
from analyzers.regime_detector import RegimeDetector
from analyzers.sentiment_analyzer import SentimentAnalyzer
from core.logger import get_logger
from core.state import (
    MarketContext,
    MarketRegime,
    RegimeAnalysis,
    SensorSnapshot,
    SentimentAnalysis,
    IndicatorValues,
)
from execution.binance_client import BinanceClient

log = get_logger("market_context")


class MarketContextBuilder(BaseAnalyzer[MarketContext]):
    """
    Builds complete market context by combining all analyses.

    Combines:
    - Regime Analysis (trend, volatility, support/resistance)
    - Sentiment Analysis (fear & greed, L/S ratio, OI)
    - Price Feed Data (OHLCV, indicators)
    - Volume Data (relative volume, buy pressure)
    - Funding Data (funding rate, trend)

    Produces actionable trading context with bias and confidence.
    """

    def __init__(
        self,
        client: BinanceClient,
        regime_cache_seconds: int = 60,
        sentiment_cache_seconds: int = 300,
    ) -> None:
        """
        Initialize market context builder.

        Args:
            client: Binance client for data access
            regime_cache_seconds: Cache duration for regime analysis
            sentiment_cache_seconds: Cache duration for sentiment analysis
        """
        super().__init__("MarketContextBuilder")
        self.client = client

        # Initialize sub-analyzers
        self.regime_detector = RegimeDetector(cache_seconds=regime_cache_seconds)
        self.sentiment_analyzer = SentimentAnalyzer(
            client=client,
            cache_seconds=sentiment_cache_seconds,
        )

    def analyze(self, snapshot: SensorSnapshot) -> MarketContext:
        """
        Build complete market context from sensor snapshot.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            MarketContext with all analyses and recommendations
        """
        symbol = snapshot.symbol
        price = snapshot.price_feed.current_price

        # Perform regime analysis
        regime = self.regime_detector.analyze_cached(snapshot)
        if regime is None:
            # Fallback regime
            regime = self._create_fallback_regime(snapshot)

        # Perform sentiment analysis
        sentiment = self.sentiment_analyzer.analyze_cached(snapshot)
        if sentiment is None:
            # Fallback sentiment
            sentiment = self._create_fallback_sentiment(snapshot)

        # Calculate derived signals
        trend_aligned = self._check_trend_alignment(snapshot, regime)
        momentum_score = self._calculate_momentum_score(snapshot, regime)
        volatility_adjusted_score = self._calculate_volatility_adjusted_score(
            momentum_score, regime.volatility_level
        )

        # Determine trading bias and confidence
        suggested_bias, confidence, risk_level = self._calculate_trading_recommendation(
            regime=regime,
            sentiment=sentiment,
            trend_aligned=trend_aligned,
            momentum_score=momentum_score,
            volatility_adjusted_score=volatility_adjusted_score,
            volume_data=snapshot.volume,
        )

        return MarketContext(
            symbol=symbol,
            current_price=price,
            regime=regime,
            sentiment=sentiment,
            price_feed=snapshot.price_feed,
            volume=snapshot.volume,
            funding=snapshot.funding,
            trend_aligned=trend_aligned,
            momentum_score=momentum_score,
            volatility_adjusted_score=volatility_adjusted_score,
            suggested_bias=suggested_bias,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.now(timezone.utc),
        )

    def _create_fallback_regime(self, snapshot: SensorSnapshot) -> RegimeAnalysis:
        """Create fallback regime analysis if detector fails."""
        return RegimeAnalysis(
            symbol=snapshot.symbol,
            regime=MarketRegime.RANGING,
            confidence=0.3,
            trend_strength=0.0,
            volatility_level="medium",
            regime_duration_hours=1,
            support_level=None,
            resistance_level=None,
            regime_description="Unable to determine regime - defaulting to ranging",
            timestamp=datetime.now(timezone.utc),
        )

    def _create_fallback_sentiment(self, snapshot: SensorSnapshot) -> SentimentAnalysis:
        """Create fallback sentiment analysis if analyzer fails."""
        return SentimentAnalysis(
            symbol=snapshot.symbol,
            fear_greed_index=50,
            fear_greed_label="Neutral",
            long_short_ratio=1.0,
            open_interest_change_24h=0.0,
            funding_sentiment="neutral",
            social_sentiment="neutral",
            overall_sentiment="neutral",
            sentiment_score=0.0,
            timestamp=datetime.now(timezone.utc),
        )

    def _check_trend_alignment(
        self,
        snapshot: SensorSnapshot,
        regime: RegimeAnalysis,
    ) -> bool:
        """
        Check if multiple timeframes agree on trend direction.

        Args:
            snapshot: Sensor data
            regime: Regime analysis

        Returns:
            True if trends are aligned
        """
        indicators = snapshot.price_feed.indicators
        price = snapshot.price_feed.current_price

        # Check alignment across timeframes
        alignments = []

        for tf in ["15m", "1h", "4h"]:
            ind = indicators.get(tf, IndicatorValues())

            if ind.ema_20 and ind.ema_50:
                if price > ind.ema_20 > ind.ema_50:
                    alignments.append(1)  # Bullish
                elif price < ind.ema_20 < ind.ema_50:
                    alignments.append(-1)  # Bearish
                else:
                    alignments.append(0)  # Mixed

        if not alignments:
            return False

        # Check if all non-zero alignments agree
        non_zero = [a for a in alignments if a != 0]
        if len(non_zero) >= 2:
            return all(a == non_zero[0] for a in non_zero)

        return False

    def _calculate_momentum_score(
        self,
        snapshot: SensorSnapshot,
        regime: RegimeAnalysis,
    ) -> float:
        """
        Calculate momentum score from -1 to +1.

        Args:
            snapshot: Sensor data
            regime: Regime analysis

        Returns:
            Momentum score
        """
        scores = []

        indicators = snapshot.price_feed.indicators
        ind_1h = indicators.get("1h", IndicatorValues())

        # RSI contribution
        if ind_1h.rsi_14:
            rsi = ind_1h.rsi_14
            # RSI 30-70 maps to -0.5 to +0.5
            rsi_score = (rsi - 50) / 40
            scores.append(np.clip(rsi_score, -1, 1))

        # MACD contribution
        if ind_1h.macd_histogram is not None:
            price = snapshot.price_feed.current_price
            # Normalize by price
            macd_score = np.clip(ind_1h.macd_histogram / (price * 0.001), -1, 1)
            scores.append(macd_score)

        # Regime trend strength contribution
        if regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND, MarketRegime.BREAKOUT_UP]:
            scores.append(regime.trend_strength)
        elif regime.regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND, MarketRegime.BREAKOUT_DOWN]:
            scores.append(-regime.trend_strength)

        # Volume contribution
        if snapshot.volume.volume_spike:
            # Volume spike amplifies momentum direction
            volume_contribution = 0.3 * np.sign(np.mean(scores) if scores else 0)
            scores.append(volume_contribution)

        if not scores:
            return 0.0

        return round(np.clip(np.mean(scores), -1, 1), 3)

    def _calculate_volatility_adjusted_score(
        self,
        momentum_score: float,
        volatility_level: str,
    ) -> float:
        """
        Adjust momentum score based on volatility.

        High volatility reduces confidence, low volatility increases it.

        Args:
            momentum_score: Raw momentum score
            volatility_level: "low"/"medium"/"high"/"extreme"

        Returns:
            Volatility-adjusted score
        """
        volatility_multipliers = {
            "low": 1.2,      # Boost signal in low vol
            "medium": 1.0,   # Neutral
            "high": 0.8,     # Reduce signal in high vol
            "extreme": 0.5,  # Significantly reduce in extreme vol
        }

        multiplier = volatility_multipliers.get(volatility_level, 1.0)
        adjusted = momentum_score * multiplier

        return round(np.clip(adjusted, -1, 1), 3)

    def _calculate_trading_recommendation(
        self,
        regime: RegimeAnalysis,
        sentiment: SentimentAnalysis,
        trend_aligned: bool,
        momentum_score: float,
        volatility_adjusted_score: float,
        volume_data: Any,
    ) -> tuple[str, float, str]:
        """
        Calculate trading recommendation.

        Args:
            regime: Regime analysis
            sentiment: Sentiment analysis
            trend_aligned: Whether trends are aligned
            momentum_score: Momentum score
            volatility_adjusted_score: Volatility-adjusted score
            volume_data: Volume data

        Returns:
            Tuple of (bias, confidence, risk_level)
        """
        # Base score from volatility-adjusted momentum
        base_score = volatility_adjusted_score

        # Sentiment contribution (weight: 20%)
        sentiment_contribution = sentiment.sentiment_score * 0.2
        combined_score = base_score * 0.8 + sentiment_contribution

        # Regime-based adjustments
        regime_boost = 0.0
        if regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.BREAKOUT_UP]:
            regime_boost = 0.2
        elif regime.regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.BREAKOUT_DOWN]:
            regime_boost = -0.2
        elif regime.regime in [MarketRegime.CHOPPY]:
            combined_score *= 0.5  # Reduce confidence in choppy markets

        combined_score = np.clip(combined_score + regime_boost, -1, 1)

        # Determine bias
        if combined_score > 0.15:
            bias = "long"
        elif combined_score < -0.15:
            bias = "short"
        else:
            bias = "neutral"

        # Calculate confidence
        confidence = abs(combined_score)

        # Boost confidence if trends are aligned
        if trend_aligned and bias != "neutral":
            confidence = min(confidence * 1.2, 0.95)

        # Boost confidence with volume confirmation
        if volume_data.volume_spike and bias != "neutral":
            confidence = min(confidence * 1.1, 0.95)

        # Reduce confidence in certain regimes
        if regime.regime in [MarketRegime.CHOPPY, MarketRegime.COMPRESSION]:
            confidence *= 0.8

        # Determine risk level
        risk_level = self._calculate_risk_level(
            regime=regime,
            sentiment=sentiment,
            confidence=confidence,
        )

        return bias, round(confidence, 3), risk_level

    def _calculate_risk_level(
        self,
        regime: RegimeAnalysis,
        sentiment: SentimentAnalysis,
        confidence: float,
    ) -> str:
        """
        Calculate risk level for trading.

        Args:
            regime: Regime analysis
            sentiment: Sentiment analysis
            confidence: Trading confidence

        Returns:
            "low", "medium", or "high"
        """
        risk_factors = 0

        # High volatility = higher risk
        if regime.volatility_level in ["high", "extreme"]:
            risk_factors += 2
        elif regime.volatility_level == "medium":
            risk_factors += 1

        # Choppy regime = higher risk
        if regime.regime == MarketRegime.CHOPPY:
            risk_factors += 2

        # Extreme sentiment = contrarian risk
        if sentiment.fear_greed_index <= 20 or sentiment.fear_greed_index >= 80:
            risk_factors += 1

        # Low confidence = higher risk
        if confidence < 0.3:
            risk_factors += 2
        elif confidence < 0.5:
            risk_factors += 1

        # Determine risk level
        if risk_factors >= 4:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"

    def get_context_summary(self, context: MarketContext) -> str:
        """
        Generate human-readable context summary.

        Args:
            context: Market context

        Returns:
            Summary string
        """
        lines = [
            f"=== {context.symbol} Market Context ===",
            f"Price: ${context.current_price:.4f}",
            "",
            f"Regime: {context.regime.regime.value} (conf: {context.regime.confidence:.0%})",
            f"  - Trend: {context.regime.trend_strength:.0%} strength, {context.regime.volatility_level} volatility",
            f"  - Duration: {context.regime.regime_duration_hours}h",
            "",
            f"Sentiment: {context.sentiment.overall_sentiment} (score: {context.sentiment.sentiment_score:+.2f})",
            f"  - Fear & Greed: {context.sentiment.fear_greed_index} ({context.sentiment.fear_greed_label})",
            f"  - L/S Ratio: {context.sentiment.long_short_ratio:.2f}",
            "",
            f"Signals:",
            f"  - Trend Aligned: {'Yes' if context.trend_aligned else 'No'}",
            f"  - Momentum: {context.momentum_score:+.2f}",
            f"  - Vol-Adjusted: {context.volatility_adjusted_score:+.2f}",
            "",
            f"Recommendation: {context.suggested_bias.upper()}",
            f"  - Confidence: {context.confidence:.0%}",
            f"  - Risk Level: {context.risk_level}",
        ]

        return "\n".join(lines)

    def build_for_all(
        self,
        snapshots: dict[str, SensorSnapshot],
    ) -> dict[str, MarketContext]:
        """
        Build context for multiple symbols.

        Args:
            snapshots: Dict of symbol -> SensorSnapshot

        Returns:
            Dict of symbol -> MarketContext
        """
        contexts = {}

        for symbol, snapshot in snapshots.items():
            try:
                context = self.safe_analyze(snapshot)
                if context:
                    contexts[symbol] = context
            except Exception as e:
                log.error(f"Failed to build context for {symbol}: {e}")

        return contexts

    def rank_opportunities(
        self,
        contexts: dict[str, MarketContext],
    ) -> list[tuple[str, MarketContext, float]]:
        """
        Rank trading opportunities by score.

        Args:
            contexts: Dict of symbol -> MarketContext

        Returns:
            List of (symbol, context, score) sorted by score descending
        """
        scored = []

        for symbol, context in contexts.items():
            # Skip neutral bias
            if context.suggested_bias == "neutral":
                continue

            # Calculate opportunity score
            score = context.confidence

            # Boost for trend alignment
            if context.trend_aligned:
                score *= 1.2

            # Reduce for high risk
            if context.risk_level == "high":
                score *= 0.7
            elif context.risk_level == "medium":
                score *= 0.9

            # Boost for breakouts
            if context.regime.regime in [MarketRegime.BREAKOUT_UP, MarketRegime.BREAKOUT_DOWN]:
                score *= 1.3

            scored.append((symbol, context, round(score, 3)))

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        return scored
