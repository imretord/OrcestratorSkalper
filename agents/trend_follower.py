"""
Trend Follower Agent for AI Trading System V3.
Specializes in trading with the trend - buys pullbacks in uptrends, sells rallies in downtrends.
"""
from __future__ import annotations

from core.logger import get_logger
from core.state import (
    MarketContext,
    MarketRegime,
    Signal,
)
from agents.base_agent import BaseAgent
from learners.meta_learner import MetaLearner
from learners.online_predictor import OnlinePredictor

log = get_logger("trend_follower")


class TrendFollowerAgent(BaseAgent):
    """
    Trend Following Agent.

    Strategy:
    - In uptrends: Buy pullbacks to EMA20/EMA50 with RSI oversold
    - In downtrends: Sell rallies to EMA20/EMA50 with RSI overbought

    Best in: STRONG_UPTREND, WEAK_UPTREND, STRONG_DOWNTREND, WEAK_DOWNTREND
    Avoids: RANGING, CHOPPY, COMPRESSION
    """

    SUITABLE_REGIMES = [
        MarketRegime.STRONG_UPTREND,
        MarketRegime.WEAK_UPTREND,
        MarketRegime.STRONG_DOWNTREND,
        MarketRegime.WEAK_DOWNTREND,
    ]

    def __init__(
        self,
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
        pullback_rsi_threshold: float = 40,
        ema_touch_tolerance: float = 0.015,  # 1.5% вместо 0.5% - расширено для больше точек входа
        min_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize TrendFollower agent.

        Args:
            predictor: Online predictor for ML predictions
            meta_learner: Meta learner for adaptive insights
            pullback_rsi_threshold: RSI threshold for pullback detection (default 40)
            ema_touch_tolerance: Tolerance for EMA touch detection (0.5% default)
            min_confidence_threshold: Minimum confidence to generate signal
        """
        super().__init__(
            name="TrendFollower",
            description="Trades pullbacks in trending markets",
            suitable_regimes=self.SUITABLE_REGIMES,
            predictor=predictor,
            meta_learner=meta_learner,
            min_confidence_threshold=min_confidence_threshold,
        )

        self.pullback_rsi_threshold = pullback_rsi_threshold
        self.ema_touch_tolerance = ema_touch_tolerance

    async def analyze(self, context: MarketContext) -> Signal | None:
        """
        Analyze market context for trend following opportunity.

        Args:
            context: Complete market context

        Returns:
            Signal if pullback entry found, None otherwise
        """
        regime = context.regime.regime

        # Check if we're suitable for this regime
        if not self.is_suitable(regime):
            log.debug(f"[TrendFollower] Not suitable for regime: {regime.value}")
            return None

        # Determine trade direction based on regime
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            direction = "LONG"
        else:
            direction = "SHORT"

        # Check for pullback entry
        is_entry, entry_reasons = self._check_pullback_entry(context, direction)

        if not is_entry:
            log.debug(f"[TrendFollower] No pullback entry for {context.symbol}")
            return None

        # Get ML prediction
        ml_prediction, ml_confidence = self._get_ml_prediction(context, direction)

        # Generate warnings
        warnings = self._generate_warnings(context, ml_prediction)

        # Calculate confidence
        confidence = self._calculate_confidence(context, entry_reasons, ml_prediction)

        # Check minimum confidence
        if confidence < self.min_confidence_threshold:
            log.info(
                f"[TrendFollower] {context.symbol} confidence too low: "
                f"{confidence:.2f} < {self.min_confidence_threshold}"
            )
            return None

        # Calculate entry, SL, TP
        entry_price = context.current_price

        # Get ATR for stop-loss
        ind_1h = context.price_feed.indicators.get("1h")
        atr = ind_1h.atr_14 if ind_1h and ind_1h.atr_14 else entry_price * 0.02

        stop_loss = self._calculate_stop_loss(direction, entry_price, atr, regime)

        # Adjust R:R based on trend strength (conservative targets for faster fills)
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            tp1_rr, tp2_rr = 1.2, 2.0  # Moderate R:R in strong trends
        else:
            tp1_rr, tp2_rr = 1.0, 1.5  # Conservative R:R in weak trends

        tp1, tp2 = self._calculate_take_profits(direction, entry_price, stop_loss, tp1_rr, tp2_rr)

        # Create signal
        signal = self._create_signal(
            context=context,
            side=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            confidence=confidence,
            reasoning=entry_reasons,
            warnings=warnings,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
        )

        log.info(
            f"[TrendFollower] Signal generated for {context.symbol}: "
            f"{direction} @ {entry_price:.2f}, conf={confidence:.2f}"
        )

        return signal

    def _check_pullback_entry(
        self,
        context: MarketContext,
        direction: str,
    ) -> tuple[bool, list[str]]:
        """
        Check for pullback entry conditions.

        Args:
            context: Market context
            direction: "LONG" or "SHORT"

        Returns:
            Tuple of (is_entry, reasons)
        """
        reasons: list[str] = []
        checks_passed = 0
        required_checks = 2  # Need at least 2 confirmations

        price = context.current_price
        ind_1h = context.price_feed.indicators.get("1h")
        ind_15m = context.price_feed.indicators.get("15m")

        if not ind_1h:
            return (False, [])

        rsi = ind_1h.rsi_14 or 50
        ema_20 = ind_1h.ema_20
        ema_50 = ind_1h.ema_50
        macd_hist = ind_1h.macd_histogram or 0

        if direction == "LONG":
            # Check 1: RSI pullback (oversold in uptrend)
            if rsi < self.pullback_rsi_threshold:
                reasons.append(f"RSI pullback: {rsi:.1f} < {self.pullback_rsi_threshold}")
                checks_passed += 1
            elif rsi < 50:
                reasons.append(f"RSI below midline: {rsi:.1f}")
                checks_passed += 0.5

            # Check 2: Price near EMA20 or EMA50
            if ema_20:
                ema_distance = (price - ema_20) / ema_20
                if abs(ema_distance) < self.ema_touch_tolerance:
                    reasons.append(f"Price touching EMA20 ({ema_distance*100:+.2f}%)")
                    checks_passed += 1
                elif ema_distance < 0 and ema_distance > -self.ema_touch_tolerance * 2:
                    reasons.append(f"Price slightly below EMA20")
                    checks_passed += 0.5

            if ema_50 and checks_passed < required_checks:
                ema_distance = (price - ema_50) / ema_50
                if abs(ema_distance) < self.ema_touch_tolerance:
                    reasons.append(f"Price touching EMA50 ({ema_distance*100:+.2f}%)")
                    checks_passed += 1

            # Alternative: RSI deep pullback without EMA touch
            # If RSI pulled back deep enough, can enter even without EMA touch
            if checks_passed < required_checks and rsi < 40 and macd_hist >= 0:
                reasons.append(f"RSI deep pullback ({rsi:.1f}) without EMA - entry allowed")
                checks_passed += 1

            # Check 3: MACD histogram still positive or recovering
            if macd_hist > 0:
                reasons.append(f"MACD histogram positive: {macd_hist:.6f}")
                checks_passed += 0.5

            # Check 4: Trend alignment from context
            if context.trend_aligned and context.suggested_bias == "long":
                reasons.append("Multi-timeframe trend aligned LONG")
                checks_passed += 1

            # Check 5: Volume confirmation
            if context.volume.relative_volume > 1.2:
                reasons.append(f"Volume above average: {context.volume.relative_volume:.1f}x")
                checks_passed += 0.5

        else:  # SHORT
            # Check 1: RSI rally (overbought in downtrend)
            if rsi > (100 - self.pullback_rsi_threshold):
                reasons.append(f"RSI rally: {rsi:.1f} > {100 - self.pullback_rsi_threshold}")
                checks_passed += 1
            elif rsi > 50:
                reasons.append(f"RSI above midline: {rsi:.1f}")
                checks_passed += 0.5

            # Check 2: Price near EMA20 or EMA50
            if ema_20:
                ema_distance = (price - ema_20) / ema_20
                if abs(ema_distance) < self.ema_touch_tolerance:
                    reasons.append(f"Price touching EMA20 ({ema_distance*100:+.2f}%)")
                    checks_passed += 1
                elif ema_distance > 0 and ema_distance < self.ema_touch_tolerance * 2:
                    reasons.append(f"Price slightly above EMA20")
                    checks_passed += 0.5

            if ema_50 and checks_passed < required_checks:
                ema_distance = (price - ema_50) / ema_50
                if abs(ema_distance) < self.ema_touch_tolerance:
                    reasons.append(f"Price touching EMA50 ({ema_distance*100:+.2f}%)")
                    checks_passed += 1

            # Alternative: RSI deep rally without EMA touch
            # If RSI rallied deep enough, can enter even without EMA touch
            if checks_passed < required_checks and rsi > 60 and macd_hist <= 0:
                reasons.append(f"RSI deep rally ({rsi:.1f}) without EMA - entry allowed")
                checks_passed += 1

            # Check 3: MACD histogram still negative or weakening
            if macd_hist < 0:
                reasons.append(f"MACD histogram negative: {macd_hist:.6f}")
                checks_passed += 0.5

            # Check 4: Trend alignment from context
            if context.trend_aligned and context.suggested_bias == "short":
                reasons.append("Multi-timeframe trend aligned SHORT")
                checks_passed += 1

            # Check 5: Volume confirmation
            if context.volume.relative_volume > 1.2:
                reasons.append(f"Volume above average: {context.volume.relative_volume:.1f}x")
                checks_passed += 0.5

        is_entry = checks_passed >= required_checks
        return (is_entry, reasons)

    def _calculate_confidence(
        self,
        context: MarketContext,
        reasons: list[str],
        ml_prediction: str,
    ) -> float:
        """
        Calculate signal confidence.

        Args:
            context: Market context
            reasons: List of entry reasons
            ml_prediction: ML prediction result

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5  # Base confidence

        # Add confidence for each reason (up to +0.3)
        reason_bonus = min(len(reasons) * 0.06, 0.3)
        confidence += reason_bonus

        # Adjust for regime strength
        if context.regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
            confidence += 0.1
        elif context.regime.regime in [MarketRegime.WEAK_UPTREND, MarketRegime.WEAK_DOWNTREND]:
            confidence += 0.05

        # Adjust for trend strength
        confidence += context.regime.trend_strength * 0.1

        # Adjust for regime confidence
        confidence += (context.regime.confidence - 0.5) * 0.1

        # Adjust for ML prediction
        if ml_prediction == "FAVORABLE":
            confidence += 0.1
        elif ml_prediction == "UNFAVORABLE":
            confidence -= 0.15

        # Adjust for sentiment alignment
        if context.regime.regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            if context.sentiment.overall_sentiment == "bullish":
                confidence += 0.05
            elif context.sentiment.overall_sentiment == "bearish":
                confidence -= 0.05
        else:
            if context.sentiment.overall_sentiment == "bearish":
                confidence += 0.05
            elif context.sentiment.overall_sentiment == "bullish":
                confidence -= 0.05

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def _generate_warnings(
        self,
        context: MarketContext,
        ml_prediction: str,
    ) -> list[str]:
        """
        Generate warnings for the signal.

        Args:
            context: Market context
            ml_prediction: ML prediction result

        Returns:
            List of warning strings
        """
        warnings: list[str] = []

        # ML warning
        if ml_prediction == "UNFAVORABLE":
            warnings.append("ML predicts unfavorable outcome")

        # Volatility warning
        if context.regime.volatility_level == "extreme":
            warnings.append("Extreme volatility - wider stops recommended")
        elif context.regime.volatility_level == "high":
            warnings.append("High volatility environment")

        # Funding rate warning
        if abs(context.funding.current_rate) > 0.001:
            if context.funding.current_rate > 0:
                warnings.append(f"High positive funding: {context.funding.current_rate*100:.3f}%")
            else:
                warnings.append(f"High negative funding: {context.funding.current_rate*100:.3f}%")

        # Volume warning
        if context.volume.relative_volume < 0.5:
            warnings.append("Low volume - potential slippage")
        elif context.volume.volume_spike:
            warnings.append("Volume spike detected - monitor closely")

        # Sentiment divergence warning
        regime = context.regime.regime
        if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            if context.sentiment.overall_sentiment == "bearish":
                warnings.append("Bearish sentiment divergence")
        elif regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
            if context.sentiment.overall_sentiment == "bullish":
                warnings.append("Bullish sentiment divergence")

        # Fear & Greed warning
        fg = context.sentiment.fear_greed_index
        if fg < 20:
            warnings.append(f"Extreme Fear ({fg}) - potential capitulation")
        elif fg > 80:
            warnings.append(f"Extreme Greed ({fg}) - potential top")

        return warnings
