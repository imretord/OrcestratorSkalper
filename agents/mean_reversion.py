"""
Mean Reversion Agent for AI Trading System V3.
Specializes in ranging/choppy markets - fades extremes.
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

log = get_logger("mean_reversion")


class MeanReversionAgent(BaseAgent):
    """
    Mean Reversion Agent.

    Strategy:
    - In ranging markets: Buy at Bollinger lower band with RSI < 30
    - In ranging markets: Sell at Bollinger upper band with RSI > 70
    - Uses tight stops and quick targets

    Best in: RANGING, CHOPPY
    Avoids: Strong trends, breakouts
    """

    SUITABLE_REGIMES = [
        MarketRegime.RANGING,
        MarketRegime.CHOPPY,
    ]

    def __init__(
        self,
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
        bb_entry_std: float = 2.0,
        rsi_extreme_threshold: float = 35,  # Было 30 - смягчено для большего количества сигналов
        min_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize MeanReversion agent.

        Args:
            predictor: Online predictor for ML predictions
            meta_learner: Meta learner for adaptive insights
            bb_entry_std: Bollinger band standard deviations for entry (default 2.0)
            rsi_extreme_threshold: RSI threshold for extreme (< threshold = oversold, > 100-threshold = overbought)
            min_confidence_threshold: Minimum confidence to generate signal
        """
        super().__init__(
            name="MeanReversion",
            description="Fades extremes in ranging/choppy markets",
            suitable_regimes=self.SUITABLE_REGIMES,
            predictor=predictor,
            meta_learner=meta_learner,
            min_confidence_threshold=min_confidence_threshold,
        )

        self.bb_entry_std = bb_entry_std
        self.rsi_extreme_threshold = rsi_extreme_threshold

    async def analyze(self, context: MarketContext) -> Signal | None:
        """
        Analyze market context for mean reversion opportunity.

        Args:
            context: Complete market context

        Returns:
            Signal if extreme found, None otherwise
        """
        regime = context.regime.regime

        # Check if we're suitable for this regime
        if not self.is_suitable(regime):
            log.debug(f"[MeanReversion] Not suitable for regime: {regime.value}")
            return None

        # Check for extreme conditions
        entry_type, entry_reasons = self._check_extreme_entry(context)

        if entry_type is None:
            log.debug(f"[MeanReversion] No extreme entry for {context.symbol}")
            return None

        direction = entry_type  # "LONG" at lower extreme, "SHORT" at upper extreme

        # Get ML prediction
        ml_prediction, ml_confidence = self._get_ml_prediction(context, direction)

        # Generate warnings
        warnings = self._generate_warnings(context, ml_prediction, direction)

        # Calculate confidence
        confidence = self._calculate_confidence(context, entry_reasons, ml_prediction)

        # Check minimum confidence
        if confidence < self.min_confidence_threshold:
            log.info(
                f"[MeanReversion] {context.symbol} confidence too low: "
                f"{confidence:.2f} < {self.min_confidence_threshold}"
            )
            return None

        # Calculate entry, SL, TP
        entry_price = context.current_price

        # Get ATR for stop-loss (tighter stops in mean reversion)
        ind_1h = context.price_feed.indicators.get("1h")
        atr = ind_1h.atr_14 if ind_1h and ind_1h.atr_14 else entry_price * 0.015

        # Mean reversion uses tighter stops
        stop_loss = self._calculate_stop_loss(direction, entry_price, atr, regime)

        # Quick targets for mean reversion (aim for middle BB)
        tp1_rr, tp2_rr = 1.0, 1.5  # Conservative targets
        tp1, tp2 = self._calculate_take_profits(direction, entry_price, stop_loss, tp1_rr, tp2_rr)

        # Optionally adjust TP1 to middle Bollinger band
        bb_middle = ind_1h.bollinger_middle if ind_1h else None
        if bb_middle:
            if direction == "LONG" and bb_middle > entry_price:
                # TP1 at middle band if reasonable
                if abs(bb_middle - entry_price) < abs(tp1 - entry_price):
                    tp1 = bb_middle
            elif direction == "SHORT" and bb_middle < entry_price:
                if abs(bb_middle - entry_price) < abs(tp1 - entry_price):
                    tp1 = bb_middle

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
            f"[MeanReversion] Signal generated for {context.symbol}: "
            f"{direction} @ {entry_price:.2f}, conf={confidence:.2f}"
        )

        return signal

    def _check_extreme_entry(
        self,
        context: MarketContext,
    ) -> tuple[str | None, list[str]]:
        """
        Check for extreme entry conditions (oversold/overbought).

        Args:
            context: Market context

        Returns:
            Tuple of (direction or None, reasons)
        """
        reasons: list[str] = []

        price = context.current_price
        ind_1h = context.price_feed.indicators.get("1h")
        ind_15m = context.price_feed.indicators.get("15m")

        if not ind_1h:
            return (None, [])

        rsi = ind_1h.rsi_14 or 50
        bb_upper = ind_1h.bollinger_upper
        bb_lower = ind_1h.bollinger_lower
        bb_middle = ind_1h.bollinger_middle

        # Calculate BB position
        bb_position = None
        if bb_upper and bb_lower and bb_middle:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_position = (price - bb_lower) / bb_range  # 0 = lower, 1 = upper

        # Check for LONG (oversold extreme)
        long_score = 0
        long_reasons: list[str] = []

        if rsi < self.rsi_extreme_threshold:
            long_reasons.append(f"RSI oversold: {rsi:.1f} < {self.rsi_extreme_threshold}")
            long_score += 2

        if bb_position is not None and bb_position < 0.1:
            long_reasons.append(f"Price at lower Bollinger band ({bb_position*100:.1f}%)")
            long_score += 2
        elif bb_lower and price <= bb_lower:
            long_reasons.append(f"Price below lower Bollinger band")
            long_score += 2

        # Additional confirmations for LONG
        if context.sentiment.sentiment_score < -0.5:
            long_reasons.append(f"Extreme negative sentiment: {context.sentiment.sentiment_score:.2f}")
            long_score += 1

        if context.volume.relative_volume > 1.5:
            long_reasons.append(f"High volume at extreme: {context.volume.relative_volume:.1f}x")
            long_score += 1

        # Alternative: Bollinger Band touch without extreme RSI
        # If price at BB edge, can enter with moderate RSI (not necessarily < 35)
        if bb_position is not None and bb_position < 0.15 and rsi < 45:
            if long_score < 2:  # Only add if not already qualified
                long_reasons.append(f"Price at lower BB ({bb_position*100:.1f}%) with supportive RSI ({rsi:.1f})")
                long_score += 2

        # Check for SHORT (overbought extreme)
        short_score = 0
        short_reasons: list[str] = []

        if rsi > (100 - self.rsi_extreme_threshold):
            short_reasons.append(f"RSI overbought: {rsi:.1f} > {100 - self.rsi_extreme_threshold}")
            short_score += 2

        if bb_position is not None and bb_position > 0.9:
            short_reasons.append(f"Price at upper Bollinger band ({bb_position*100:.1f}%)")
            short_score += 2
        elif bb_upper and price >= bb_upper:
            short_reasons.append(f"Price above upper Bollinger band")
            short_score += 2

        # Additional confirmations for SHORT
        if context.sentiment.sentiment_score > 0.5:
            short_reasons.append(f"Extreme positive sentiment: {context.sentiment.sentiment_score:.2f}")
            short_score += 1

        if context.volume.relative_volume > 1.5:
            short_reasons.append(f"High volume at extreme: {context.volume.relative_volume:.1f}x")
            short_score += 1

        # Alternative: Bollinger Band touch without extreme RSI
        # If price at BB edge, can enter with moderate RSI (not necessarily > 65)
        if bb_position is not None and bb_position > 0.85 and rsi > 55:
            if short_score < 2:  # Only add if not already qualified
                short_reasons.append(f"Price at upper BB ({bb_position*100:.1f}%) with supportive RSI ({rsi:.1f})")
                short_score += 2

        # Determine which extreme is stronger
        min_score = 2  # Было 3 - снижено для большего количества сигналов

        if long_score >= min_score and long_score > short_score:
            return ("LONG", long_reasons)
        elif short_score >= min_score and short_score > long_score:
            return ("SHORT", short_reasons)
        else:
            return (None, [])

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

        # Add confidence for each reason
        reason_bonus = min(len(reasons) * 0.08, 0.25)
        confidence += reason_bonus

        # Mean reversion works better in ranging
        if context.regime.regime == MarketRegime.RANGING:
            confidence += 0.1
        elif context.regime.regime == MarketRegime.CHOPPY:
            confidence += 0.05

        # Regime confidence bonus
        confidence += (context.regime.confidence - 0.5) * 0.1

        # LOW trend strength is good for mean reversion
        confidence += (1.0 - context.regime.trend_strength) * 0.1

        # ML adjustment
        if ml_prediction == "FAVORABLE":
            confidence += 0.1
        elif ml_prediction == "UNFAVORABLE":
            confidence -= 0.15

        # Volume at extreme is good confirmation
        if context.volume.relative_volume > 1.5:
            confidence += 0.05

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))

    def _generate_warnings(
        self,
        context: MarketContext,
        ml_prediction: str,
        direction: str,
    ) -> list[str]:
        """
        Generate warnings for the signal.

        Args:
            context: Market context
            ml_prediction: ML prediction result
            direction: Trade direction

        Returns:
            List of warning strings
        """
        warnings: list[str] = []

        # ML warning
        if ml_prediction == "UNFAVORABLE":
            warnings.append("ML predicts unfavorable outcome")

        # Volatility warning
        if context.regime.volatility_level == "extreme":
            warnings.append("Extreme volatility - mean reversion risky")
        elif context.regime.volatility_level == "high":
            warnings.append("High volatility - use caution")

        # Trend warning - mean reversion against strong trend is dangerous
        if context.regime.trend_strength > 0.7:
            warnings.append(f"Strong trend detected ({context.regime.trend_strength:.2f}) - potential breakout")

        # Funding rate warning
        if direction == "LONG" and context.funding.current_rate < -0.001:
            warnings.append(f"Negative funding favors shorts")
        elif direction == "SHORT" and context.funding.current_rate > 0.001:
            warnings.append(f"Positive funding favors longs")

        # Volume warning
        if context.volume.relative_volume < 0.5:
            warnings.append("Low volume - potential slippage")

        # Fear & Greed extremes - might indicate trend continuation
        fg = context.sentiment.fear_greed_index
        if fg < 10 and direction == "LONG":
            warnings.append(f"Extreme Fear ({fg}) - capitulation possible")
        elif fg > 90 and direction == "SHORT":
            warnings.append(f"Extreme Greed ({fg}) - blow-off top possible")

        # Breakout potential warning
        if context.regime.volatility_level == "low":
            warnings.append("Low volatility compression - breakout risk")

        return warnings
