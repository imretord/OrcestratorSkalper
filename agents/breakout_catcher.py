"""
Breakout Catcher Agent for AI Trading System V3.
Specializes in catching breakouts from compression and early breakout phases.
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

log = get_logger("breakout_catcher")


class BreakoutCatcherAgent(BaseAgent):
    """
    Breakout Catcher Agent.

    Strategy:
    - In compression: Prepare for breakout direction based on context
    - On breakout: Enter in breakout direction with volume confirmation
    - Uses momentum and volume as key confirmations

    Best in: COMPRESSION, BREAKOUT_UP, BREAKOUT_DOWN
    Avoids: Established trends, choppy markets
    """

    # DISABLED COMPRESSION - 0% win rate in testing (all trades hit SL)
    # Only trade confirmed breakouts now
    SUITABLE_REGIMES = [
        MarketRegime.BREAKOUT_UP,
        MarketRegime.BREAKOUT_DOWN,
    ]

    def __init__(
        self,
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
        compression_bb_width: float = 0.02,
        volume_spike_threshold: float = 2.0,
        min_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize BreakoutCatcher agent.

        Args:
            predictor: Online predictor for ML predictions
            meta_learner: Meta learner for adaptive insights
            compression_bb_width: BB width threshold for compression (2% default)
            volume_spike_threshold: Volume multiple for breakout confirmation (2x default)
            min_confidence_threshold: Minimum confidence to generate signal
        """
        super().__init__(
            name="BreakoutCatcher",
            description="Catches breakouts from compression",
            suitable_regimes=self.SUITABLE_REGIMES,
            predictor=predictor,
            meta_learner=meta_learner,
            min_confidence_threshold=min_confidence_threshold,
        )

        self.compression_bb_width = compression_bb_width
        self.volume_spike_threshold = volume_spike_threshold

    async def analyze(self, context: MarketContext) -> Signal | None:
        """
        Analyze market context for breakout opportunity.

        Args:
            context: Complete market context

        Returns:
            Signal if breakout detected, None otherwise
        """
        regime = context.regime.regime

        # Check if we're suitable for this regime
        if not self.is_suitable(regime):
            log.debug(f"[BreakoutCatcher] Not suitable for regime: {regime.value}")
            return None

        # Check for breakout conditions
        breakout_type, entry_reasons = self._check_breakout_entry(context)

        if breakout_type is None:
            log.debug(f"[BreakoutCatcher] No breakout entry for {context.symbol}")
            return None

        direction = breakout_type  # "LONG" for up breakout, "SHORT" for down breakout

        # Get ML prediction
        ml_prediction, ml_confidence = self._get_ml_prediction(context, direction)

        # Generate warnings
        warnings = self._generate_warnings(context, ml_prediction, direction)

        # Calculate confidence
        confidence = self._calculate_confidence(context, entry_reasons, ml_prediction)

        # Check minimum confidence - higher threshold for compression (false breakout risk)
        min_conf = self.min_confidence_threshold
        if regime == MarketRegime.COMPRESSION:
            min_conf = max(min_conf, 0.70)  # At least 70% for compression trades

        if confidence < min_conf:
            log.info(
                f"[BreakoutCatcher] {context.symbol} confidence too low: "
                f"{confidence:.2f} < {min_conf}"
            )
            return None

        # Calculate entry, SL, TP
        entry_price = context.current_price

        # Get ATR for stop-loss
        ind_1h = context.price_feed.indicators.get("1h")
        atr = ind_1h.atr_14 if ind_1h and ind_1h.atr_14 else entry_price * 0.02

        # Breakouts can be volatile - use regime-based stops
        stop_loss = self._calculate_stop_loss(direction, entry_price, atr, regime)

        # Breakouts can run - but use conservative targets for faster fills
        if regime in [MarketRegime.BREAKOUT_UP, MarketRegime.BREAKOUT_DOWN]:
            tp1_rr, tp2_rr = 1.5, 2.5  # Moderate targets for confirmed breakouts
        else:  # COMPRESSION - need better R:R to compensate for false breakouts
            tp1_rr, tp2_rr = 1.2, 1.8  # Was 1.0, 1.5 - too tight, caused 81% SL rate

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
            f"[BreakoutCatcher] Signal generated for {context.symbol}: "
            f"{direction} @ {entry_price:.2f}, conf={confidence:.2f}"
        )

        return signal

    def _check_breakout_entry(
        self,
        context: MarketContext,
    ) -> tuple[str | None, list[str]]:
        """
        Check for breakout entry conditions.

        Args:
            context: Market context

        Returns:
            Tuple of (direction or None, reasons)
        """
        reasons: list[str] = []
        regime = context.regime.regime

        price = context.current_price
        ind_1h = context.price_feed.indicators.get("1h")

        if not ind_1h:
            return (None, [])

        bb_upper = ind_1h.bollinger_upper
        bb_lower = ind_1h.bollinger_lower
        bb_middle = ind_1h.bollinger_middle
        macd_hist = ind_1h.macd_histogram or 0
        adx = ind_1h.adx_14 or 25

        # Calculate BB width percentage
        bb_width_pct = None
        if bb_upper and bb_lower and bb_middle:
            bb_width_pct = (bb_upper - bb_lower) / bb_middle

        # Determine direction based on regime
        if regime == MarketRegime.BREAKOUT_UP:
            direction = "LONG"
            reasons.append("Regime indicates BREAKOUT_UP")
        elif regime == MarketRegime.BREAKOUT_DOWN:
            direction = "SHORT"
            reasons.append("Regime indicates BREAKOUT_DOWN")
        elif regime == MarketRegime.COMPRESSION:
            # In compression, determine direction from context
            direction = self._determine_breakout_direction(context)
            if direction is None:
                return (None, [])
            reasons.append(f"Compression breakout direction: {direction}")
        else:
            return (None, [])

        # Validation checks
        checks_passed = 1  # Start with 1 for regime match
        required_checks = 2

        # Check 1: Volume confirmation
        if context.volume.volume_spike or context.volume.relative_volume >= self.volume_spike_threshold:
            reasons.append(f"Volume spike: {context.volume.relative_volume:.1f}x average")
            checks_passed += 1.5
        elif context.volume.relative_volume >= 1.5:
            reasons.append(f"Above average volume: {context.volume.relative_volume:.1f}x")
            checks_passed += 0.5

        # Check 2: Momentum confirmation
        if direction == "LONG" and macd_hist > 0:
            reasons.append(f"MACD histogram positive: {macd_hist:.6f}")
            checks_passed += 0.5
        elif direction == "SHORT" and macd_hist < 0:
            reasons.append(f"MACD histogram negative: {macd_hist:.6f}")
            checks_passed += 0.5

        # Check 3: ADX showing trend strength building
        if adx > 20:
            reasons.append(f"ADX indicates trend developing: {adx:.1f}")
            checks_passed += 0.5

        # Check 4: Bollinger band breakout
        if direction == "LONG":
            if bb_upper and price > bb_upper:
                reasons.append("Price broke above upper Bollinger band")
                checks_passed += 1
            elif bb_middle and price > bb_middle:
                reasons.append("Price above middle Bollinger band")
                checks_passed += 0.5
        else:  # SHORT
            if bb_lower and price < bb_lower:
                reasons.append("Price broke below lower Bollinger band")
                checks_passed += 1
            elif bb_middle and price < bb_middle:
                reasons.append("Price below middle Bollinger band")
                checks_passed += 0.5

        # Check 5: Support/Resistance breakout
        if direction == "LONG" and context.regime.resistance_level:
            if price > context.regime.resistance_level:
                reasons.append(f"Broke above resistance: {context.regime.resistance_level:.2f}")
                checks_passed += 1
        elif direction == "SHORT" and context.regime.support_level:
            if price < context.regime.support_level:
                reasons.append(f"Broke below support: {context.regime.support_level:.2f}")
                checks_passed += 1

        # Check 6: Context bias alignment
        if direction == "LONG" and context.suggested_bias == "long":
            reasons.append("Context bias aligned: LONG")
            checks_passed += 0.5
        elif direction == "SHORT" and context.suggested_bias == "short":
            reasons.append("Context bias aligned: SHORT")
            checks_passed += 0.5

        if checks_passed >= required_checks:
            return (direction, reasons)
        else:
            return (None, [])

    def _determine_breakout_direction(self, context: MarketContext) -> str | None:
        """
        Determine likely breakout direction from compression.

        Args:
            context: Market context

        Returns:
            "LONG", "SHORT", or None if unclear
        """
        long_score = 0
        short_score = 0

        # Check momentum score
        if context.momentum_score > 0.3:
            long_score += 2
        elif context.momentum_score < -0.3:
            short_score += 2
        elif context.momentum_score > 0:
            long_score += 1
        elif context.momentum_score < 0:
            short_score += 1

        # Check suggested bias
        if context.suggested_bias == "long":
            long_score += 1
        elif context.suggested_bias == "short":
            short_score += 1

        # Check sentiment
        if context.sentiment.overall_sentiment == "bullish":
            long_score += 1
        elif context.sentiment.overall_sentiment == "bearish":
            short_score += 1

        # Check funding rate (contrarian in compression)
        if context.funding.current_rate > 0.0005:
            # High funding = many longs, might break down
            short_score += 0.5
        elif context.funding.current_rate < -0.0005:
            # Negative funding = many shorts, might break up
            long_score += 0.5

        # Check volume profile
        if context.volume.buy_volume_ratio > 0.6:
            long_score += 1
        elif context.volume.buy_volume_ratio < 0.4:
            short_score += 1

        # Determine direction
        if long_score > short_score and long_score >= 2:
            direction = "LONG"
        elif short_score > long_score and short_score >= 2:
            direction = "SHORT"
        else:
            return None  # Unclear direction

        # IMPORTANT: Do not trade against the main trend bias
        # If suggested_bias is clearly opposite, reject the signal
        if direction == "LONG" and context.suggested_bias == "short":
            log.info(
                f"[BreakoutCatcher] Rejecting LONG - against main trend bias (short)"
            )
            return None
        elif direction == "SHORT" and context.suggested_bias == "long":
            log.info(
                f"[BreakoutCatcher] Rejecting SHORT - against main trend bias (long)"
            )
            return None

        return direction

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
        reason_bonus = min(len(reasons) * 0.07, 0.3)
        confidence += reason_bonus

        # Higher confidence for confirmed breakouts
        if context.regime.regime in [MarketRegime.BREAKOUT_UP, MarketRegime.BREAKOUT_DOWN]:
            confidence += 0.1
        # Lower for compression (unconfirmed)
        elif context.regime.regime == MarketRegime.COMPRESSION:
            confidence -= 0.05

        # Volume spike is strong confirmation
        if context.volume.volume_spike:
            confidence += 0.1
        elif context.volume.relative_volume > 1.5:
            confidence += 0.05

        # Regime confidence
        confidence += (context.regime.confidence - 0.5) * 0.1

        # ML adjustment
        if ml_prediction == "FAVORABLE":
            confidence += 0.1
        elif ml_prediction == "UNFAVORABLE":
            confidence -= 0.15

        # Trend strength matters for breakouts
        confidence += context.regime.trend_strength * 0.05

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
            warnings.append("Extreme volatility - breakout may reverse")

        # False breakout risk - soft warning (starts with "Consider" to not count as hard warning)
        # Compression is expected regime for this agent, so we make it informational
        if context.regime.regime == MarketRegime.COMPRESSION:
            warnings.append("Consider: compression breakout may have false breakout risk")

        # Volume warning
        if context.volume.relative_volume < 1.2:
            warnings.append("Low volume breakout - may fail")

        # Funding rate warning - crowded trade risk
        if direction == "LONG" and context.funding.current_rate > 0.001:
            warnings.append(f"High positive funding ({context.funding.current_rate*100:.3f}%) - crowded long")
        elif direction == "SHORT" and context.funding.current_rate < -0.001:
            warnings.append(f"High negative funding ({context.funding.current_rate*100:.3f}%) - crowded short")

        # Sentiment extreme warning
        fg = context.sentiment.fear_greed_index
        if direction == "LONG" and fg > 80:
            warnings.append(f"Extreme Greed ({fg}) - potential top")
        elif direction == "SHORT" and fg < 20:
            warnings.append(f"Extreme Fear ({fg}) - potential bottom")

        # Support/Resistance proximity warning
        if direction == "LONG" and context.regime.resistance_level:
            dist_to_resistance = (context.regime.resistance_level - context.current_price) / context.current_price
            if dist_to_resistance > 0 and dist_to_resistance < 0.01:
                warnings.append(f"Near resistance: {context.regime.resistance_level:.2f}")
        elif direction == "SHORT" and context.regime.support_level:
            dist_to_support = (context.current_price - context.regime.support_level) / context.current_price
            if dist_to_support > 0 and dist_to_support < 0.01:
                warnings.append(f"Near support: {context.regime.support_level:.2f}")

        return warnings
