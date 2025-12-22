"""
Market regime detector for AI Trading System V3.
Detects market regimes: trending, ranging, choppy, compression, breakout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from analyzers.base_analyzer import CachedAnalyzer
from core.logger import get_logger
from core.state import (
    MarketRegime,
    RegimeAnalysis,
    SensorSnapshot,
    IndicatorValues,
    OHLCVBar,
)

log = get_logger("regime_detector")


class RegimeDetector(CachedAnalyzer[RegimeAnalysis]):
    """
    Detects current market regime using multiple indicators.

    Regimes:
    - STRONG_UPTREND: Strong bullish momentum (ADX > 25, price above EMAs, RSI > 60)
    - WEAK_UPTREND: Moderate bullish bias (ADX 15-25, price above EMA50)
    - STRONG_DOWNTREND: Strong bearish momentum (ADX > 25, price below EMAs, RSI < 40)
    - WEAK_DOWNTREND: Moderate bearish bias (ADX 15-25, price below EMA50)
    - RANGING: Low volatility sideways movement (ADX < 20, price between Bollinger bands)
    - CHOPPY: High volatility without direction (ADX < 20, wide Bollinger bands)
    - COMPRESSION: Tightening volatility (narrowing Bollinger bands, decreasing ATR)
    - BREAKOUT_UP: Breaking out of compression/range upward
    - BREAKOUT_DOWN: Breaking out of compression/range downward
    """

    # Regime detection thresholds
    ADX_STRONG_TREND = 25.0
    ADX_WEAK_TREND = 15.0
    RSI_OVERBOUGHT = 70.0
    RSI_OVERSOLD = 30.0
    RSI_BULLISH = 60.0
    RSI_BEARISH = 40.0
    BB_COMPRESSION_PCT = 0.02  # Bollinger width < 2% of price
    VOLUME_SPIKE = 2.0  # 2x average volume for breakout confirmation

    def __init__(self, cache_seconds: int = 60) -> None:
        """Initialize regime detector."""
        super().__init__("RegimeDetector", cache_seconds)
        self._regime_history: dict[str, list[tuple[MarketRegime, datetime]]] = {}

    def analyze(self, snapshot: SensorSnapshot) -> RegimeAnalysis:
        """
        Analyze market regime from sensor snapshot.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            RegimeAnalysis with detected regime and confidence
        """
        symbol = snapshot.symbol
        price = snapshot.price_feed.current_price
        indicators = snapshot.price_feed.indicators

        # Get indicators from multiple timeframes
        ind_1h = indicators.get("1h", IndicatorValues())
        ind_15m = indicators.get("15m", IndicatorValues())
        ind_4h = indicators.get("4h", IndicatorValues())

        # Calculate regime components
        trend_direction, trend_strength = self._analyze_trend(price, ind_1h, ind_4h)
        volatility_level, volatility_score = self._analyze_volatility(price, ind_1h, ind_15m)
        momentum_bias = self._analyze_momentum(ind_1h, ind_15m)

        # Detect support/resistance
        ohlcv_1h = snapshot.price_feed.ohlcv_1h
        support, resistance = self._find_support_resistance(ohlcv_1h, price)

        # Check for compression (narrowing Bollinger bands)
        is_compression = self._detect_compression(ind_1h, ind_15m)

        # Check for breakout
        breakout_direction = self._detect_breakout(
            price, support, resistance, ind_1h, snapshot.volume.relative_volume
        )

        # Determine regime
        regime, confidence = self._classify_regime(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            momentum_bias=momentum_bias,
            is_compression=is_compression,
            breakout_direction=breakout_direction,
            adx=ind_1h.adx_14,
        )

        # Calculate regime duration
        duration_hours = self._get_regime_duration(symbol, regime)

        # Generate description
        description = self._generate_description(
            regime, trend_strength, volatility_level, momentum_bias, confidence
        )

        return RegimeAnalysis(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            regime_duration_hours=duration_hours,
            support_level=support,
            resistance_level=resistance,
            regime_description=description,
            timestamp=datetime.now(timezone.utc),
        )

    def _analyze_trend(
        self,
        price: float,
        ind_1h: IndicatorValues,
        ind_4h: IndicatorValues,
    ) -> tuple[str, float]:
        """
        Analyze trend direction and strength.

        Returns:
            Tuple of (direction: "up"/"down"/"neutral", strength: 0-1)
        """
        scores = []

        # EMA alignment check (1h)
        if ind_1h.ema_20 and ind_1h.ema_50 and ind_1h.ema_200:
            if price > ind_1h.ema_20 > ind_1h.ema_50 > ind_1h.ema_200:
                scores.append(1.0)  # Perfect bullish alignment
            elif price < ind_1h.ema_20 < ind_1h.ema_50 < ind_1h.ema_200:
                scores.append(-1.0)  # Perfect bearish alignment
            elif price > ind_1h.ema_50:
                scores.append(0.5)  # Moderate bullish
            elif price < ind_1h.ema_50:
                scores.append(-0.5)  # Moderate bearish
            else:
                scores.append(0.0)

        # MACD trend (1h)
        if ind_1h.macd_histogram is not None:
            macd_norm = np.clip(ind_1h.macd_histogram / (price * 0.001), -1, 1)
            scores.append(macd_norm)

        # ADX-based trend strength (1h)
        adx_strength = 0.0
        if ind_1h.adx_14:
            adx_strength = min(ind_1h.adx_14 / 50, 1.0)  # Normalize to 0-1

        # 4h trend confirmation
        if ind_4h.ema_20 and ind_4h.ema_50:
            if price > ind_4h.ema_20 > ind_4h.ema_50:
                scores.append(0.8)
            elif price < ind_4h.ema_20 < ind_4h.ema_50:
                scores.append(-0.8)
            else:
                scores.append(0.0)

        if not scores:
            return "neutral", 0.0

        avg_score = np.mean(scores)

        if avg_score > 0.3:
            direction = "up"
        elif avg_score < -0.3:
            direction = "down"
        else:
            direction = "neutral"

        # Combine direction score with ADX strength
        strength = min(abs(avg_score) * (0.5 + 0.5 * adx_strength), 1.0)

        return direction, strength

    def _analyze_volatility(
        self,
        price: float,
        ind_1h: IndicatorValues,
        ind_15m: IndicatorValues,
    ) -> tuple[str, float]:
        """
        Analyze volatility level.

        Returns:
            Tuple of (level: "low"/"medium"/"high"/"extreme", score: 0-1)
        """
        volatility_measures = []

        # ATR-based volatility (1h)
        if ind_1h.atr_14 and price > 0:
            atr_pct = (ind_1h.atr_14 / price) * 100
            volatility_measures.append(atr_pct)

        # Bollinger band width (1h)
        if ind_1h.bollinger_upper and ind_1h.bollinger_lower and ind_1h.bollinger_middle:
            bb_width_pct = ((ind_1h.bollinger_upper - ind_1h.bollinger_lower) /
                           ind_1h.bollinger_middle) * 100
            volatility_measures.append(bb_width_pct)

        # 15m ATR for short-term volatility
        if ind_15m.atr_14 and price > 0:
            atr_15m_pct = (ind_15m.atr_14 / price) * 100
            volatility_measures.append(atr_15m_pct * 0.8)  # Weight less

        if not volatility_measures:
            return "medium", 0.5

        avg_volatility = np.mean(volatility_measures)

        # Classify volatility level
        if avg_volatility < 1.0:
            level = "low"
            score = avg_volatility / 1.0 * 0.25
        elif avg_volatility < 2.5:
            level = "medium"
            score = 0.25 + (avg_volatility - 1.0) / 1.5 * 0.25
        elif avg_volatility < 5.0:
            level = "high"
            score = 0.5 + (avg_volatility - 2.5) / 2.5 * 0.25
        else:
            level = "extreme"
            score = 0.75 + min((avg_volatility - 5.0) / 5.0 * 0.25, 0.25)

        return level, min(score, 1.0)

    def _analyze_momentum(
        self,
        ind_1h: IndicatorValues,
        ind_15m: IndicatorValues,
    ) -> float:
        """
        Analyze momentum bias.

        Returns:
            Momentum score from -1 (bearish) to +1 (bullish)
        """
        scores = []

        # RSI momentum (1h)
        if ind_1h.rsi_14:
            rsi = ind_1h.rsi_14
            if rsi > self.RSI_OVERBOUGHT:
                scores.append(0.8)  # Strong bullish but possibly exhausted
            elif rsi > self.RSI_BULLISH:
                scores.append(0.6)
            elif rsi < self.RSI_OVERSOLD:
                scores.append(-0.8)  # Strong bearish but possibly oversold
            elif rsi < self.RSI_BEARISH:
                scores.append(-0.6)
            else:
                # Neutral zone - slight bias based on which side of 50
                scores.append((rsi - 50) / 50 * 0.3)

        # MACD momentum (1h)
        if ind_1h.macd_histogram is not None:
            if ind_1h.macd_histogram > 0:
                scores.append(0.5)
            else:
                scores.append(-0.5)

        # RSI momentum (15m) for short-term
        if ind_15m.rsi_14:
            rsi_15m = ind_15m.rsi_14
            short_term_bias = (rsi_15m - 50) / 50 * 0.4
            scores.append(short_term_bias)

        if not scores:
            return 0.0

        return np.clip(np.mean(scores), -1, 1)

    def _find_support_resistance(
        self,
        ohlcv: list[OHLCVBar],
        current_price: float,
    ) -> tuple[float | None, float | None]:
        """
        Find nearest support and resistance levels.

        Uses swing high/low detection from recent price action.
        """
        if len(ohlcv) < 20:
            return None, None

        # Get recent highs and lows
        highs = [bar.high for bar in ohlcv[-50:]] if len(ohlcv) >= 50 else [bar.high for bar in ohlcv]
        lows = [bar.low for bar in ohlcv[-50:]] if len(ohlcv) >= 50 else [bar.low for bar in ohlcv]

        # Find swing highs (local maxima)
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                if highs[i] > current_price:
                    resistance_levels.append(highs[i])

        # Find swing lows (local minima)
        support_levels = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                if lows[i] < current_price:
                    support_levels.append(lows[i])

        # Get nearest levels
        support = max(support_levels) if support_levels else min(lows)
        resistance = min(resistance_levels) if resistance_levels else max(highs)

        return support, resistance

    def _detect_compression(
        self,
        ind_1h: IndicatorValues,
        ind_15m: IndicatorValues,
    ) -> bool:
        """
        Detect if market is in compression (narrowing volatility).

        Returns:
            True if compression detected
        """
        if not (ind_1h.bollinger_upper and ind_1h.bollinger_lower and ind_1h.bollinger_middle):
            return False

        # Check Bollinger band squeeze
        bb_width = (ind_1h.bollinger_upper - ind_1h.bollinger_lower) / ind_1h.bollinger_middle

        # Compression when BB width is less than 2%
        if bb_width < self.BB_COMPRESSION_PCT:
            return True

        # Also check if 15m BB is compressing
        if ind_15m.bollinger_upper and ind_15m.bollinger_lower and ind_15m.bollinger_middle:
            bb_width_15m = (ind_15m.bollinger_upper - ind_15m.bollinger_lower) / ind_15m.bollinger_middle
            if bb_width_15m < self.BB_COMPRESSION_PCT * 0.8:  # Tighter threshold for 15m
                return True

        return False

    def _detect_breakout(
        self,
        price: float,
        support: float | None,
        resistance: float | None,
        ind_1h: IndicatorValues,
        relative_volume: float,
    ) -> str | None:
        """
        Detect breakout direction.

        Returns:
            "up", "down", or None
        """
        # Breakout requires volume confirmation
        if relative_volume < self.VOLUME_SPIKE:
            return None

        # Check Bollinger band breakout
        if ind_1h.bollinger_upper and ind_1h.bollinger_lower:
            if price > ind_1h.bollinger_upper:
                return "up"
            if price < ind_1h.bollinger_lower:
                return "down"

        # Check support/resistance breakout
        if resistance and price > resistance * 1.005:  # 0.5% above resistance
            return "up"
        if support and price < support * 0.995:  # 0.5% below support
            return "down"

        return None

    def _classify_regime(
        self,
        trend_direction: str,
        trend_strength: float,
        volatility_level: str,
        momentum_bias: float,
        is_compression: bool,
        breakout_direction: str | None,
        adx: float | None,
    ) -> tuple[MarketRegime, float]:
        """
        Classify market regime based on all factors.

        Returns:
            Tuple of (regime, confidence)
        """
        adx = adx or 20.0  # Default if not available

        # Priority 1: Breakout detection
        if breakout_direction == "up":
            confidence = 0.7 + trend_strength * 0.3
            return MarketRegime.BREAKOUT_UP, confidence

        if breakout_direction == "down":
            confidence = 0.7 + trend_strength * 0.3
            return MarketRegime.BREAKOUT_DOWN, confidence

        # Priority 2: Compression detection
        if is_compression:
            confidence = 0.6 + (1 - trend_strength) * 0.2  # Higher confidence if trend is weak
            return MarketRegime.COMPRESSION, confidence

        # Priority 3: Strong trends
        if adx > self.ADX_STRONG_TREND:
            if trend_direction == "up" and momentum_bias > 0.3:
                confidence = 0.5 + trend_strength * 0.3 + (adx - 25) / 50 * 0.2
                return MarketRegime.STRONG_UPTREND, min(confidence, 0.95)

            if trend_direction == "down" and momentum_bias < -0.3:
                confidence = 0.5 + trend_strength * 0.3 + (adx - 25) / 50 * 0.2
                return MarketRegime.STRONG_DOWNTREND, min(confidence, 0.95)

        # Priority 4: Weak trends
        if adx > self.ADX_WEAK_TREND:
            if trend_direction == "up":
                confidence = 0.4 + trend_strength * 0.3
                return MarketRegime.WEAK_UPTREND, confidence

            if trend_direction == "down":
                confidence = 0.4 + trend_strength * 0.3
                return MarketRegime.WEAK_DOWNTREND, confidence

        # Priority 5: Ranging vs Choppy
        if volatility_level in ("high", "extreme"):
            confidence = 0.5 + (0.1 if volatility_level == "extreme" else 0)
            return MarketRegime.CHOPPY, confidence

        # Default: Ranging
        confidence = 0.4 + (1 - trend_strength) * 0.2
        return MarketRegime.RANGING, confidence

    def _get_regime_duration(self, symbol: str, current_regime: MarketRegime) -> int:
        """
        Calculate how long the current regime has been active.

        Returns:
            Duration in hours
        """
        now = datetime.now(timezone.utc)

        # Initialize history for new symbols
        if symbol not in self._regime_history:
            self._regime_history[symbol] = []

        history = self._regime_history[symbol]

        # Add current regime to history
        if not history or history[-1][0] != current_regime:
            history.append((current_regime, now))
            # Keep only last 100 entries
            if len(history) > 100:
                history.pop(0)

        # Find when current regime started
        regime_start = now
        for regime, timestamp in reversed(history):
            if regime == current_regime:
                regime_start = timestamp
            else:
                break

        duration_hours = int((now - regime_start).total_seconds() / 3600)
        return max(duration_hours, 1)  # Minimum 1 hour

    def _generate_description(
        self,
        regime: MarketRegime,
        trend_strength: float,
        volatility_level: str,
        momentum_bias: float,
        confidence: float,
    ) -> str:
        """Generate human-readable regime description."""
        descriptions = {
            MarketRegime.STRONG_UPTREND: "Strong bullish trend with momentum confirmation",
            MarketRegime.WEAK_UPTREND: "Moderate bullish bias, trend establishing",
            MarketRegime.STRONG_DOWNTREND: "Strong bearish trend with momentum confirmation",
            MarketRegime.WEAK_DOWNTREND: "Moderate bearish bias, trend establishing",
            MarketRegime.RANGING: "Sideways consolidation, low directional movement",
            MarketRegime.CHOPPY: "High volatility without clear direction",
            MarketRegime.COMPRESSION: "Volatility squeeze, potential breakout forming",
            MarketRegime.BREAKOUT_UP: "Bullish breakout with volume confirmation",
            MarketRegime.BREAKOUT_DOWN: "Bearish breakdown with volume confirmation",
        }

        base_desc = descriptions.get(regime, "Unknown regime")

        # Add confidence qualifier
        if confidence > 0.8:
            conf_str = "High confidence"
        elif confidence > 0.6:
            conf_str = "Moderate confidence"
        else:
            conf_str = "Low confidence"

        # Add volatility note
        vol_str = f"{volatility_level} volatility"

        return f"{base_desc}. {conf_str}, {vol_str}."

    def get_regime_history(self, symbol: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent regime history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum entries to return

        Returns:
            List of regime changes with timestamps
        """
        history = self._regime_history.get(symbol, [])

        return [
            {
                "regime": regime.value,
                "timestamp": timestamp.isoformat(),
            }
            for regime, timestamp in history[-limit:]
        ]
