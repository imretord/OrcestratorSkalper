"""
Price Feed Sensor for AI Trading System V3.
Collects OHLCV data and calculates technical indicators.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from core.logger import get_logger
from core.state import IndicatorValues, OHLCVBar, PriceFeedData
from execution.binance_client import BinanceClient
from sensors.base_sensor import BaseSensor

log = get_logger("price_feed")


class PriceFeedSensor(BaseSensor):
    """
    Sensor for price data and technical indicators.

    Collects:
    - Current price
    - OHLCV candles for multiple timeframes
    - Technical indicators (RSI, MACD, EMA, ATR, ADX, Bollinger Bands)
    """

    # Timeframes to collect
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

    # Number of candles to fetch per timeframe
    CANDLE_LIMITS = {
        "1m": 60,
        "5m": 60,
        "15m": 100,
        "1h": 100,
        "4h": 100,
    }

    def __init__(
        self,
        client: BinanceClient,
        update_interval_seconds: int = 30,
    ) -> None:
        """
        Initialize PriceFeedSensor.

        Args:
            client: BinanceClient instance
            update_interval_seconds: Update interval
        """
        super().__init__(name="PriceFeed", update_interval_seconds=update_interval_seconds)
        self.client = client

    async def collect(self, symbol: str) -> PriceFeedData:
        """
        Collect price data and indicators for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            PriceFeedData object
        """
        # Get current price
        current_price = await asyncio.to_thread(
            self.client.get_current_price, symbol
        )

        # Collect OHLCV for all timeframes
        ohlcv_data: dict[str, list[OHLCVBar]] = {}
        indicators: dict[str, IndicatorValues] = {}

        for timeframe in self.TIMEFRAMES:
            limit = self.CANDLE_LIMITS.get(timeframe, 100)

            # Fetch OHLCV
            bars = await asyncio.to_thread(
                self.client.get_ohlcv, symbol, timeframe, limit
            )

            ohlcv_data[timeframe] = bars

            # Calculate indicators for this timeframe
            if len(bars) >= 30:  # Need minimum data for indicators
                indicators[timeframe] = self._calculate_indicators(bars)
            else:
                indicators[timeframe] = IndicatorValues()

        return PriceFeedData(
            symbol=symbol,
            current_price=current_price,
            ohlcv_1m=ohlcv_data.get("1m", []),
            ohlcv_5m=ohlcv_data.get("5m", []),
            ohlcv_15m=ohlcv_data.get("15m", []),
            ohlcv_1h=ohlcv_data.get("1h", []),
            ohlcv_4h=ohlcv_data.get("4h", []),
            indicators=indicators,
            timestamp=datetime.now(timezone.utc),
        )

    def _calculate_indicators(self, bars: list[OHLCVBar]) -> IndicatorValues:
        """
        Calculate all technical indicators from OHLCV data.

        Args:
            bars: List of OHLCVBar objects

        Returns:
            IndicatorValues object
        """
        if len(bars) < 30:
            return IndicatorValues()

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            for bar in bars
        ])

        close = df['close']
        high = df['high']
        low = df['low']

        # RSI (14 periods)
        rsi_14 = self._calculate_rsi(close, 14)

        # MACD (12, 26, 9)
        macd_line, macd_signal, macd_histogram = self._calculate_macd(close)

        # EMAs
        ema_20 = self._calculate_ema(close, 20)
        ema_50 = self._calculate_ema(close, 50)
        ema_200 = self._calculate_ema(close, 200) if len(close) >= 200 else None

        # ATR (14 periods)
        atr_14 = self._calculate_atr(high, low, close, 14)

        # ADX (14 periods)
        adx_14 = self._calculate_adx(high, low, close, 14)

        # Bollinger Bands (20, 2)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2)

        return IndicatorValues(
            rsi_14=rsi_14,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            atr_14=atr_14,
            adx_14=adx_14,
            bollinger_upper=bb_upper,
            bollinger_middle=bb_middle,
            bollinger_lower=bb_lower,
        )

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float | None:
        """
        Calculate RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(close) < period + 1:
            return None

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    def _calculate_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[float | None, float | None, float | None]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        """
        if len(close) < slow:
            return None, None, None

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None,
            float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None,
        )

    def _calculate_ema(self, close: pd.Series, period: int) -> float | None:
        """Calculate Exponential Moving Average."""
        if len(close) < period:
            return None

        ema = close.ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float | None:
        """
        Calculate ATR (Average True Range).

        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = SMA(TR, period)
        """
        if len(close) < period + 1:
            return None

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> float | None:
        """
        Calculate ADX (Average Directional Index).

        +DM = high - prev_high (if > 0 and > |low - prev_low|)
        -DM = prev_low - low (if > 0 and > high - prev_high)
        +DI = 100 * EMA(+DM) / ATR
        -DI = 100 * EMA(-DM) / ATR
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = EMA(DX)
        """
        if len(close) < period * 2:
            return None

        try:
            # Calculate +DM and -DM
            high_diff = high.diff()
            low_diff = -low.diff()

            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

            # Calculate ATR
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Smooth with EMA
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.ewm(span=period, adjust=False).mean()

            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None

        except Exception as e:
            log.warning(f"ADX calculation error: {e}")
            return None

    def _calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[float | None, float | None, float | None]:
        """
        Calculate Bollinger Bands.

        Middle = SMA(close, period)
        Upper = Middle + std_dev * STD(close, period)
        Lower = Middle - std_dev * STD(close, period)
        """
        if len(close) < period:
            return None, None, None

        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return (
            float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
            float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None,
            float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
        )
