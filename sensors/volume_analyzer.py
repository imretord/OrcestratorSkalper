"""
Volume Analyzer Sensor for AI Trading System V3.
Analyzes trading volume patterns and detects anomalies.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from core.logger import get_logger
from core.state import VolumeData
from execution.binance_client import BinanceClient
from sensors.base_sensor import BaseSensor

log = get_logger("volume_analyzer")


class VolumeAnalyzerSensor(BaseSensor):
    """
    Sensor for volume analysis and anomaly detection.

    Calculates:
    - Current volume
    - Volume SMA (20 periods)
    - Relative volume (current / SMA)
    - Volume spikes (relative > threshold)
    - Buy/sell volume ratio estimate
    """

    def __init__(
        self,
        client: BinanceClient,
        sma_period: int = 20,
        spike_threshold: float = 2.0,
        update_interval_seconds: int = 60,
    ) -> None:
        """
        Initialize VolumeAnalyzerSensor.

        Args:
            client: BinanceClient instance
            sma_period: Period for volume SMA
            spike_threshold: Threshold for volume spike detection
            update_interval_seconds: Update interval
        """
        super().__init__(name="VolumeAnalyzer", update_interval_seconds=update_interval_seconds)
        self.client = client
        self.sma_period = sma_period
        self.spike_threshold = spike_threshold

    async def collect(self, symbol: str) -> VolumeData:
        """
        Collect volume data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            VolumeData object
        """
        # Fetch last 30 hourly candles for analysis
        bars = await asyncio.to_thread(
            self.client.get_ohlcv, symbol, "1h", 30
        )

        if len(bars) < self.sma_period:
            log.warning(f"[VolumeAnalyzer] Insufficient data for {symbol}: {len(bars)} bars")
            return VolumeData(
                symbol=symbol,
                current_volume=0.0,
                volume_sma_20=0.0,
                relative_volume=0.0,
                volume_spike=False,
                buy_volume_ratio=0.5,
                timestamp=datetime.now(timezone.utc),
            )

        # Get current volume (last candle)
        current_volume = bars[-1].volume

        # Calculate SMA of volume
        volumes = [bar.volume for bar in bars]
        sma_volumes = volumes[-self.sma_period:]
        volume_sma = sum(sma_volumes) / len(sma_volumes)

        # Calculate relative volume
        relative_volume = current_volume / volume_sma if volume_sma > 0 else 0.0

        # Detect spike
        volume_spike = relative_volume > self.spike_threshold

        # Calculate buy/sell ratio
        buy_volume_ratio = self._calculate_buy_sell_ratio(bars)

        if volume_spike:
            log.info(f"[VolumeAnalyzer] {symbol}: Volume spike detected! {relative_volume:.2f}x average")

        return VolumeData(
            symbol=symbol,
            current_volume=current_volume,
            volume_sma_20=volume_sma,
            relative_volume=relative_volume,
            volume_spike=volume_spike,
            buy_volume_ratio=buy_volume_ratio,
            timestamp=datetime.now(timezone.utc),
        )

    def _calculate_buy_sell_ratio(self, bars: list) -> float:
        """
        Estimate buy/sell volume ratio from candlestick data.

        Logic: If close > open, candle is "bullish" (buying pressure)
               If close < open, candle is "bearish" (selling pressure)
               Weight by volume

        Args:
            bars: List of OHLCVBar objects

        Returns:
            Ratio of buy volume (0.0 to 1.0)
        """
        if not bars:
            return 0.5

        total_volume = 0.0
        buy_volume = 0.0

        for bar in bars:
            volume = bar.volume
            total_volume += volume

            if bar.close > bar.open:
                # Bullish candle - count as buy volume
                buy_volume += volume
            elif bar.close == bar.open:
                # Doji - count as 50% buy
                buy_volume += volume * 0.5
            # else: bearish - contributes to sell volume (not counted as buy)

        if total_volume == 0:
            return 0.5

        return buy_volume / total_volume
