"""
State Aggregator for AI Trading System V3.
Combines all sensor data into unified MarketState.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import FundingData, MarketState, PriceFeedData, SensorSnapshot, VolumeData
from execution.binance_client import BinanceClient
from sensors.funding_rate import FundingRateSensor
from sensors.price_feed import PriceFeedSensor
from sensors.volume_analyzer import VolumeAnalyzerSensor

log = get_logger("aggregator")


class StateAggregator:
    """
    Aggregates data from all sensors into unified MarketState.

    Features:
    - Parallel data collection from all sensors
    - Creates SensorSnapshot for each symbol
    - Generates human-readable summary for LLM
    """

    def __init__(
        self,
        client: BinanceClient,
        symbols: list[str],
    ) -> None:
        """
        Initialize StateAggregator.

        Args:
            client: BinanceClient instance
            symbols: List of symbols to track
        """
        self.client = client
        self.symbols = symbols

        # Initialize sensors
        self.price_feed = PriceFeedSensor(client)
        self.volume_analyzer = VolumeAnalyzerSensor(client)
        self.funding_rate = FundingRateSensor(client)

        log.info(f"[Aggregator] Initialized with {len(symbols)} symbols")

    async def collect_all(self) -> MarketState:
        """
        Collect data from all sensors for all symbols.

        Returns:
            Complete MarketState object
        """
        start_time = time.time()
        snapshots: dict[str, SensorSnapshot] = {}

        # Collect for each symbol in parallel
        tasks = [self.collect_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                log.error(f"[Aggregator] Failed to collect {symbol}: {result}")
                continue
            if result:
                snapshots[symbol] = result

        elapsed = time.time() - start_time
        log.info(f"[Aggregator] Collected {len(snapshots)}/{len(self.symbols)} symbols in {elapsed:.2f}s")

        return MarketState(
            snapshots=snapshots,
            global_metrics={},  # Can add BTC dominance, fear/greed index, etc.
            timestamp=datetime.now(timezone.utc),
        )

    async def collect_symbol(self, symbol: str) -> SensorSnapshot | None:
        """
        Collect all sensor data for a single symbol.

        Args:
            symbol: Trading symbol

        Returns:
            SensorSnapshot or None on failure
        """
        try:
            start = time.time()

            # Collect from all sensors in parallel
            price_task = self.price_feed.collect(symbol)
            volume_task = self.volume_analyzer.collect(symbol)
            funding_task = self.funding_rate.collect(symbol)

            price_data, volume_data, funding_data = await asyncio.gather(
                price_task, volume_task, funding_task,
                return_exceptions=True
            )

            # Handle any exceptions
            if isinstance(price_data, Exception):
                log.error(f"[Aggregator] Price feed error for {symbol}: {price_data}")
                price_data = self._empty_price_feed(symbol)

            if isinstance(volume_data, Exception):
                log.error(f"[Aggregator] Volume error for {symbol}: {volume_data}")
                volume_data = self._empty_volume_data(symbol)

            if isinstance(funding_data, Exception):
                log.error(f"[Aggregator] Funding error for {symbol}: {funding_data}")
                funding_data = self._empty_funding_data(symbol)

            elapsed = time.time() - start
            log.debug(f"[Aggregator] {symbol}: collected in {elapsed:.2f}s")

            return SensorSnapshot(
                symbol=symbol,
                price_feed=price_data,
                volume=volume_data,
                funding=funding_data,
                collected_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            log.error(f"[Aggregator] Failed to collect {symbol}: {e}")
            return None

    def get_summary(self, state: MarketState) -> str:
        """
        Generate human-readable summary of market state for LLM.

        Args:
            state: MarketState object

        Returns:
            Formatted summary string
        """
        lines = [
            f"Market State at {state.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        for symbol, snapshot in state.snapshots.items():
            price = snapshot.price_feed
            volume = snapshot.volume
            funding = snapshot.funding

            # Get 1h indicators for summary
            ind = price.indicators.get("1h")

            # Calculate 24h change approximation from price feed
            if price.ohlcv_1h and len(price.ohlcv_1h) >= 24:
                price_24h_ago = price.ohlcv_1h[-24].close
                change_pct = ((price.current_price - price_24h_ago) / price_24h_ago) * 100
            else:
                change_pct = 0.0

            change_sign = "+" if change_pct >= 0 else ""

            # Format symbol line
            lines.append(f"{symbol}: ${price.current_price:.4f} ({change_sign}{change_pct:.1f}%)")

            # Indicators line
            if ind:
                rsi_str = f"RSI: {ind.rsi_14:.1f}" if ind.rsi_14 else "RSI: N/A"

                if ind.macd_histogram:
                    macd_trend = "bullish" if ind.macd_histogram > 0 else "bearish"
                else:
                    macd_trend = "N/A"

                lines.append(f"  {rsi_str} | MACD: {macd_trend}")
            else:
                lines.append("  Indicators: N/A")

            # Volume and funding line
            vol_str = f"Volume: {volume.relative_volume:.1f}x avg"
            if volume.volume_spike:
                vol_str += " (SPIKE)"

            funding_pct = funding.current_rate * 100
            funding_str = f"Funding: {funding_pct:.4f}%"

            if funding_pct > 0.01:
                funding_str += " (bullish)"
            elif funding_pct < -0.01:
                funding_str += " (bearish)"
            else:
                funding_str += " (neutral)"

            lines.append(f"  {vol_str} | {funding_str}")
            lines.append("")

        return "\n".join(lines)

    def _empty_price_feed(self, symbol: str) -> PriceFeedData:
        """Create empty PriceFeedData for error cases."""
        return PriceFeedData(
            symbol=symbol,
            current_price=0.0,
            indicators={},
            timestamp=datetime.now(timezone.utc),
        )

    def _empty_volume_data(self, symbol: str) -> VolumeData:
        """Create empty VolumeData for error cases."""
        return VolumeData(
            symbol=symbol,
            current_volume=0.0,
            volume_sma_20=0.0,
            relative_volume=0.0,
            volume_spike=False,
            buy_volume_ratio=0.5,
            timestamp=datetime.now(timezone.utc),
        )

    def _empty_funding_data(self, symbol: str) -> FundingData:
        """Create empty FundingData for error cases."""
        return FundingData(
            symbol=symbol,
            current_rate=0.0,
            predicted_rate=0.0,
            next_funding_time=datetime.now(timezone.utc),
            rate_trend="stable",
            timestamp=datetime.now(timezone.utc),
        )

    def get_sensor_status(self) -> dict[str, Any]:
        """Get status of all sensors."""
        return {
            'price_feed': self.price_feed.get_status(),
            'volume_analyzer': self.volume_analyzer.get_status(),
            'funding_rate': self.funding_rate.get_status(),
        }
