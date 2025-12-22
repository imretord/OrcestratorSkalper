"""
Funding Rate Sensor for AI Trading System V3.
Monitors perpetual futures funding rates.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from core.logger import get_logger
from core.state import FundingData
from execution.binance_client import BinanceClient
from sensors.base_sensor import BaseSensor

log = get_logger("funding_rate")


class FundingRateSensor(BaseSensor):
    """
    Sensor for funding rate data.

    Collects:
    - Current funding rate
    - Predicted (next) funding rate
    - Time until next funding
    - Rate trend (rising/falling/stable)

    Interpretation:
    - Positive (> 0.01%): Overleveraged longs, market may be overheated upward
    - Negative (< -0.01%): Overleveraged shorts, market may be overheated downward
    - Neutral (-0.01% to 0.01%): Balanced market
    """

    def __init__(
        self,
        client: BinanceClient,
        update_interval_seconds: int = 300,  # 5 minutes
    ) -> None:
        """
        Initialize FundingRateSensor.

        Args:
            client: BinanceClient instance
            update_interval_seconds: Update interval
        """
        super().__init__(name="FundingRate", update_interval_seconds=update_interval_seconds)
        self.client = client

    async def collect(self, symbol: str) -> FundingData:
        """
        Collect funding rate data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            FundingData object
        """
        funding_data = await asyncio.to_thread(
            self.client.get_funding_rate, symbol
        )

        if funding_data:
            # Log significant funding rates
            rate_pct = funding_data.current_rate * 100
            if abs(rate_pct) > 0.05:
                direction = "high positive (longs pay)" if rate_pct > 0 else "high negative (shorts pay)"
                log.info(f"[FundingRate] {symbol}: {rate_pct:.4f}% - {direction}")

            return funding_data

        # Return empty data on failure
        log.warning(f"[FundingRate] Failed to fetch funding rate for {symbol}")
        return FundingData(
            symbol=symbol,
            current_rate=0.0,
            predicted_rate=0.0,
            next_funding_time=datetime.now(timezone.utc),
            rate_trend="stable",
            timestamp=datetime.now(timezone.utc),
        )

    def interpret_funding_rate(self, rate: float) -> str:
        """
        Provide interpretation of funding rate.

        Args:
            rate: Funding rate as decimal (e.g., 0.0001 = 0.01%)

        Returns:
            Interpretation string
        """
        rate_pct = rate * 100

        if rate_pct > 0.05:
            return "highly_bullish_overheated"
        elif rate_pct > 0.01:
            return "bullish_bias"
        elif rate_pct < -0.05:
            return "highly_bearish_overheated"
        elif rate_pct < -0.01:
            return "bearish_bias"
        else:
            return "neutral"

    def get_hours_until_funding(self, next_funding_time: datetime) -> float:
        """
        Calculate hours until next funding.

        Args:
            next_funding_time: Next funding timestamp

        Returns:
            Hours until funding
        """
        now = datetime.now(timezone.utc)
        delta = next_funding_time - now
        return max(0, delta.total_seconds() / 3600)
