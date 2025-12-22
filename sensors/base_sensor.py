"""
Base sensor class for AI Trading System V3.
All sensors inherit from this abstract base class.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, TypeVar

from core.logger import get_logger

log = get_logger("sensor")

T = TypeVar("T")


class BaseSensor(ABC):
    """
    Abstract base class for all market data sensors.

    Sensors are responsible for collecting specific types of market data
    (prices, volume, funding rates, etc.) and returning structured data.

    Features:
    - Update interval control
    - Health monitoring
    - Parallel data collection for multiple symbols
    """

    def __init__(
        self,
        name: str,
        update_interval_seconds: int = 60,
    ) -> None:
        """
        Initialize base sensor.

        Args:
            name: Sensor name for logging
            update_interval_seconds: Minimum seconds between updates
        """
        self.name = name
        self.update_interval_seconds = update_interval_seconds
        self.last_update: datetime | None = None
        self.is_healthy: bool = True
        self._consecutive_failures: int = 0
        self._max_failures: int = 3

    @abstractmethod
    async def collect(self, symbol: str) -> Any:
        """
        Collect data for a single symbol.

        Args:
            symbol: Trading symbol (e.g., "DOGEUSDT")

        Returns:
            Collected data (type depends on sensor implementation)
        """
        pass

    async def collect_all(self, symbols: list[str]) -> dict[str, Any]:
        """
        Collect data for all symbols in parallel.

        Args:
            symbols: List of trading symbols

        Returns:
            Dict mapping symbol to collected data
        """
        results: dict[str, Any] = {}

        # Create tasks for parallel execution
        tasks = [self._collect_with_error_handling(symbol) for symbol in symbols]
        collected = await asyncio.gather(*tasks)

        for symbol, data in zip(symbols, collected):
            if data is not None:
                results[symbol] = data

        self.last_update = datetime.now(timezone.utc)

        return results

    async def _collect_with_error_handling(self, symbol: str) -> Any:
        """
        Wrapper for collect() with error handling.

        Args:
            symbol: Trading symbol

        Returns:
            Collected data or None on error
        """
        try:
            data = await self.collect(symbol)
            self._consecutive_failures = 0
            return data

        except Exception as e:
            self._consecutive_failures += 1
            log.error(f"[{self.name}] Error collecting {symbol}: {e}")

            if self._consecutive_failures >= self._max_failures:
                self.is_healthy = False
                log.warning(f"[{self.name}] Marked unhealthy after {self._consecutive_failures} failures")

            return None

    def should_update(self) -> bool:
        """
        Check if enough time has passed for an update.

        Returns:
            True if sensor should update
        """
        if self.last_update is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self.last_update).total_seconds()
        return elapsed >= self.update_interval_seconds

    def health_check(self) -> bool:
        """
        Check sensor health.

        Returns:
            True if sensor is healthy
        """
        return self.is_healthy

    def reset_health(self) -> None:
        """Reset health status after recovery."""
        self.is_healthy = True
        self._consecutive_failures = 0
        log.info(f"[{self.name}] Health reset")

    def get_status(self) -> dict[str, Any]:
        """
        Get sensor status information.

        Returns:
            Dict with status info
        """
        return {
            'name': self.name,
            'healthy': self.is_healthy,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval': self.update_interval_seconds,
            'consecutive_failures': self._consecutive_failures,
        }
