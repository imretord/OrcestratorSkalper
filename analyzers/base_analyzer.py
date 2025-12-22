"""
Base analyzer class for AI Trading System V3.
All analyzers inherit from this base class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from core.logger import get_logger
from core.state import SensorSnapshot

log = get_logger("analyzers")

# Generic type for analysis result
T = TypeVar("T")


class BaseAnalyzer(ABC, Generic[T]):
    """
    Base class for all market analyzers.

    Analyzers process SensorSnapshot data and produce typed analysis results.
    Each analyzer focuses on a specific aspect of market analysis.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize base analyzer.

        Args:
            name: Analyzer name for logging
        """
        self.name = name
        self._last_analysis: T | None = None
        self._last_analysis_time: datetime | None = None
        self._analysis_count: int = 0
        self._error_count: int = 0

    @abstractmethod
    def analyze(self, snapshot: SensorSnapshot) -> T:
        """
        Perform analysis on sensor snapshot.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            Typed analysis result
        """
        pass

    def safe_analyze(self, snapshot: SensorSnapshot) -> T | None:
        """
        Safely perform analysis with error handling.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            Analysis result or None if failed
        """
        try:
            result = self.analyze(snapshot)
            self._last_analysis = result
            self._last_analysis_time = datetime.now(timezone.utc)
            self._analysis_count += 1
            return result

        except Exception as e:
            self._error_count += 1
            log.error(f"[{self.name}] Analysis failed for {snapshot.symbol}: {e}")
            return None

    def get_last_analysis(self) -> T | None:
        """Get the most recent analysis result."""
        return self._last_analysis

    def get_stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "name": self.name,
            "analysis_count": self._analysis_count,
            "error_count": self._error_count,
            "success_rate": (
                self._analysis_count / max(1, self._analysis_count + self._error_count) * 100
            ),
            "last_analysis_time": (
                self._last_analysis_time.isoformat() if self._last_analysis_time else None
            ),
        }

    def reset_stats(self) -> None:
        """Reset analyzer statistics."""
        self._analysis_count = 0
        self._error_count = 0
        self._last_analysis = None
        self._last_analysis_time = None


class CachedAnalyzer(BaseAnalyzer[T]):
    """
    Analyzer with result caching.

    Caches results for a configurable duration to avoid redundant calculations.
    """

    def __init__(self, name: str, cache_seconds: int = 60) -> None:
        """
        Initialize cached analyzer.

        Args:
            name: Analyzer name for logging
            cache_seconds: How long to cache results (default 60s)
        """
        super().__init__(name)
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[T, datetime]] = {}

    def analyze_cached(self, snapshot: SensorSnapshot) -> T | None:
        """
        Perform analysis with caching.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            Cached or fresh analysis result
        """
        symbol = snapshot.symbol
        now = datetime.now(timezone.utc)

        # Check cache
        if symbol in self._cache:
            cached_result, cached_time = self._cache[symbol]
            age_seconds = (now - cached_time).total_seconds()

            if age_seconds < self.cache_seconds:
                log.debug(f"[{self.name}] Using cached result for {symbol} (age: {age_seconds:.1f}s)")
                return cached_result

        # Perform fresh analysis
        result = self.safe_analyze(snapshot)

        if result is not None:
            self._cache[symbol] = (result, now)

        return result

    def clear_cache(self, symbol: str | None = None) -> None:
        """
        Clear cached results.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now(timezone.utc)
        stats = self.get_stats()

        cache_info = {}
        for symbol, (_, cached_time) in self._cache.items():
            age_seconds = (now - cached_time).total_seconds()
            cache_info[symbol] = {
                "age_seconds": age_seconds,
                "is_valid": age_seconds < self.cache_seconds,
            }

        stats["cache"] = {
            "size": len(self._cache),
            "cache_seconds": self.cache_seconds,
            "entries": cache_info,
        }

        return stats
