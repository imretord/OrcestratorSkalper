"""
Market sentiment analyzer for AI Trading System V3.
Analyzes Fear & Greed Index, Long/Short ratios, Open Interest, and funding rates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np

from analyzers.base_analyzer import CachedAnalyzer
from core.logger import get_logger
from core.state import SensorSnapshot, SentimentAnalysis
from execution.binance_client import BinanceClient

log = get_logger("sentiment_analyzer")


class SentimentAnalyzer(CachedAnalyzer[SentimentAnalysis]):
    """
    Analyzes market sentiment using multiple data sources.

    Data sources:
    - Fear & Greed Index (alternative.me API)
    - Long/Short Ratio (Binance)
    - Open Interest Changes (Binance)
    - Funding Rate Analysis (from sensor data)
    - Taker Buy/Sell Volume (Binance)

    Produces sentiment scores from -1 (bearish) to +1 (bullish).
    """

    # Fear & Greed API endpoint
    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    # Sentiment thresholds
    EXTREME_FEAR_THRESHOLD = 25
    FEAR_THRESHOLD = 45
    GREED_THRESHOLD = 55
    EXTREME_GREED_THRESHOLD = 75

    def __init__(
        self,
        client: BinanceClient,
        cache_seconds: int = 300,  # 5 minute cache for sentiment
    ) -> None:
        """
        Initialize sentiment analyzer.

        Args:
            client: Binance client for market data
            cache_seconds: Cache duration for sentiment analysis
        """
        super().__init__("SentimentAnalyzer", cache_seconds)
        self.client = client
        self._fear_greed_cache: tuple[int, str, datetime] | None = None
        self._fear_greed_cache_duration = 600  # 10 minutes for F&G

    def analyze(self, snapshot: SensorSnapshot) -> SentimentAnalysis:
        """
        Analyze market sentiment for a symbol.

        Args:
            snapshot: Complete sensor data for a symbol

        Returns:
            SentimentAnalysis with sentiment indicators
        """
        symbol = snapshot.symbol

        # Fetch Fear & Greed Index (global crypto sentiment)
        fear_greed_index, fear_greed_label = self._get_fear_greed_index()

        # Fetch Long/Short ratio from Binance
        long_short_ratio = self._get_long_short_ratio(symbol)

        # Calculate Open Interest change
        oi_change_24h = self._calculate_oi_change(symbol)

        # Analyze funding rate sentiment
        funding_sentiment = self._analyze_funding_sentiment(snapshot.funding.current_rate)

        # Get taker volume sentiment
        social_sentiment = self._analyze_taker_volume(symbol)

        # Calculate overall sentiment score
        sentiment_score = self._calculate_overall_score(
            fear_greed_index=fear_greed_index,
            long_short_ratio=long_short_ratio,
            oi_change_24h=oi_change_24h,
            funding_sentiment=funding_sentiment,
            social_sentiment=social_sentiment,
        )

        # Determine overall sentiment label
        overall_sentiment = self._score_to_label(sentiment_score)

        return SentimentAnalysis(
            symbol=symbol,
            fear_greed_index=fear_greed_index,
            fear_greed_label=fear_greed_label,
            long_short_ratio=long_short_ratio,
            open_interest_change_24h=oi_change_24h,
            funding_sentiment=funding_sentiment,
            social_sentiment=social_sentiment,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(timezone.utc),
        )

    def _get_fear_greed_index(self) -> tuple[int, str]:
        """
        Fetch Fear & Greed Index from alternative.me API.

        Returns:
            Tuple of (index value 0-100, classification label)
        """
        # Check cache
        if self._fear_greed_cache:
            index, label, cached_time = self._fear_greed_cache
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self._fear_greed_cache_duration:
                return index, label

        try:
            with httpx.Client(timeout=10.0) as http_client:
                response = http_client.get(self.FEAR_GREED_URL)
                response.raise_for_status()
                data = response.json()

            if data and 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                index = int(latest.get('value', 50))
                label = latest.get('value_classification', 'Neutral')

                # Cache the result
                self._fear_greed_cache = (index, label, datetime.now(timezone.utc))

                log.debug(f"Fear & Greed Index: {index} ({label})")
                return index, label

        except Exception as e:
            log.warning(f"Failed to fetch Fear & Greed Index: {e}")

        # Return neutral if failed
        return 50, "Neutral"

    def _get_long_short_ratio(self, symbol: str) -> float:
        """
        Get Long/Short ratio from Binance.

        Args:
            symbol: Trading symbol

        Returns:
            Long/Short ratio (> 1 means more longs)
        """
        try:
            data = self.client.get_long_short_ratio(symbol, period="1h")
            if data:
                return data.get('long_short_ratio', 1.0)
        except Exception as e:
            log.warning(f"Failed to get long/short ratio for {symbol}: {e}")

        return 1.0  # Neutral default

    def _calculate_oi_change(self, symbol: str) -> float:
        """
        Calculate Open Interest change over 24 hours.

        Args:
            symbol: Trading symbol

        Returns:
            Percentage change in OI (e.g., 5.5 means +5.5%)
        """
        try:
            # Get historical OI data
            oi_history = self.client.get_open_interest_history(symbol, period="1h", limit=24)

            if len(oi_history) >= 2:
                oldest_oi = oi_history[0].get('open_interest_value', 0)
                latest_oi = oi_history[-1].get('open_interest_value', 0)

                if oldest_oi > 0:
                    change_pct = ((latest_oi - oldest_oi) / oldest_oi) * 100
                    return round(change_pct, 2)

        except Exception as e:
            log.warning(f"Failed to calculate OI change for {symbol}: {e}")

        return 0.0

    def _analyze_funding_sentiment(self, funding_rate: float) -> str:
        """
        Analyze sentiment from funding rate.

        Args:
            funding_rate: Current funding rate

        Returns:
            "bullish", "bearish", or "neutral"
        """
        # High positive funding = too many longs = bearish signal
        # High negative funding = too many shorts = bullish signal
        if funding_rate > 0.0005:  # > 0.05%
            return "bearish"  # Crowded long trade
        elif funding_rate < -0.0005:  # < -0.05%
            return "bullish"  # Crowded short trade
        else:
            return "neutral"

    def _analyze_taker_volume(self, symbol: str) -> str:
        """
        Analyze taker buy/sell volume for sentiment.

        Args:
            symbol: Trading symbol

        Returns:
            "bullish", "bearish", or "neutral"
        """
        try:
            taker_data = self.client.get_taker_buy_sell_volume(symbol, period="1h", limit=4)

            if taker_data:
                # Average buy/sell ratio over recent periods
                ratios = [d.get('buy_sell_ratio', 1.0) for d in taker_data]
                avg_ratio = np.mean(ratios)

                if avg_ratio > 1.1:  # 10% more buy volume
                    return "bullish"
                elif avg_ratio < 0.9:  # 10% more sell volume
                    return "bearish"

        except Exception as e:
            log.warning(f"Failed to analyze taker volume for {symbol}: {e}")

        return "neutral"

    def _calculate_overall_score(
        self,
        fear_greed_index: int,
        long_short_ratio: float,
        oi_change_24h: float,
        funding_sentiment: str,
        social_sentiment: str,
    ) -> float:
        """
        Calculate overall sentiment score.

        Args:
            fear_greed_index: 0-100 index
            long_short_ratio: L/S ratio
            oi_change_24h: OI change percentage
            funding_sentiment: "bullish"/"bearish"/"neutral"
            social_sentiment: "bullish"/"bearish"/"neutral"

        Returns:
            Score from -1 (bearish) to +1 (bullish)
        """
        scores = []
        weights = []

        # Fear & Greed Index (25% weight)
        # Convert 0-100 to -1 to +1
        fg_score = (fear_greed_index - 50) / 50
        scores.append(fg_score)
        weights.append(0.25)

        # Long/Short Ratio (20% weight)
        # > 1 means more longs (bullish), < 1 means more shorts (bearish)
        # Normalize: 0.5 = -1, 1.0 = 0, 2.0 = +1
        ls_score = np.clip((long_short_ratio - 1.0) * 2, -1, 1)
        scores.append(ls_score)
        weights.append(0.20)

        # OI Change (15% weight)
        # Rising OI with price up = bullish confirmation
        # Rising OI alone could be either direction
        # Normalize: -20% = -1, 0% = 0, +20% = +1
        oi_score = np.clip(oi_change_24h / 20, -1, 1)
        scores.append(oi_score)
        weights.append(0.15)

        # Funding Sentiment (20% weight)
        # Contrarian: high positive funding = bearish, high negative = bullish
        funding_scores = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}
        scores.append(funding_scores.get(funding_sentiment, 0.0))
        weights.append(0.20)

        # Social/Taker Sentiment (20% weight)
        social_scores = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}
        scores.append(social_scores.get(social_sentiment, 0.0))
        weights.append(0.20)

        # Weighted average
        total_score = sum(s * w for s, w in zip(scores, weights))

        return round(np.clip(total_score, -1, 1), 3)

    def _score_to_label(self, score: float) -> str:
        """
        Convert sentiment score to label.

        Args:
            score: -1 to +1 score

        Returns:
            "bullish", "bearish", or "neutral"
        """
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        else:
            return "neutral"

    def get_fear_greed_description(self) -> str:
        """Get human-readable Fear & Greed description."""
        index, label = self._get_fear_greed_index()

        if index <= self.EXTREME_FEAR_THRESHOLD:
            return f"Extreme Fear ({index}/100) - Potential buying opportunity"
        elif index <= self.FEAR_THRESHOLD:
            return f"Fear ({index}/100) - Market is cautious"
        elif index >= self.EXTREME_GREED_THRESHOLD:
            return f"Extreme Greed ({index}/100) - Market may be overheated"
        elif index >= self.GREED_THRESHOLD:
            return f"Greed ({index}/100) - Market is optimistic"
        else:
            return f"Neutral ({index}/100) - Market is balanced"

    def get_sentiment_summary(self, analysis: SentimentAnalysis) -> str:
        """
        Generate human-readable sentiment summary.

        Args:
            analysis: SentimentAnalysis result

        Returns:
            Summary string
        """
        summary_parts = [
            f"Overall: {analysis.overall_sentiment.upper()} (score: {analysis.sentiment_score:+.2f})",
            f"Fear & Greed: {analysis.fear_greed_index} ({analysis.fear_greed_label})",
            f"Long/Short: {analysis.long_short_ratio:.2f}",
            f"OI Change 24h: {analysis.open_interest_change_24h:+.1f}%",
            f"Funding: {analysis.funding_sentiment}",
            f"Taker Flow: {analysis.social_sentiment}",
        ]

        return " | ".join(summary_parts)
