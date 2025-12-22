"""
Experience buffer for AI Trading System V3.
Stores trade experiences for online learning.
"""
from __future__ import annotations

import json
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from core.logger import get_logger
from core.state import (
    MarketContext,
    MarketRegime,
    TradeExperience,
)

log = get_logger("experience_buffer")


class ExperienceBuffer:
    """
    Circular buffer for storing trade experiences.

    Features:
    - Configurable capacity with FIFO eviction
    - Persistence to disk (JSON)
    - Sampling methods (random, recent, by regime)
    - Statistics tracking
    - Feature extraction for learning
    """

    def __init__(
        self,
        capacity: int = 1000,
        persist_path: str | None = None,
    ) -> None:
        """
        Initialize experience buffer.

        Args:
            capacity: Maximum number of experiences to store
            persist_path: Path to persist experiences (optional)
        """
        self.capacity = capacity
        self.persist_path = Path(persist_path) if persist_path else None

        # Internal storage
        self._buffer: deque[TradeExperience] = deque(maxlen=capacity)
        self._trade_counter: int = 0

        # Statistics
        self._stats = {
            'total_added': 0,
            'total_wins': 0,
            'total_losses': 0,
            'total_pnl': 0.0,
            'best_trade_pnl': 0.0,
            'worst_trade_pnl': 0.0,
            'avg_duration_seconds': 0.0,
        }

        # Load existing data if available
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

        log.info(f"[ExperienceBuffer] Initialized with capacity={capacity}, loaded={len(self._buffer)}")

    def add(self, experience: TradeExperience) -> None:
        """
        Add an experience to the buffer.

        Args:
            experience: Trade experience to add
        """
        self._buffer.append(experience)
        self._trade_counter += 1
        self._update_stats(experience)

        log.debug(f"[ExperienceBuffer] Added trade {experience.trade_id}: PnL={experience.pnl_percent:+.2f}%")

        # Auto-persist periodically
        if self.persist_path and self._trade_counter % 10 == 0:
            self._save_to_disk()

    def add_from_context(
        self,
        context: MarketContext,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        exit_reason: str,
        entry_time: datetime,
        exit_time: datetime,
    ) -> TradeExperience:
        """
        Create and add experience from market context.

        Args:
            context: Market context at trade entry
            side: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            exit_reason: Reason for exit
            entry_time: Entry timestamp
            exit_time: Exit timestamp

        Returns:
            Created TradeExperience
        """
        # Calculate PnL
        if side == "long":
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        pnl_usdt = quantity * entry_price * (pnl_percent / 100)

        # Calculate reward for learning (-1 to +1 scale)
        # Clip extreme values
        reward = np.clip(pnl_percent / 5.0, -1, 1)  # ±5% maps to ±1

        # Extract indicator values
        ind_1h = context.price_feed.indicators.get("1h")
        rsi = ind_1h.rsi_14 if ind_1h and ind_1h.rsi_14 else 50.0
        macd_hist = ind_1h.macd_histogram if ind_1h and ind_1h.macd_histogram else 0.0

        experience = TradeExperience(
            trade_id=f"trade_{self._trade_counter + 1}",
            symbol=context.symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl_usdt=pnl_usdt,
            pnl_percent=pnl_percent,
            duration_seconds=int((exit_time - entry_time).total_seconds()),
            regime_at_entry=context.regime.regime,
            sentiment_at_entry=context.sentiment.sentiment_score,
            volatility_at_entry=context.regime.volatility_level,
            rsi_at_entry=rsi,
            macd_histogram_at_entry=macd_hist,
            volume_relative_at_entry=context.volume.relative_volume,
            funding_rate_at_entry=context.funding.current_rate,
            exit_reason=exit_reason,
            success=pnl_percent > 0,
            reward=reward,
            entry_time=entry_time,
            exit_time=exit_time,
        )

        self.add(experience)
        return experience

    def add_from_context_dict(
        self,
        context_dict: dict[str, Any],
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        pnl_percent: float,
    ) -> TradeExperience:
        """
        Create and add experience from context dictionary (snapshot saved at entry).

        Args:
            context_dict: Context snapshot saved at entry time
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            exit_reason: Reason for exit
            pnl_percent: Realized PnL percentage

        Returns:
            Created TradeExperience
        """
        # Calculate reward for learning (-1 to +1 scale)
        reward = np.clip(pnl_percent / 5.0, -1, 1)

        # Extract values from context dict with defaults
        regime_str = context_dict.get('regime', 'ranging')
        try:
            regime = MarketRegime(regime_str)
        except ValueError:
            regime = MarketRegime.RANGING

        # Convert sentiment from string to float if needed
        sentiment_raw = context_dict.get('sentiment', 0.0)
        if isinstance(sentiment_raw, str):
            sentiment_map = {
                'bullish': 0.5,
                'slightly_bullish': 0.25,
                'neutral': 0.0,
                'slightly_bearish': -0.25,
                'bearish': -0.5,
            }
            sentiment = sentiment_map.get(sentiment_raw.lower(), 0.0)
        else:
            sentiment = float(sentiment_raw) if sentiment_raw is not None else 0.0

        experience = TradeExperience(
            trade_id=f"trade_{self._trade_counter + 1}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=0.0,  # Not available from context dict
            pnl_usdt=0.0,  # Will be calculated if needed
            pnl_percent=pnl_percent,
            duration_seconds=0,  # Not available
            regime_at_entry=regime,
            sentiment_at_entry=sentiment,
            volatility_at_entry=context_dict.get('volatility', 'medium'),
            rsi_at_entry=50.0,  # Not in snapshot
            macd_histogram_at_entry=0.0,  # Not in snapshot
            volume_relative_at_entry=1.0,  # Not in snapshot
            funding_rate_at_entry=context_dict.get('funding_rate', 0.0),
            exit_reason=exit_reason,
            success=pnl_percent > 0,
            reward=reward,
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
        )

        self.add(experience)

        log.info(
            f"[ExperienceBuffer] Added from context dict: {symbol} {side} "
            f"PnL={pnl_percent:+.2f}% reward={reward:.2f}"
        )

        return experience

    def sample(self, n: int = 1, method: str = "random") -> list[TradeExperience]:
        """
        Sample experiences from buffer.

        Args:
            n: Number of samples
            method: Sampling method ("random", "recent", "uniform_regime")

        Returns:
            List of sampled experiences
        """
        if len(self._buffer) == 0:
            return []

        n = min(n, len(self._buffer))

        if method == "random":
            return random.sample(list(self._buffer), n)

        elif method == "recent":
            return list(self._buffer)[-n:]

        elif method == "uniform_regime":
            # Sample uniformly across regimes
            return self._sample_uniform_regime(n)

        else:
            log.warning(f"Unknown sampling method: {method}, using random")
            return random.sample(list(self._buffer), n)

    def _sample_uniform_regime(self, n: int) -> list[TradeExperience]:
        """
        Sample uniformly across different regimes.

        Args:
            n: Number of samples

        Returns:
            Sampled experiences
        """
        # Group by regime
        regime_groups: dict[MarketRegime, list[TradeExperience]] = {}
        for exp in self._buffer:
            regime = exp.regime_at_entry
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append(exp)

        # Sample from each regime
        samples = []
        regimes = list(regime_groups.keys())
        samples_per_regime = max(1, n // len(regimes))

        for regime in regimes:
            group = regime_groups[regime]
            group_samples = min(samples_per_regime, len(group))
            samples.extend(random.sample(group, group_samples))

        # Fill remaining slots randomly
        remaining = n - len(samples)
        if remaining > 0:
            all_remaining = [e for e in self._buffer if e not in samples]
            if all_remaining:
                samples.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))

        return samples[:n]

    def get_by_regime(self, regime: MarketRegime) -> list[TradeExperience]:
        """
        Get all experiences for a specific regime.

        Args:
            regime: Market regime to filter by

        Returns:
            List of experiences
        """
        return [exp for exp in self._buffer if exp.regime_at_entry == regime]

    def get_winners(self) -> list[TradeExperience]:
        """Get all winning trades."""
        return [exp for exp in self._buffer if exp.success]

    def get_losers(self) -> list[TradeExperience]:
        """Get all losing trades."""
        return [exp for exp in self._buffer if not exp.success]

    def extract_features(self, experience: TradeExperience) -> dict[str, float]:
        """
        Extract feature dict from experience for learning.

        Args:
            experience: Trade experience

        Returns:
            Feature dictionary
        """
        # Encode regime as one-hot
        regime_features = {
            f"regime_{r.value}": 1.0 if experience.regime_at_entry == r else 0.0
            for r in MarketRegime
        }

        # Encode volatility
        vol_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "extreme": 1.0}

        features = {
            # Continuous features (normalized)
            "sentiment": experience.sentiment_at_entry,
            "rsi_normalized": (experience.rsi_at_entry - 50) / 50,
            "macd_histogram": np.clip(experience.macd_histogram_at_entry / 0.01, -1, 1),
            "volume_relative": np.clip((experience.volume_relative_at_entry - 1) / 2, -1, 1),
            "funding_rate": np.clip(experience.funding_rate_at_entry / 0.001, -1, 1),
            "volatility": vol_map.get(experience.volatility_at_entry, 0.5),
            "side_long": 1.0 if experience.side == "long" else 0.0,
        }

        # Add regime features
        features.update(regime_features)

        return features

    def get_feature_matrix(
        self,
        n: int | None = None,
        method: str = "recent",
    ) -> tuple[list[dict[str, float]], list[float]]:
        """
        Get feature matrix and targets for learning.

        Args:
            n: Number of samples (None = all)
            method: Sampling method

        Returns:
            Tuple of (feature_dicts, rewards)
        """
        if n is None:
            experiences = list(self._buffer)
        else:
            experiences = self.sample(n, method)

        features = [self.extract_features(exp) for exp in experiences]
        rewards = [exp.reward for exp in experiences]

        return features, rewards

    def _update_stats(self, experience: TradeExperience) -> None:
        """Update internal statistics."""
        self._stats['total_added'] += 1

        if experience.success:
            self._stats['total_wins'] += 1
        else:
            self._stats['total_losses'] += 1

        self._stats['total_pnl'] += experience.pnl_usdt

        if experience.pnl_usdt > self._stats['best_trade_pnl']:
            self._stats['best_trade_pnl'] = experience.pnl_usdt

        if experience.pnl_usdt < self._stats['worst_trade_pnl']:
            self._stats['worst_trade_pnl'] = experience.pnl_usdt

        # Update average duration
        total = self._stats['total_added']
        current_avg = self._stats['avg_duration_seconds']
        new_avg = current_avg + (experience.duration_seconds - current_avg) / total
        self._stats['avg_duration_seconds'] = new_avg

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        total = self._stats['total_added']
        wins = self._stats['total_wins']

        return {
            'buffer_size': len(self._buffer),
            'capacity': self.capacity,
            'total_trades': total,
            'win_rate': (wins / total * 100) if total > 0 else 0.0,
            'total_pnl': self._stats['total_pnl'],
            'best_trade': self._stats['best_trade_pnl'],
            'worst_trade': self._stats['worst_trade_pnl'],
            'avg_duration_seconds': self._stats['avg_duration_seconds'],
        }

    def get_regime_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics by regime."""
        regime_stats = {}

        for regime in MarketRegime:
            regime_experiences = self.get_by_regime(regime)
            if not regime_experiences:
                continue

            wins = sum(1 for e in regime_experiences if e.success)
            total = len(regime_experiences)
            pnl = sum(e.pnl_percent for e in regime_experiences)

            regime_stats[regime.value] = {
                'count': total,
                'win_rate': (wins / total * 100) if total > 0 else 0.0,
                'total_pnl_pct': pnl,
                'avg_pnl_pct': pnl / total if total > 0 else 0.0,
            }

        return regime_stats

    def _save_to_disk(self) -> None:
        """Persist buffer to disk."""
        if not self.persist_path:
            return

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'experiences': [exp.model_dump() for exp in self._buffer],
                'stats': self._stats,
                'trade_counter': self._trade_counter,
            }

            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            log.debug(f"[ExperienceBuffer] Saved {len(self._buffer)} experiences to disk")

        except Exception as e:
            log.error(f"[ExperienceBuffer] Failed to save: {e}")

    def _load_from_disk(self) -> None:
        """Load buffer from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            experiences = data.get('experiences', [])
            for exp_dict in experiences:
                # Convert regime string back to enum
                if 'regime_at_entry' in exp_dict:
                    exp_dict['regime_at_entry'] = MarketRegime(exp_dict['regime_at_entry'])

                # Parse datetime strings
                for field in ['entry_time', 'exit_time']:
                    if field in exp_dict and isinstance(exp_dict[field], str):
                        exp_dict[field] = datetime.fromisoformat(exp_dict[field].replace('Z', '+00:00'))

                exp = TradeExperience(**exp_dict)
                self._buffer.append(exp)

            self._stats = data.get('stats', self._stats)
            self._trade_counter = data.get('trade_counter', 0)

            log.info(f"[ExperienceBuffer] Loaded {len(self._buffer)} experiences from disk")

        except Exception as e:
            log.error(f"[ExperienceBuffer] Failed to load: {e}")

    def clear(self) -> None:
        """Clear all experiences."""
        self._buffer.clear()
        self._trade_counter = 0
        self._stats = {
            'total_added': 0,
            'total_wins': 0,
            'total_losses': 0,
            'total_pnl': 0.0,
            'best_trade_pnl': 0.0,
            'worst_trade_pnl': 0.0,
            'avg_duration_seconds': 0.0,
        }

        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

        log.info("[ExperienceBuffer] Cleared all experiences")

    def __len__(self) -> int:
        """Return buffer size."""
        return len(self._buffer)

    def __iter__(self) -> Iterator[TradeExperience]:
        """Iterate over experiences."""
        return iter(self._buffer)
