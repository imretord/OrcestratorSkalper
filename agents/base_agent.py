"""
Base agent class for AI Trading System V3.
All trading agents inherit from this base class.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.logger import get_logger
from core.state import (
    AgentState,
    MarketContext,
    MarketRegime,
    Signal,
)
from learners.meta_learner import MetaLearner
from learners.online_predictor import OnlinePredictor

log = get_logger("agents")


class BaseAgent(ABC):
    """
    Base class for all trading agents.

    An agent is NOT a simple strategy - it's an intelligent system that:
    - Specializes in specific market regimes
    - Uses ML predictions to improve decisions
    - Can refuse trades if conditions aren't ideal
    - Learns from its results
    """

    # ATR multipliers for stop-loss by regime
    # INCREASED 2024-12-22: 100% of SL trades were in profit before being stopped
    # Need wider stops to avoid being stopped out by noise
    ATR_MULTIPLIERS = {
        MarketRegime.STRONG_UPTREND: 2.0,   # Was 1.5
        MarketRegime.WEAK_UPTREND: 2.5,     # Was 2.0
        MarketRegime.STRONG_DOWNTREND: 2.0, # Was 1.5
        MarketRegime.WEAK_DOWNTREND: 2.5,   # Was 2.0
        MarketRegime.RANGING: 2.0,          # Was 1.5
        MarketRegime.CHOPPY: 3.0,           # Was 2.5 - wider in chaos
        MarketRegime.COMPRESSION: 2.5,      # Was 2.0
        MarketRegime.BREAKOUT_UP: 2.0,      # Was 1.5
        MarketRegime.BREAKOUT_DOWN: 2.0,    # Was 1.5
    }

    def __init__(
        self,
        name: str,
        description: str,
        suitable_regimes: list[MarketRegime],
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
        min_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialize base agent.

        Args:
            name: Agent name
            description: Agent description
            suitable_regimes: List of regimes this agent handles
            predictor: Online predictor for ML predictions
            meta_learner: Meta learner for adaptive insights
            min_confidence_threshold: Minimum confidence to generate signal
        """
        self.name = name
        self.description = description
        self.suitable_regimes = suitable_regimes
        self.predictor = predictor
        self.meta_learner = meta_learner
        self.min_confidence_threshold = min_confidence_threshold

        # Initialize state
        self.state = AgentState(
            name=name,
            status="ACTIVE",
            suitable_regimes=suitable_regimes,
        )

        log.info(f"[{self.name}] Initialized: {description}")

    def is_suitable(self, regime: MarketRegime) -> bool:
        """Check if agent is suitable for given regime."""
        return regime in self.suitable_regimes

    @abstractmethod
    async def analyze(self, context: MarketContext) -> Signal | None:
        """
        Analyze market context and generate signal.

        Args:
            context: Complete market context for a symbol

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    def _calculate_stop_loss(
        self,
        side: str,
        entry: float,
        atr: float,
        regime: MarketRegime,
    ) -> float:
        """
        Calculate stop-loss based on ATR and regime.

        Args:
            side: "LONG" or "SHORT"
            entry: Entry price
            atr: ATR value
            regime: Current market regime

        Returns:
            Stop-loss price
        """
        multiplier = self.ATR_MULTIPLIERS.get(regime, 2.0)

        if side == "LONG":
            return entry - (atr * multiplier)
        else:
            return entry + (atr * multiplier)

    def _calculate_take_profits(
        self,
        side: str,
        entry: float,
        stop_loss: float,
        tp1_rr: float = 1.5,
        tp2_rr: float = 2.5,
    ) -> tuple[float, float]:
        """
        Calculate take-profit levels based on R:R.

        Args:
            side: "LONG" or "SHORT"
            entry: Entry price
            stop_loss: Stop-loss price
            tp1_rr: R:R for TP1
            tp2_rr: R:R for TP2

        Returns:
            Tuple of (TP1, TP2) prices
        """
        risk = abs(entry - stop_loss)

        if side == "LONG":
            tp1 = entry + (risk * tp1_rr)
            tp2 = entry + (risk * tp2_rr)
        else:
            tp1 = entry - (risk * tp1_rr)
            tp2 = entry - (risk * tp2_rr)

        return (tp1, tp2)

    def _get_ml_prediction(self, context: MarketContext, side: str) -> tuple[str, float]:
        """
        Get prediction from Online Predictor.

        Args:
            context: Market context
            side: Proposed trade side

        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.predictor:
            return ("NEUTRAL", 0.0)

        try:
            prediction = self.predictor.predict(context, side.lower())

            # Map prediction direction to FAVORABLE/UNFAVORABLE
            if side == "LONG":
                if prediction.predicted_direction == "up":
                    ml_result = "FAVORABLE"
                elif prediction.predicted_direction == "down":
                    ml_result = "UNFAVORABLE"
                else:
                    ml_result = "NEUTRAL"
            else:  # SHORT
                if prediction.predicted_direction == "down":
                    ml_result = "FAVORABLE"
                elif prediction.predicted_direction == "up":
                    ml_result = "UNFAVORABLE"
                else:
                    ml_result = "NEUTRAL"

            return (ml_result, prediction.confidence)

        except Exception as e:
            log.warning(f"[{self.name}] ML prediction failed: {e}")
            return ("NEUTRAL", 0.0)

    def _determine_position_size(
        self,
        confidence: float,
        ml_prediction: str,
        regime: MarketRegime,
    ) -> str:
        """
        Determine recommended position size.

        Args:
            confidence: Signal confidence
            ml_prediction: ML prediction result
            regime: Current market regime

        Returns:
            Size recommendation: "micro" / "small" / "normal" / "large"
        """
        size_order = ["micro", "small", "normal", "large"]

        # Base size by confidence
        if confidence >= 0.8:
            base = "large"
        elif confidence >= 0.7:
            base = "normal"
        elif confidence >= 0.6:
            base = "small"
        else:
            base = "micro"

        current_idx = size_order.index(base)

        # Downgrade if ML says UNFAVORABLE
        if ml_prediction == "UNFAVORABLE":
            current_idx = max(0, current_idx - 1)

        # Downgrade in dangerous regimes
        if regime in [MarketRegime.CHOPPY]:
            current_idx = max(0, current_idx - 1)

        return size_order[current_idx]

    def _create_signal(
        self,
        context: MarketContext,
        side: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        confidence: float,
        reasoning: list[str],
        warnings: list[str],
        ml_prediction: str,
        ml_confidence: float,
    ) -> Signal:
        """
        Create a Signal object.

        Args:
            context: Market context
            side: Trade side
            entry_price: Entry price
            stop_loss: Stop-loss price
            tp1: Take-profit 1
            tp2: Take-profit 2
            confidence: Signal confidence
            reasoning: List of reasons
            warnings: List of warnings
            ml_prediction: ML prediction
            ml_confidence: ML confidence

        Returns:
            Signal object
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(tp1 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        position_size = self._determine_position_size(
            confidence, ml_prediction, context.regime.regime
        )

        return Signal(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            symbol=context.symbol,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=round(rr_ratio, 2),
            position_size_recommendation=position_size,
            agent_name=self.name,
            regime_at_signal=context.regime.regime,
            reasoning=reasoning,
            warnings=warnings,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,
        )

    def record_result(
        self,
        signal_id: str,
        pnl_percent: float,
        outcome: int,  # 1 = win, 0 = loss
    ) -> None:
        """
        Record trade result for learning.

        Args:
            signal_id: Signal ID
            pnl_percent: PnL percentage
            outcome: 1 for win, 0 for loss
        """
        self.state.total_signals += 1
        self.state.signals_taken += 1

        if outcome == 1:
            self.state.wins += 1

        # Update win rate
        if self.state.signals_taken > 0:
            self.state.win_rate = self.state.wins / self.state.signals_taken

        # Update PnL tracking
        self.state.total_pnl_percent += pnl_percent
        self.state.avg_pnl_percent = (
            self.state.total_pnl_percent / self.state.signals_taken
        )
        self.state.last_trade_result = pnl_percent
        self.state.last_signal_time = datetime.now(timezone.utc)

        log.info(
            f"[{self.name}] Trade result: PnL={pnl_percent:+.2f}%, "
            f"Win rate={self.state.win_rate:.0%}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "name": self.name,
            "status": self.state.status,
            "total_signals": self.state.total_signals,
            "signals_taken": self.state.signals_taken,
            "win_rate": self.state.win_rate,
            "avg_pnl_percent": self.state.avg_pnl_percent,
            "suitable_regimes": [r.value for r in self.suitable_regimes],
        }

    def reset_stats(self) -> None:
        """Reset agent statistics."""
        self.state.total_signals = 0
        self.state.signals_taken = 0
        self.state.wins = 0
        self.state.win_rate = 0.0
        self.state.avg_pnl_percent = 0.0
        self.state.total_pnl_percent = 0.0
        self.state.last_signal_time = None
        self.state.last_trade_result = None
