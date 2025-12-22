"""
Agent Manager for AI Trading System V3.
Coordinates all trading agents and collects their signals.
"""
from __future__ import annotations

import asyncio
from typing import Any

from core.logger import get_logger
from core.state import (
    MarketContext,
    MarketRegime,
    Signal,
)
from agents.base_agent import BaseAgent
from agents.trend_follower import TrendFollowerAgent
from agents.mean_reversion import MeanReversionAgent
from agents.breakout_catcher import BreakoutCatcherAgent
from learners.meta_learner import MetaLearner
from learners.online_predictor import OnlinePredictor

log = get_logger("agent_manager")


class AgentManager:
    """
    Manages all trading agents.

    Responsibilities:
    - Initialize and configure agents
    - Route market context to suitable agents
    - Collect and rank signals
    - Track agent performance
    - Handle agent lifecycle
    """

    def __init__(
        self,
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
    ) -> None:
        """
        Initialize Agent Manager.

        Args:
            predictor: Shared online predictor for all agents
            meta_learner: Shared meta learner for all agents
        """
        self.predictor = predictor
        self.meta_learner = meta_learner

        # Initialize all agents
        self.agents: dict[str, BaseAgent] = {}
        self._init_agents()

        # Track active signals
        self._active_signals: dict[str, Signal] = {}

        log.info(f"[AgentManager] Initialized with {len(self.agents)} agents")

    def _init_agents(self) -> None:
        """Initialize all trading agents."""
        # TrendFollower agent
        self.agents["TrendFollower"] = TrendFollowerAgent(
            predictor=self.predictor,
            meta_learner=self.meta_learner,
        )

        # MeanReversion agent
        self.agents["MeanReversion"] = MeanReversionAgent(
            predictor=self.predictor,
            meta_learner=self.meta_learner,
        )

        # BreakoutCatcher agent
        self.agents["BreakoutCatcher"] = BreakoutCatcherAgent(
            predictor=self.predictor,
            meta_learner=self.meta_learner,
        )

        log.info(
            f"[AgentManager] Agents initialized: {list(self.agents.keys())}"
        )

    def get_suitable_agents(self, regime: MarketRegime) -> list[BaseAgent]:
        """
        Get agents suitable for the current market regime.

        Args:
            regime: Current market regime

        Returns:
            List of suitable agents
        """
        suitable = [
            agent for agent in self.agents.values()
            if agent.is_suitable(regime)
        ]

        log.debug(
            f"[AgentManager] Suitable agents for {regime.value}: "
            f"{[a.name for a in suitable]}"
        )

        return suitable

    async def collect_signals(
        self,
        context: MarketContext,
    ) -> list[Signal]:
        """
        Collect signals from all suitable agents.

        Args:
            context: Market context to analyze

        Returns:
            List of signals from all agents, sorted by confidence
        """
        regime = context.regime.regime
        suitable_agents = self.get_suitable_agents(regime)

        if not suitable_agents:
            log.warning(
                f"[AgentManager] No agents suitable for regime: {regime.value}"
            )
            return []

        # Run all suitable agents concurrently
        tasks = [
            agent.analyze(context)
            for agent in suitable_agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid signals
        signals: list[Signal] = []
        for agent, result in zip(suitable_agents, results):
            if isinstance(result, Exception):
                log.error(f"[AgentManager] {agent.name} error: {result}")
                continue
            if result is not None:
                signals.append(result)
                log.info(
                    f"[AgentManager] {agent.name} generated signal: "
                    f"{result.side} {result.symbol} @ {result.confidence:.2f}"
                )

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        log.info(
            f"[AgentManager] Collected {len(signals)} signals for {context.symbol}"
        )

        return signals

    async def analyze_all_symbols(
        self,
        contexts: list[MarketContext],
    ) -> dict[str, list[Signal]]:
        """
        Analyze multiple symbols concurrently.

        Args:
            contexts: List of market contexts for different symbols

        Returns:
            Dict of symbol -> signals
        """
        tasks = [
            self.collect_signals(context)
            for context in contexts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_signals: dict[str, list[Signal]] = {}
        for context, result in zip(contexts, results):
            if isinstance(result, Exception):
                log.error(f"[AgentManager] Error analyzing {context.symbol}: {result}")
                all_signals[context.symbol] = []
            else:
                all_signals[context.symbol] = result

        # Log summary
        total_signals = sum(len(sigs) for sigs in all_signals.values())
        log.info(
            f"[AgentManager] Analyzed {len(contexts)} symbols, "
            f"generated {total_signals} total signals"
        )

        return all_signals

    def get_best_signal(
        self,
        signals: list[Signal],
        max_warnings: int = 3,
    ) -> Signal | None:
        """
        Get the best signal from a list of signals.

        Args:
            signals: List of signals to choose from
            max_warnings: Maximum acceptable warnings

        Returns:
            Best signal or None
        """
        if not signals:
            return None

        # Filter by warnings
        filtered = [s for s in signals if len(s.warnings) <= max_warnings]

        if not filtered:
            log.warning(
                f"[AgentManager] All signals have too many warnings"
            )
            # Return best signal anyway with warning
            return signals[0]

        # Return highest confidence signal
        return filtered[0]

    def record_trade_result(
        self,
        signal_id: str,
        agent_name: str,
        pnl_percent: float,
        outcome: int,
    ) -> None:
        """
        Record trade result for agent learning.

        Args:
            signal_id: Signal ID
            agent_name: Name of the agent that generated the signal
            pnl_percent: PnL percentage
            outcome: 1 for win, 0 for loss
        """
        if agent_name not in self.agents:
            log.warning(f"[AgentManager] Unknown agent: {agent_name}")
            return

        self.agents[agent_name].record_result(signal_id, pnl_percent, outcome)

        log.info(
            f"[AgentManager] Recorded result for {agent_name}: "
            f"PnL={pnl_percent:+.2f}%"
        )

    def get_agent_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all agents.

        Returns:
            Dict of agent_name -> stats
        """
        return {
            name: agent.get_stats()
            for name, agent in self.agents.items()
        }

    def get_agent(self, name: str) -> BaseAgent | None:
        """
        Get agent by name.

        Args:
            name: Agent name

        Returns:
            Agent or None
        """
        return self.agents.get(name)

    def disable_agent(self, name: str) -> bool:
        """
        Disable an agent.

        Args:
            name: Agent name

        Returns:
            True if disabled, False if not found
        """
        if name not in self.agents:
            return False

        self.agents[name].state.status = "DISABLED"
        log.info(f"[AgentManager] Disabled agent: {name}")
        return True

    def enable_agent(self, name: str) -> bool:
        """
        Enable an agent.

        Args:
            name: Agent name

        Returns:
            True if enabled, False if not found
        """
        if name not in self.agents:
            return False

        self.agents[name].state.status = "ACTIVE"
        log.info(f"[AgentManager] Enabled agent: {name}")
        return True

    def reset_all_stats(self) -> None:
        """Reset statistics for all agents."""
        for agent in self.agents.values():
            agent.reset_stats()

        log.info("[AgentManager] Reset all agent statistics")

    def get_regime_coverage(self) -> dict[str, list[str]]:
        """
        Get which agents cover which regimes.

        Returns:
            Dict of regime -> list of agent names
        """
        coverage: dict[str, list[str]] = {}

        for regime in MarketRegime:
            suitable = self.get_suitable_agents(regime)
            coverage[regime.value] = [a.name for a in suitable]

        return coverage
