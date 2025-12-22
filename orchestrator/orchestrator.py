"""
Main Orchestrator for AI Trading System V3.
The brain of the trading system - coordinates all components.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import (
    Decision,
    MarketContext,
    Signal,
)
from agents.agent_manager import AgentManager
from learners.online_predictor import OnlinePredictor
from learners.meta_learner import MetaLearner
from learners.experience_buffer import ExperienceBuffer
from orchestrator.decision_maker import RulesDecisionMaker
from orchestrator.llm_decision_maker import LLMDecisionMaker
from orchestrator.risk_guardian import RiskGuardian

log = get_logger("orchestrator")


class Orchestrator:
    """
    Main orchestrator for the trading system.

    Coordinates:
    - Agent Manager (signal generation)
    - Decision Makers (LLM + Rules fallback)
    - Risk Guardian (final validation)
    - Trade Execution
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        predictor: OnlinePredictor | None = None,
        meta_learner: MetaLearner | None = None,
        anthropic_api_key: str | None = None,
        use_llm: bool = True,
        max_exposure_usd: float = 1000.0,
        llm_model: str | None = None,
        position_size_mode: str = "fixed",
        position_size_pct: float = 2.0,
        leverage: int = 3,
    ) -> None:
        """
        Initialize Orchestrator.

        Args:
            experience_buffer: Buffer for trade experiences
            predictor: Online predictor instance
            meta_learner: Meta learner instance
            anthropic_api_key: API key for Claude (if using LLM)
            use_llm: Whether to use LLM for decisions
            max_exposure_usd: Maximum exposure in USD
            llm_model: LLM model to use
            position_size_mode: "fixed" or "dynamic"
            position_size_pct: Position size as % of balance (when mode=fixed)
            leverage: Leverage multiplier for position sizing
        """
        self.use_llm = use_llm and anthropic_api_key is not None
        self.max_exposure_usd = max_exposure_usd

        # Initialize components
        self.experience_buffer = experience_buffer

        # Create or use provided predictor
        self.predictor = predictor or OnlinePredictor(
            experience_buffer=experience_buffer,
            model_type="ensemble",
        )

        # Create or use provided meta learner
        self.meta_learner = meta_learner or MetaLearner(
            experience_buffer=experience_buffer,
            online_predictor=self.predictor,
        )

        # Initialize Agent Manager
        self.agent_manager = AgentManager(
            predictor=self.predictor,
            meta_learner=self.meta_learner,
        )

        # Initialize Decision Makers
        self.rules_decision_maker = RulesDecisionMaker(
            max_exposure_usd=max_exposure_usd,
            position_size_mode=position_size_mode,
            position_size_pct=position_size_pct,
            leverage=leverage,
        )

        self.llm_decision_maker: LLMDecisionMaker | None = None
        if self.use_llm and anthropic_api_key:
            self.llm_decision_maker = LLMDecisionMaker(
                api_key=anthropic_api_key,
                model=llm_model,
            )
            log.info("[Orchestrator] LLM decision maker enabled")
        else:
            log.info("[Orchestrator] Using rules-based decision maker only")

        # Initialize Risk Guardian
        self.risk_guardian = RiskGuardian(
            max_exposure_usd=max_exposure_usd,
        )

        # State tracking
        self._decisions_made = 0
        self._trades_executed = 0
        self._llm_decisions = 0
        self._rules_decisions = 0

        # Active signals and decisions
        self._last_decision: Decision | None = None
        self._active_signals: dict[str, list[Signal]] = {}

        log.info(
            f"[Orchestrator] Initialized - LLM: {self.use_llm}, "
            f"Max exposure: ${max_exposure_usd}"
        )

    async def process_context(
        self,
        context: MarketContext,
        current_positions: list[dict[str, Any]],
        account_balance: float,
        account_equity: float | None = None,
    ) -> Decision:
        """
        Process market context and make trading decision.

        Args:
            context: Market context from analyzers
            current_positions: Current open positions
            account_balance: Current wallet balance (for position sizing)
            account_equity: Total equity including unrealized PnL (for drawdown)

        Returns:
            Trading decision
        """
        log.info(
            f"[Orchestrator] Processing {context.symbol} - "
            f"Regime: {context.regime.regime.value}, "
            f"Price: ${context.current_price:.2f}"
        )

        # Step 1: Collect signals from agents
        signals = await self.agent_manager.collect_signals(context)
        self._active_signals[context.symbol] = signals

        log.info(
            f"[Orchestrator] Collected {len(signals)} signals for {context.symbol}"
        )

        # Step 2: Make decision (LLM first, fallback to rules)
        decision = await self._make_decision(
            signals=signals,
            context=context,
            positions=current_positions,
            balance=account_balance,
        )

        # Step 3: Apply Risk Guardian validation
        decision = self._apply_risk_guardian(
            decision, context, current_positions, account_balance, account_equity
        )

        # Track decision
        self._decisions_made += 1
        self._last_decision = decision

        if decision.decision_source == "LLM":
            self._llm_decisions += 1
        else:
            self._rules_decisions += 1

        log.info(
            f"[Orchestrator] Decision: {decision.action} "
            f"(source: {decision.decision_source}, conf: {decision.confidence:.2f})"
        )

        return decision

    async def process_multiple_contexts(
        self,
        contexts: list[MarketContext],
        current_positions: list[dict[str, Any]],
        account_balance: float,
    ) -> list[Decision]:
        """
        Process multiple market contexts and return decisions.

        Args:
            contexts: List of market contexts
            current_positions: Current positions
            account_balance: Account balance

        Returns:
            List of decisions (one per context)
        """
        decisions = []

        for context in contexts:
            decision = await self.process_context(
                context=context,
                current_positions=current_positions,
                account_balance=account_balance,
            )
            decisions.append(decision)

            # If we decided to trade, update exposure tracking
            if decision.action == "TRADE" and decision.position_size_usd:
                self.rules_decision_maker.update_exposure(
                    self.rules_decision_maker._current_exposure + decision.position_size_usd
                )

        return decisions

    async def _make_decision(
        self,
        signals: list[Signal],
        context: MarketContext,
        positions: list[dict[str, Any]],
        balance: float,
    ) -> Decision:
        """
        Make decision using LLM or fallback to rules.

        Args:
            signals: Available signals
            context: Market context
            positions: Current positions
            balance: Account balance

        Returns:
            Trading decision
        """
        decision = None

        # Try LLM first if enabled
        if self.llm_decision_maker and self.use_llm:
            try:
                decision = await self.llm_decision_maker.make_decision(
                    signals=signals,
                    context=context,
                    current_positions=positions,
                    account_balance=balance,
                    agent_stats=self.agent_manager.get_agent_stats(),
                )
            except Exception as e:
                log.error(f"[Orchestrator] LLM decision failed: {e}")
                decision = None

        # Fallback to rules if LLM failed or disabled
        if decision is None:
            log.info("[Orchestrator] Using rules-based decision maker")
            decision = self.rules_decision_maker.make_decision(
                signals=signals,
                context=context,
                current_positions=positions,
                account_balance=balance,
            )

        return decision

    def _apply_risk_guardian(
        self,
        decision: Decision,
        context: MarketContext,
        positions: list[dict[str, Any]],
        account_balance: float,
        account_equity: float | None = None,
    ) -> Decision:
        """
        Apply Risk Guardian validation to decision.

        Args:
            decision: Proposed decision
            context: Market context
            positions: Current positions
            account_balance: Current wallet balance (for position sizing)
            account_equity: Total equity including unrealized PnL (for drawdown)

        Returns:
            Validated/modified decision
        """
        # First check for emergency conditions
        emergency = self.risk_guardian.check_emergency_conditions(context, positions)
        if emergency:
            return emergency

        # Validate the decision
        return self.risk_guardian.validate_decision(
            decision=decision,
            context=context,
            current_positions=positions,
            account_balance=account_balance,
            account_equity=account_equity,
        )

    def record_trade_result(
        self,
        signal: Signal,
        pnl_percent: float,
        outcome: int,
    ) -> None:
        """
        Record trade result for learning.

        Args:
            signal: Original signal
            pnl_percent: PnL percentage
            outcome: 1 for win, 0 for loss
        """
        # Record in agent manager
        self.agent_manager.record_trade_result(
            signal_id=signal.id,
            agent_name=signal.agent_name,
            pnl_percent=pnl_percent,
            outcome=outcome,
        )

        # Update risk guardian PnL tracking
        self.risk_guardian.record_pnl(pnl_percent)

        # Track execution
        self._trades_executed += 1

        log.info(
            f"[Orchestrator] Recorded trade result: "
            f"{signal.side} {signal.symbol} = {pnl_percent:+.2f}%"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "decisions_made": self._decisions_made,
            "trades_executed": self._trades_executed,
            "llm_decisions": self._llm_decisions,
            "rules_decisions": self._rules_decisions,
            "llm_enabled": self.use_llm,
            "max_exposure_usd": self.max_exposure_usd,
            "agent_stats": self.agent_manager.get_agent_stats(),
            "rules_maker_stats": self.rules_decision_maker.get_stats(),
            "risk_guardian_stats": self.risk_guardian.get_stats(),
        }

        if self.llm_decision_maker:
            stats["llm_maker_stats"] = self.llm_decision_maker.get_stats()

        return stats

    def get_last_decision(self) -> Decision | None:
        """Get the last decision made."""
        return self._last_decision

    def get_active_signals(self, symbol: str) -> list[Signal]:
        """Get active signals for a symbol."""
        return self._active_signals.get(symbol, [])

    def get_regime_coverage(self) -> dict[str, list[str]]:
        """Get agent coverage by regime."""
        return self.agent_manager.get_regime_coverage()

    async def close(self) -> None:
        """Clean up resources."""
        if self.llm_decision_maker:
            await self.llm_decision_maker.close()

        log.info("[Orchestrator] Closed")
