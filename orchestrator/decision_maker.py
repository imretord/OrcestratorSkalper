"""
Rules-based Decision Maker for AI Trading System V3.
Provides fallback decision making when LLM is unavailable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import (
    Decision,
    MarketContext,
    MarketRegime,
    Signal,
)

log = get_logger("decision_maker")


class RulesDecisionMaker:
    """
    Rules-based decision maker.

    Uses predefined rules to make trading decisions.
    Acts as fallback when LLM is unavailable.
    """

    # Risk thresholds
    MAX_SIGNALS_PER_HOUR = 20  # Increased for micro mode learning
    MIN_CONFIDENCE = 0.6
    MAX_WARNINGS = 5  # Увеличено с 3 - учитываются только hard warnings

    # Soft warning patterns - информационные, не блокируют сигнал
    SOFT_WARNING_PATTERNS = [
        "compression",      # Информационный
        "choppy",           # Информационный
        "may fail",         # Вероятностный
        "consider",         # Рекомендательный
        "monitor",          # Рекомендательный
    ]

    # ML prediction blocking - if UNFAVORABLE, reduce confidence
    ML_UNFAVORABLE_CONFIDENCE_PENALTY = 0.15  # Снижаем confidence на 15%

    # Position limits
    MAX_EXPOSURE_USD = 1000.0
    POSITION_SIZES = {
        "micro": 50.0,
        "small": 100.0,
        "normal": 200.0,
        "large": 500.0,
    }

    # Regime-based adjustments
    REGIME_POSITION_MULTIPLIER = {
        MarketRegime.STRONG_UPTREND: 1.2,
        MarketRegime.WEAK_UPTREND: 1.0,
        MarketRegime.STRONG_DOWNTREND: 1.2,
        MarketRegime.WEAK_DOWNTREND: 1.0,
        MarketRegime.RANGING: 0.8,
        MarketRegime.CHOPPY: 0.5,
        MarketRegime.COMPRESSION: 0.6,
        MarketRegime.BREAKOUT_UP: 1.0,
        MarketRegime.BREAKOUT_DOWN: 1.0,
    }

    # Cooldown after stop-loss (in minutes)
    SL_COOLDOWN_MINUTES = 10

    def __init__(
        self,
        max_exposure_usd: float = 1000.0,
        min_confidence: float = 0.6,
        max_warnings: int = 3,
        position_size_mode: str = "fixed",
        position_size_pct: float = 2.0,
        leverage: int = 3,
    ) -> None:
        """
        Initialize rules-based decision maker.

        Args:
            max_exposure_usd: Maximum total exposure in USD
            min_confidence: Minimum signal confidence
            max_warnings: Maximum acceptable warnings on signal
            position_size_mode: "fixed" or "dynamic"
            position_size_pct: Position size as % of balance (when mode=fixed)
            leverage: Leverage multiplier for position sizing
        """
        self.max_exposure_usd = max_exposure_usd
        self.min_confidence = min_confidence
        self.max_warnings = max_warnings
        self.position_size_mode = position_size_mode
        self.position_size_pct = position_size_pct
        self.leverage = leverage

        # Track recent decisions
        self._recent_signals: list[datetime] = []
        self._current_exposure: float = 0.0

        # Aggressive mode tracking
        self.trades_today = 0
        self.last_trade_time: datetime | None = None
        self.aggressive_mode = False

        # Stop-loss cooldown tracking: symbol -> last SL time
        self._sl_cooldowns: dict[str, datetime] = {}

        log.info(
            f"[RulesDecisionMaker] Initialized with max_exposure={max_exposure_usd}, "
            f"min_confidence={min_confidence}, size_mode={position_size_mode}, "
            f"size_pct={position_size_pct}%, leverage={leverage}x"
        )

    def make_decision(
        self,
        signals: list[Signal],
        context: MarketContext,
        current_positions: list[dict[str, Any]],
        account_balance: float,
    ) -> Decision:
        """
        Make trading decision based on rules.

        Args:
            signals: List of signals from agents
            context: Current market context
            current_positions: List of open positions
            account_balance: Current account balance in USD

        Returns:
            Decision object with action and reasoning
        """
        # Calculate current exposure
        self._current_exposure = sum(
            abs(pos.get("contracts", 0) * pos.get("mark_price", 0))
            for pos in current_positions
        )

        # Clean old signals (keep last hour)
        now = datetime.now(timezone.utc)
        self._recent_signals = [
            ts for ts in self._recent_signals
            if (now - ts).total_seconds() < 3600
        ]

        # Check for emergency conditions
        emergency = self._check_emergency_conditions(context, current_positions)
        if emergency:
            return emergency

        # If no signals, wait
        if not signals:
            return Decision(
                timestamp=now,
                action="WAIT",
                reasoning="No signals from agents",
                key_factors=["No trading opportunities detected"],
                risks_identified=[],
                decision_source="RULES",
                confidence=1.0,
            )

        # Check and update aggressive mode
        self._check_aggressive_mode()

        # Determine thresholds based on mode
        min_confidence = 0.55 if self.aggressive_mode else self.min_confidence
        max_warnings = 6 if self.aggressive_mode else self.max_warnings

        if self.aggressive_mode:
            log.info("[DECISION] Aggressive mode ACTIVE - relaxed thresholds")

        # Filter signals by confidence, hard warnings, AND ML prediction
        valid_signals = []
        for s in signals:
            # Calculate effective confidence (penalize UNFAVORABLE ML prediction)
            effective_confidence = s.confidence
            if s.ml_prediction == "UNFAVORABLE":
                effective_confidence -= self.ML_UNFAVORABLE_CONFIDENCE_PENALTY
                log.info(
                    f"[FILTER] {s.symbol}: ML UNFAVORABLE penalty applied "
                    f"({s.confidence:.2f} → {effective_confidence:.2f})"
                )

            hard_warnings = self._count_hard_warnings(s.warnings)

            if effective_confidence >= min_confidence and hard_warnings <= max_warnings:
                valid_signals.append(s)
            else:
                # Log rejection reason
                if effective_confidence < min_confidence:
                    ml_note = " (ML penalty)" if s.ml_prediction == "UNFAVORABLE" else ""
                    log.info(
                        f"[FILTER] {s.symbol} rejected: effective confidence "
                        f"{effective_confidence:.2f} < {min_confidence}{ml_note}"
                    )
                elif hard_warnings > max_warnings:
                    log.info(
                        f"[FILTER] {s.symbol} rejected: {hard_warnings} hard warnings > {max_warnings}"
                    )
                    log.info(f"         Warnings: {s.warnings}")

        if not valid_signals:
            return Decision(
                timestamp=now,
                action="WAIT",
                reasoning="No signals meet quality criteria",
                key_factors=[
                    f"All signals below {self.min_confidence} confidence or have too many warnings",
                ],
                risks_identified=[s.warnings[0] if s.warnings else "Low confidence" for s in signals[:3]],
                decision_source="RULES",
                confidence=0.8,
            )

        # Get best signal
        best_signal = valid_signals[0]

        # Check rate limit
        if len(self._recent_signals) >= self.MAX_SIGNALS_PER_HOUR:
            return Decision(
                timestamp=now,
                action="WAIT",
                signal=best_signal,
                reasoning="Rate limit reached - too many signals this hour",
                key_factors=[
                    f"Already processed {len(self._recent_signals)} signals this hour",
                ],
                risks_identified=["Overtrading risk"],
                decision_source="RULES",
                confidence=0.9,
            )

        # Check exposure limit
        position_size = self._calculate_position_size(best_signal, context, account_balance)

        if self._current_exposure + position_size > self.max_exposure_usd:
            return Decision(
                timestamp=now,
                action="WAIT",
                signal=best_signal,
                reasoning="Exposure limit would be exceeded",
                key_factors=[
                    f"Current exposure: ${self._current_exposure:.2f}",
                    f"New position: ${position_size:.2f}",
                    f"Max allowed: ${self.max_exposure_usd:.2f}",
                ],
                risks_identified=["Overexposure risk"],
                decision_source="RULES",
                confidence=0.9,
            )

        # Check for conflicting positions
        conflict = self._check_position_conflict(best_signal, current_positions)
        if conflict:
            return Decision(
                timestamp=now,
                action="WAIT",
                signal=best_signal,
                reasoning=f"Signal conflicts with existing position: {conflict}",
                key_factors=[f"Existing position: {conflict}"],
                risks_identified=["Position conflict"],
                decision_source="RULES",
                confidence=0.85,
            )

        # Check for SL cooldown
        cooldown_remaining = self._check_sl_cooldown(best_signal.symbol)
        if cooldown_remaining > 0:
            return Decision(
                timestamp=now,
                action="WAIT",
                signal=best_signal,
                reasoning=f"Symbol in cooldown after stop-loss ({cooldown_remaining:.0f}m remaining)",
                key_factors=[
                    f"Recent SL on {best_signal.symbol}",
                    f"Cooldown: {self.SL_COOLDOWN_MINUTES} minutes",
                ],
                risks_identified=["Avoid re-entry after SL"],
                decision_source="RULES",
                confidence=0.95,
            )

        # All checks passed - take the trade
        self._recent_signals.append(now)

        return Decision(
            timestamp=now,
            action="TRADE",
            signal=best_signal,
            position_size_usd=position_size,
            reasoning=self._build_reasoning(best_signal, context),
            key_factors=best_signal.reasoning[:5],
            risks_identified=best_signal.warnings[:3],
            decision_source="RULES",
            confidence=best_signal.confidence,
        )

    def _check_emergency_conditions(
        self,
        context: MarketContext,
        positions: list[dict[str, Any]],
    ) -> Decision | None:
        """
        Check for emergency conditions that require immediate action.

        Args:
            context: Market context
            positions: Current positions

        Returns:
            Emergency decision or None
        """
        now = datetime.now(timezone.utc)

        # Check for extreme fear/greed
        fg = context.sentiment.fear_greed_index

        # Extreme fear with positions - consider reducing
        if fg < 10 and positions:
            long_positions = [p for p in positions if p.get("side") == "long"]
            if long_positions:
                return Decision(
                    timestamp=now,
                    action="REDUCE_EXPOSURE",
                    reasoning="Extreme Fear detected - reduce long exposure",
                    key_factors=[
                        f"Fear & Greed Index: {fg}",
                        f"Long positions at risk",
                    ],
                    risks_identified=["Potential capitulation", "Black swan risk"],
                    decision_source="EMERGENCY",
                    confidence=0.9,
                )

        # Extreme greed with positions - consider reducing
        if fg > 95 and positions:
            long_positions = [p for p in positions if p.get("side") == "long"]
            if long_positions:
                return Decision(
                    timestamp=now,
                    action="REDUCE_EXPOSURE",
                    reasoning="Extreme Greed detected - take profits",
                    key_factors=[
                        f"Fear & Greed Index: {fg}",
                        f"Market possibly overheated",
                    ],
                    risks_identified=["Blow-off top risk", "Sharp correction possible"],
                    decision_source="EMERGENCY",
                    confidence=0.85,
                )

        # Extreme volatility
        if context.regime.volatility_level == "extreme":
            if positions:
                total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
                if total_pnl < 0:
                    return Decision(
                        timestamp=now,
                        action="CLOSE_ALL",
                        reasoning="Extreme volatility with negative PnL - exit all",
                        key_factors=[
                            "Extreme volatility detected",
                            f"Unrealized PnL: ${total_pnl:.2f}",
                        ],
                        risks_identified=["Volatility spike", "Stop hunt risk"],
                        decision_source="EMERGENCY",
                        confidence=0.95,
                    )

        return None

    def _calculate_position_size(
        self,
        signal: Signal,
        context: MarketContext,
        account_balance: float,
    ) -> float:
        """
        Calculate position size based on signal and context.

        Args:
            signal: Trading signal
            context: Market context
            account_balance: Account balance

        Returns:
            Position size in USD
        """
        # Fixed mode: margin = % of balance, position = margin × leverage
        if self.position_size_mode == "fixed":
            margin = account_balance * (self.position_size_pct / 100)
            size = margin * self.leverage
            log.debug(
                f"[PositionSize] Fixed mode: {self.position_size_pct}% margin × {self.leverage}x = "
                f"${margin:.2f} margin → ${size:.2f} position"
            )
        else:
            # Dynamic mode: based on signal, regime, confidence
            base_size = self.POSITION_SIZES.get(
                signal.position_size_recommendation,
                self.POSITION_SIZES["small"]
            )

            # Apply regime multiplier
            regime_mult = self.REGIME_POSITION_MULTIPLIER.get(
                context.regime.regime,
                1.0
            )

            # Apply confidence multiplier
            conf_mult = 0.5 + (signal.confidence * 0.5)  # 0.5x to 1.0x

            size = base_size * regime_mult * conf_mult
            log.debug(
                f"[PositionSize] Dynamic mode: base=${base_size} × regime={regime_mult} × conf={conf_mult:.2f} = ${size:.2f}"
            )

        # Cap at max exposure and account limits
        max_from_exposure = self.max_exposure_usd - self._current_exposure
        max_from_balance = account_balance * 0.2  # Max 20% of balance per position

        final_size = min(size, max_from_exposure, max_from_balance)

        if final_size < size:
            log.debug(f"[PositionSize] Capped from ${size:.2f} to ${final_size:.2f}")

        return final_size

    def _check_position_conflict(
        self,
        signal: Signal,
        positions: list[dict[str, Any]],
    ) -> str | None:
        """
        Check if signal conflicts with existing positions.

        Args:
            signal: New signal
            positions: Current positions

        Returns:
            Conflict description or None
        """
        for pos in positions:
            if pos.get("symbol") == signal.symbol:
                existing_side = pos.get("side", "").upper()
                if existing_side == signal.side:
                    return f"Already {existing_side} on {signal.symbol}"
                else:
                    return f"Opposite position ({existing_side}) on {signal.symbol}"

        return None

    def _build_reasoning(self, signal: Signal, context: MarketContext) -> str:
        """Build reasoning string for decision."""
        parts = [
            f"Taking {signal.side} on {signal.symbol}",
            f"Confidence: {signal.confidence:.0%}",
            f"R:R = {signal.risk_reward_ratio:.1f}",
            f"Agent: {signal.agent_name}",
            f"Regime: {context.regime.regime.value}",
        ]

        if signal.ml_prediction == "FAVORABLE":
            parts.append("ML: Favorable")

        return ". ".join(parts)

    def _count_hard_warnings(self, warnings: list[str]) -> int:
        """
        Count only serious (hard) warnings, ignoring informational ones.

        Args:
            warnings: List of warning strings

        Returns:
            Count of hard warnings
        """
        hard_count = 0
        for w in warnings:
            w_lower = w.lower()
            is_soft = any(pattern in w_lower for pattern in self.SOFT_WARNING_PATTERNS)
            if not is_soft:
                hard_count += 1

        return hard_count

    def _check_aggressive_mode(self) -> None:
        """Enable aggressive mode if no trades for too long."""
        hours_since_last = 0
        if self.last_trade_time:
            hours_since_last = (
                datetime.now(timezone.utc) - self.last_trade_time
            ).total_seconds() / 3600

        # If > 12 hours without trades — aggressive mode
        # Or if 0 trades today and > 6 hours since last
        old_mode = self.aggressive_mode

        if hours_since_last > 12 or (self.trades_today == 0 and hours_since_last > 6):
            self.aggressive_mode = True
        else:
            self.aggressive_mode = False

        if self.aggressive_mode and not old_mode:
            log.info(
                f"[DECISION] Aggressive mode ON - no trades for {hours_since_last:.1f}h"
            )
        elif not self.aggressive_mode and old_mode:
            log.info("[DECISION] Aggressive mode OFF - normal trading resumed")

    def record_trade(self) -> None:
        """Record that a trade was made (call after successful trade)."""
        self.last_trade_time = datetime.now(timezone.utc)
        self.trades_today += 1
        log.info(f"[DECISION] Trade recorded. Total today: {self.trades_today}")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new trading day)."""
        self.trades_today = 0
        log.info("[DECISION] Daily stats reset")

    def update_exposure(self, exposure: float) -> None:
        """Update current exposure tracking."""
        self._current_exposure = exposure

    def _check_sl_cooldown(self, symbol: str) -> float:
        """
        Check if symbol is in cooldown after stop-loss.

        Args:
            symbol: Trading symbol

        Returns:
            Remaining cooldown in minutes (0 if not in cooldown)
        """
        if symbol not in self._sl_cooldowns:
            return 0

        sl_time = self._sl_cooldowns[symbol]
        elapsed_minutes = (datetime.now(timezone.utc) - sl_time).total_seconds() / 60

        if elapsed_minutes >= self.SL_COOLDOWN_MINUTES:
            # Cooldown expired, remove entry
            del self._sl_cooldowns[symbol]
            return 0

        return self.SL_COOLDOWN_MINUTES - elapsed_minutes

    def record_stop_loss(self, symbol: str) -> None:
        """
        Record that a position was closed by stop-loss.
        Starts cooldown timer for this symbol.

        Args:
            symbol: Trading symbol that hit SL
        """
        self._sl_cooldowns[symbol] = datetime.now(timezone.utc)
        log.info(
            f"[RulesDecisionMaker] SL recorded for {symbol} - "
            f"cooldown {self.SL_COOLDOWN_MINUTES}m started"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get decision maker statistics."""
        # Calculate active cooldowns
        active_cooldowns = {}
        for symbol, sl_time in list(self._sl_cooldowns.items()):
            remaining = self._check_sl_cooldown(symbol)
            if remaining > 0:
                active_cooldowns[symbol] = round(remaining, 1)

        return {
            "recent_signals_count": len(self._recent_signals),
            "current_exposure": self._current_exposure,
            "max_exposure": self.max_exposure_usd,
            "min_confidence": self.min_confidence,
            "aggressive_mode": self.aggressive_mode,
            "trades_today": self.trades_today,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "sl_cooldowns": active_cooldowns,
        }
