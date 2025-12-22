"""
Risk Guardian for AI Trading System V3.
Final validation layer before trade execution.
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

log = get_logger("risk_guardian")


class RiskGuardian:
    """
    Risk Guardian - final validation before trade execution.

    Responsibilities:
    - Position size validation and adjustment
    - Exposure limit enforcement
    - Volatility-based position scaling
    - ML prediction alignment
    - Emergency stop conditions
    """

    # Default risk parameters
    DEFAULT_MAX_EXPOSURE_USD = 1000.0
    DEFAULT_MAX_POSITION_PCT = 0.2  # Max 20% of balance per position
    DEFAULT_MIN_RR_RATIO = 1.0

    # Volatility scaling
    VOLATILITY_SCALE = {
        "low": 1.2,
        "medium": 1.0,
        "high": 0.7,
        "extreme": 0.4,
    }

    # Regime risk multipliers
    REGIME_RISK = {
        MarketRegime.STRONG_UPTREND: 1.0,
        MarketRegime.WEAK_UPTREND: 0.9,
        MarketRegime.STRONG_DOWNTREND: 1.0,
        MarketRegime.WEAK_DOWNTREND: 0.9,
        MarketRegime.RANGING: 0.7,
        MarketRegime.CHOPPY: 0.5,
        MarketRegime.COMPRESSION: 0.6,
        MarketRegime.BREAKOUT_UP: 0.8,
        MarketRegime.BREAKOUT_DOWN: 0.8,
    }

    def __init__(
        self,
        max_exposure_usd: float = DEFAULT_MAX_EXPOSURE_USD,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        min_rr_ratio: float = DEFAULT_MIN_RR_RATIO,
        max_daily_loss_pct: float = 5.0,
        max_drawdown_pct: float = 10.0,
    ) -> None:
        """
        Initialize Risk Guardian.

        Args:
            max_exposure_usd: Maximum total exposure in USD
            max_position_pct: Maximum position size as % of balance
            min_rr_ratio: Minimum risk/reward ratio
            max_daily_loss_pct: Maximum daily loss percentage
            max_drawdown_pct: Maximum drawdown percentage
        """
        self.max_exposure_usd = max_exposure_usd
        self.max_position_pct = max_position_pct
        self.min_rr_ratio = min_rr_ratio
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        # Track state
        self._daily_pnl: float = 0.0
        self._peak_balance: float = 0.0
        self._trades_blocked: int = 0
        self._trades_modified: int = 0

        log.info(
            f"[RiskGuardian] Initialized - max_exposure=${max_exposure_usd}, "
            f"max_position={max_position_pct*100}%, min_rr={min_rr_ratio}"
        )

    def validate_decision(
        self,
        decision: Decision,
        context: MarketContext,
        current_positions: list[dict[str, Any]],
        account_balance: float,
        account_equity: float | None = None,
    ) -> Decision:
        """
        Validate and potentially modify a trading decision.

        Args:
            decision: Decision from orchestrator
            context: Market context
            current_positions: Current open positions
            account_balance: Current wallet balance (for position sizing)
            account_equity: Total equity including unrealized PnL (for drawdown)
                           If None, uses account_balance

        Returns:
            Validated/modified decision
        """
        # Skip non-trade decisions
        if decision.action != "TRADE":
            return decision

        if not decision.signal:
            return decision

        signal = decision.signal
        modifications: list[str] = []
        warnings: list[str] = list(decision.risks_identified)

        # Use equity for drawdown tracking (includes unrealized PnL)
        equity = account_equity if account_equity is not None else account_balance

        # Update peak equity for drawdown tracking
        if equity > self._peak_balance:
            self._peak_balance = equity

        # Check 1: Daily loss limit
        if self._check_daily_loss_limit():
            return self._block_decision(
                decision,
                f"Daily loss limit reached ({self._daily_pnl:.2f}%)",
            )

        # Check 2: Drawdown limit (based on equity, not wallet balance)
        if self._check_drawdown_limit(equity):
            return self._block_decision(
                decision,
                f"Drawdown limit reached",
            )

        # Check 3: R:R ratio
        if signal.risk_reward_ratio < self.min_rr_ratio:
            warnings.append(
                f"R:R ratio below minimum ({signal.risk_reward_ratio:.2f} < {self.min_rr_ratio})"
            )
            # Could block here, but let's warn and reduce size
            if decision.position_size_usd:
                decision.position_size_usd *= 0.5
                modifications.append("Size halved due to low R:R")

        # Check 4: Calculate current exposure
        current_exposure = sum(
            abs(p.get("contracts", 0) * p.get("mark_price", 0))
            for p in current_positions
        )

        # Check 5: Apply volatility scaling
        vol_scale = self.VOLATILITY_SCALE.get(
            context.regime.volatility_level, 1.0
        )
        if vol_scale < 1.0 and decision.position_size_usd:
            original = decision.position_size_usd
            decision.position_size_usd *= vol_scale
            modifications.append(
                f"Size reduced {(1-vol_scale)*100:.0f}% for {context.regime.volatility_level} volatility"
            )

        # Check 6: Apply regime risk scaling
        regime_scale = self.REGIME_RISK.get(context.regime.regime, 1.0)
        if regime_scale < 1.0 and decision.position_size_usd:
            decision.position_size_usd *= regime_scale
            modifications.append(
                f"Size adjusted for {context.regime.regime.value} regime"
            )

        # Check 7: ML prediction alignment
        if signal.ml_prediction == "UNFAVORABLE":
            if decision.position_size_usd:
                decision.position_size_usd *= 0.7
                modifications.append("Size reduced 30% due to unfavorable ML")
            warnings.append("ML prediction: UNFAVORABLE")

        # Check 8: Max position size (% of balance)
        if decision.position_size_usd:
            max_from_balance = account_balance * self.max_position_pct
            if decision.position_size_usd > max_from_balance:
                decision.position_size_usd = max_from_balance
                modifications.append(
                    f"Size capped at {self.max_position_pct*100:.0f}% of balance"
                )

        # Check 9: Exposure limit
        if decision.position_size_usd:
            max_new_position = self.max_exposure_usd - current_exposure
            if max_new_position <= 0:
                return self._block_decision(
                    decision,
                    "Maximum exposure limit reached",
                )

            if decision.position_size_usd > max_new_position:
                decision.position_size_usd = max_new_position
                modifications.append(
                    f"Size capped at ${max_new_position:.2f} due to exposure limit"
                )

        # Check 10: Minimum position size (worth trading?)
        # $5 minimum for micro mode
        if decision.position_size_usd and decision.position_size_usd < 5.0:
            return self._block_decision(
                decision,
                f"Position size too small (${decision.position_size_usd:.2f})",
            )

        # Check 11: High funding rate impact
        funding_cost = abs(context.funding.current_rate) * 100  # as percentage
        if funding_cost > 0.1:  # > 0.1% funding
            warnings.append(
                f"High funding: {context.funding.current_rate*100:.3f}%"
            )

        # Check 12: Fear & Greed extremes
        fg = context.sentiment.fear_greed_index
        if fg < 10:
            warnings.append(f"Extreme Fear ({fg}) - high risk")
        elif fg > 90:
            warnings.append(f"Extreme Greed ({fg}) - high risk")

        # Apply modifications
        if modifications:
            self._trades_modified += 1
            decision.risks_identified = warnings + modifications
            log.info(
                f"[RiskGuardian] Modified decision: {', '.join(modifications)}"
            )

        return decision

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached."""
        return self._daily_pnl <= -self.max_daily_loss_pct

    def _check_drawdown_limit(self, current_balance: float) -> bool:
        """Check if drawdown limit has been reached."""
        if self._peak_balance <= 0:
            return False

        drawdown = (self._peak_balance - current_balance) / self._peak_balance * 100
        return drawdown >= self.max_drawdown_pct

    def _block_decision(self, decision: Decision, reason: str) -> Decision:
        """Block a decision and return WAIT."""
        self._trades_blocked += 1

        log.warning(f"[RiskGuardian] Blocked trade: {reason}")

        return Decision(
            timestamp=datetime.now(timezone.utc),
            action="WAIT",
            signal=decision.signal,
            reasoning=f"Risk Guardian blocked: {reason}",
            key_factors=decision.key_factors,
            risks_identified=[reason],
            decision_source="RISK_GUARDIAN",
            confidence=1.0,
        )

    def record_pnl(self, pnl_percent: float) -> None:
        """Record PnL for daily tracking."""
        self._daily_pnl += pnl_percent
        log.debug(f"[RiskGuardian] Daily PnL updated: {self._daily_pnl:+.2f}%")

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL counter (call at day start)."""
        log.info(f"[RiskGuardian] Resetting daily PnL (was {self._daily_pnl:+.2f}%)")
        self._daily_pnl = 0.0

    def check_emergency_conditions(
        self,
        context: MarketContext,
        positions: list[dict[str, Any]],
    ) -> Decision | None:
        """
        Check for emergency conditions requiring immediate action.

        Args:
            context: Market context
            positions: Current positions

        Returns:
            Emergency decision or None
        """
        if not positions:
            return None

        now = datetime.now(timezone.utc)
        fg = context.sentiment.fear_greed_index

        # Calculate total unrealized PnL
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

        # Calculate total position value for percentage-based thresholds
        total_value = sum(abs(p.get("notional", 0) or p.get("contracts", 0) * p.get("markPrice", 0)) for p in positions)
        if total_value == 0:
            total_value = self.max_exposure_usd  # Fallback

        pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0

        # Emergency 1: Extreme volatility + significant losses (>10% of exposure)
        if context.regime.volatility_level == "extreme" and pnl_pct < -10:
            return Decision(
                timestamp=now,
                action="CLOSE_ALL",
                reasoning="Emergency: Extreme volatility with significant losses",
                key_factors=[
                    "Extreme volatility",
                    f"Unrealized PnL: ${total_pnl:.2f} ({pnl_pct:.1f}%)",
                ],
                risks_identified=["Volatility spike", "Further loss risk"],
                decision_source="EMERGENCY",
                confidence=0.95,
            )

        # Emergency 2: Market crash (extreme fear with big losses >15%)
        if fg < 5 and pnl_pct < -15:
            return Decision(
                timestamp=now,
                action="CLOSE_ALL",
                reasoning="Emergency: Potential market crash conditions",
                key_factors=[
                    f"Fear & Greed: {fg}",
                    f"Unrealized PnL: ${total_pnl:.2f} ({pnl_pct:.1f}%)",
                ],
                risks_identified=["Black swan risk", "Cascade liquidations"],
                decision_source="EMERGENCY",
                confidence=0.9,
            )

        # Emergency 3: All positions significantly losing (>3% each) - DISABLED for now
        # This was triggering too often on minor fluctuations
        # all_significantly_losing = all(
        #     p.get("unrealized_pnl", 0) < -0.03 * abs(p.get("notional", 1))
        #     for p in positions
        # )
        # Removed - individual SL orders handle this better

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get Risk Guardian statistics."""
        return {
            "trades_blocked": self._trades_blocked,
            "trades_modified": self._trades_modified,
            "daily_pnl": self._daily_pnl,
            "peak_balance": self._peak_balance,
            "max_exposure_usd": self.max_exposure_usd,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
        }
