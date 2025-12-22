"""
Risk Guardian for AI Trading System V3.
Monitors and enforces risk limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger

log = get_logger("risk_guardian")


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_exposure_pct: float = 50.0      # Max % of capital in positions
    max_position_pct: float = 20.0      # Max % of capital per position
    max_daily_loss_pct: float = 5.0     # Max daily loss %
    max_positions: int = 3              # Max concurrent positions
    min_balance_usd: float = 50.0       # Minimum balance to trade


class RiskGuardian:
    """
    Monitors risk limits and blocks trades that violate rules.

    Features:
    - Position size limits
    - Daily loss tracking
    - Exposure limits
    - Emergency stop capability
    """

    def __init__(
        self,
        initial_capital: float,
        limits: RiskLimits | None = None,
    ) -> None:
        """
        Initialize RiskGuardian.

        Args:
            initial_capital: Starting capital in USDT
            limits: Risk limits configuration
        """
        self.initial_capital = initial_capital
        self.limits = limits or RiskLimits()

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = initial_capital
        self.last_reset_date = datetime.now(timezone.utc).date()

        # Emergency stop
        self.emergency_stop = False
        self.stop_reason = ""

        log.info(
            f"[RiskGuardian] Initialized with capital=${initial_capital:.2f}, "
            f"max_loss={self.limits.max_daily_loss_pct}%"
        )

    def check_new_position(
        self,
        current_balance: float,
        current_exposure: float,
        position_size_usd: float,
        num_positions: int,
    ) -> tuple[bool, str]:
        """
        Check if a new position is allowed.

        Args:
            current_balance: Current account balance
            current_exposure: Current total exposure in USD
            position_size_usd: Proposed position size in USD
            num_positions: Current number of open positions

        Returns:
            Tuple of (is_allowed, reason)
        """
        self._check_daily_reset()

        # Emergency stop
        if self.emergency_stop:
            return False, f"Emergency stop active: {self.stop_reason}"

        # Minimum balance
        if current_balance < self.limits.min_balance_usd:
            return False, f"Balance ${current_balance:.2f} below minimum ${self.limits.min_balance_usd}"

        # Max positions
        if num_positions >= self.limits.max_positions:
            return False, f"Max positions ({self.limits.max_positions}) reached"

        # Position size limit
        max_position = current_balance * (self.limits.max_position_pct / 100)
        if position_size_usd > max_position:
            return False, f"Position ${position_size_usd:.2f} exceeds max ${max_position:.2f}"

        # Exposure limit
        new_exposure = current_exposure + position_size_usd
        max_exposure = current_balance * (self.limits.max_exposure_pct / 100)
        if new_exposure > max_exposure:
            return False, f"Total exposure ${new_exposure:.2f} would exceed max ${max_exposure:.2f}"

        # Daily loss limit
        daily_loss_limit = self.daily_start_balance * (self.limits.max_daily_loss_pct / 100)
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"

        return True, "OK"

    def record_trade_pnl(self, pnl: float) -> None:
        """
        Record PnL from a closed trade.

        Args:
            pnl: Profit/loss in USD
        """
        self._check_daily_reset()
        self.daily_pnl += pnl

        daily_loss_limit = self.daily_start_balance * (self.limits.max_daily_loss_pct / 100)

        if self.daily_pnl < -daily_loss_limit:
            self.trigger_emergency_stop(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")

        log.info(f"[RiskGuardian] Trade PnL: ${pnl:+.2f}, Daily: ${self.daily_pnl:+.2f}")

    def trigger_emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        self.emergency_stop = True
        self.stop_reason = reason
        log.critical(f"[RiskGuardian] EMERGENCY STOP: {reason}")

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop state."""
        self.emergency_stop = False
        self.stop_reason = ""
        log.info("[RiskGuardian] Emergency stop cleared")

    def _check_daily_reset(self) -> None:
        """Reset daily tracking at start of new day."""
        today = datetime.now(timezone.utc).date()

        if today > self.last_reset_date:
            log.info(f"[RiskGuardian] New day - resetting daily PnL from ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.last_reset_date = today

            # Clear daily loss emergency stop
            if self.emergency_stop and "Daily loss" in self.stop_reason:
                self.clear_emergency_stop()

    def get_status(self) -> dict[str, Any]:
        """Get risk guardian status."""
        daily_loss_limit = self.daily_start_balance * (self.limits.max_daily_loss_pct / 100)
        daily_pnl_pct = (self.daily_pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0

        return {
            'emergency_stop': self.emergency_stop,
            'stop_reason': self.stop_reason,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'daily_loss_limit': daily_loss_limit,
            'daily_remaining': daily_loss_limit + self.daily_pnl,
            'limits': {
                'max_exposure_pct': self.limits.max_exposure_pct,
                'max_position_pct': self.limits.max_position_pct,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct,
                'max_positions': self.limits.max_positions,
            }
        }
