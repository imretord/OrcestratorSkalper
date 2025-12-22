"""
Trailing Stop Manager for AI Trading System V3.
Implements adaptive trailing stop logic based on ROI levels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

log = get_logger("trailing_stop")


@dataclass
class TrailingConfig:
    """Configuration for trailing stop behavior."""
    # Activation thresholds (real ROI = displayed ROI / leverage)
    activation_real_roi_pct: float = 1.5  # Activate trailing at 1.5% real ROI
    profit_protection_pct: float = 0.5    # Protect 50% of current profit

    # Level 1: ROI 1.5-2.5% - First profit lock
    level1_roi_min: float = 1.5
    level1_roi_max: float = 2.5
    level1_min_distance_pct: float = 0.8  # Min 0.8% distance

    # Level 2: ROI 2.5-4.0% - Conservative trailing
    level2_roi_min: float = 2.5
    level2_roi_max: float = 4.0
    level2_min_distance_pct: float = 0.6  # Min 0.6% distance

    # Level 3: ROI >4.0% - Aggressive trailing
    level3_roi_min: float = 4.0
    level3_min_distance_pct: float = 0.5  # Min 0.5% distance


@dataclass
class TrailingState:
    """State for a single position's trailing stop."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: float
    current_level: int = 0  # 0=initial, 1-3=trailing levels
    current_sl: float = 0.0
    last_update_time: float = 0.0
    leverage: int = 1
    trailing_activated: bool = False
    max_roi_reached: float = 0.0
    sl_updates_count: int = 0


class TrailingStopManager:
    """
    Manages trailing stops for multiple positions.

    Features:
    - Multi-level trailing based on ROI
    - Profit protection (locks in percentage of gains)
    - Adaptive distance based on volatility
    """

    def __init__(self, config: TrailingConfig | None = None) -> None:
        """
        Initialize TrailingStopManager.

        Args:
            config: Trailing stop configuration
        """
        self.config = config or TrailingConfig()
        self.states: dict[str, TrailingState] = {}

        log.info("[TRAILING] Manager initialized")
        log.info(f"[TRAILING] Activation at {self.config.activation_real_roi_pct}% real ROI")

    def initialize_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        leverage: int = 1,
        entry_time: float | None = None,
    ) -> None:
        """
        Initialize trailing for a new position.

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            entry_price: Position entry price
            stop_loss: Initial stop loss price
            leverage: Position leverage
            entry_time: Entry timestamp (Unix)
        """
        self.states[symbol] = TrailingState(
            symbol=symbol,
            side=side.upper(),
            entry_price=entry_price,
            entry_time=entry_time or time.time(),
            current_sl=stop_loss,
            leverage=leverage,
            last_update_time=time.time(),
        )

        log.info(
            f"[TRAILING] {symbol}: Initialized {side} @ ${entry_price:.4f}, "
            f"SL @ ${stop_loss:.4f}, leverage={leverage}x"
        )

    def update(
        self,
        symbol: str,
        current_price: float,
        displayed_roi_pct: float,
        current_atr: float | None = None,
    ) -> tuple[bool, float | None, str]:
        """
        Update trailing stop for a position.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            displayed_roi_pct: ROI percentage shown (includes leverage)
            current_atr: Current ATR (optional, for dynamic distance)

        Returns:
            Tuple of (should_update, new_sl, reason)
        """
        state = self.states.get(symbol)
        if not state:
            return False, None, "Position not initialized"

        # Calculate real ROI (remove leverage effect)
        leverage = max(state.leverage, 1)
        real_roi_pct = displayed_roi_pct / leverage

        # Track max ROI
        state.max_roi_reached = max(state.max_roi_reached, real_roi_pct)

        # Don't trail losing positions
        if real_roi_pct <= 0:
            return False, None, f"ROI {real_roi_pct:.2f}% <= 0"

        # Check activation threshold
        if not state.trailing_activated:
            if real_roi_pct >= self.config.activation_real_roi_pct:
                state.trailing_activated = True
                log.info(f"[TRAILING] {symbol}: Activated at {real_roi_pct:.2f}% real ROI")
            else:
                return False, None, f"Waiting for {self.config.activation_real_roi_pct}% ROI (current: {real_roi_pct:.2f}%)"

        # Determine level
        target_level = self._determine_level(real_roi_pct)

        if target_level > state.current_level:
            log.info(f"[TRAILING] {symbol}: Level {state.current_level} -> {target_level}")
            state.current_level = target_level

        # Calculate new stop loss
        new_sl = self._calculate_trailing_sl(
            state=state,
            current_price=current_price,
            real_roi_pct=real_roi_pct,
            atr=current_atr,
        )

        if new_sl is None:
            return False, None, "No SL update needed"

        # Check if new SL is better
        if state.side == 'LONG':
            if new_sl <= state.current_sl:
                return False, None, f"New SL ${new_sl:.4f} not better than ${state.current_sl:.4f}"
        else:  # SHORT
            if new_sl >= state.current_sl:
                return False, None, f"New SL ${new_sl:.4f} not better than ${state.current_sl:.4f}"

        # Update state
        old_sl = state.current_sl
        state.current_sl = new_sl
        state.last_update_time = time.time()
        state.sl_updates_count += 1

        reason = f"Level {state.current_level}: ROI {real_roi_pct:.2f}%, SL moved ${old_sl:.4f} -> ${new_sl:.4f}"
        log.info(f"[TRAILING] {symbol}: {reason}")

        return True, new_sl, reason

    def _determine_level(self, real_roi_pct: float) -> int:
        """Determine trailing level based on ROI."""
        if real_roi_pct >= self.config.level3_roi_min:
            return 3
        elif real_roi_pct >= self.config.level2_roi_min:
            return 2
        elif real_roi_pct >= self.config.level1_roi_min:
            return 1
        return 0

    def _calculate_trailing_sl(
        self,
        state: TrailingState,
        current_price: float,
        real_roi_pct: float,
        atr: float | None,
    ) -> float | None:
        """
        Calculate new trailing stop price.

        Args:
            state: Position state
            current_price: Current market price
            real_roi_pct: Real ROI percentage
            atr: ATR for dynamic distance

        Returns:
            New stop loss price or None
        """
        # Get minimum distance based on level
        if state.current_level == 3:
            min_distance_pct = self.config.level3_min_distance_pct
        elif state.current_level == 2:
            min_distance_pct = self.config.level2_min_distance_pct
        else:
            min_distance_pct = self.config.level1_min_distance_pct

        # Calculate distance
        distance = current_price * (min_distance_pct / 100.0)

        # Use ATR if available and larger
        if atr and atr > distance:
            distance = atr * 0.8  # 80% of ATR

        # Calculate trailing SL
        if state.side == 'LONG':
            trailing_sl = current_price - distance
        else:  # SHORT
            trailing_sl = current_price + distance

        # Apply profit protection
        protected_sl = self._apply_profit_protection(state, current_price, trailing_sl)

        return protected_sl

    def _apply_profit_protection(
        self,
        state: TrailingState,
        current_price: float,
        trailing_sl: float,
    ) -> float:
        """
        Ensure SL protects minimum profit percentage.

        Args:
            state: Position state
            current_price: Current price
            trailing_sl: Calculated trailing SL

        Returns:
            Protected stop loss price
        """
        protection_pct = self.config.profit_protection_pct

        if state.side == 'LONG':
            current_profit = current_price - state.entry_price
            min_profit_sl = state.entry_price + (current_profit * protection_pct)
            return max(trailing_sl, min_profit_sl)
        else:  # SHORT
            current_profit = state.entry_price - current_price
            min_profit_sl = state.entry_price - (current_profit * protection_pct)
            return min(trailing_sl, min_profit_sl)

    def get_state(self, symbol: str) -> TrailingState | None:
        """Get state for a symbol."""
        return self.states.get(symbol)

    def cleanup(self, symbol: str) -> None:
        """Remove state for a symbol."""
        if symbol in self.states:
            del self.states[symbol]
            log.info(f"[TRAILING] {symbol}: State cleaned up")

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all position states as dict."""
        result = {}
        for symbol, state in self.states.items():
            result[symbol] = {
                'side': state.side,
                'entry_price': state.entry_price,
                'current_sl': state.current_sl,
                'level': state.current_level,
                'max_roi': state.max_roi_reached,
                'updates': state.sl_updates_count,
                'activated': state.trailing_activated,
            }
        return result
