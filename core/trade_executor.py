"""
Trade Executor for AI Trading System V3.
Executes trading decisions with proper sizing and protective orders.
"""
from __future__ import annotations

import asyncio
from typing import Any

from core.logger import get_logger
from core.state import Decision, Signal
from core.position_tracker import PositionTracker
from execution.binance_client import BinanceClient

log = get_logger("trade_executor")


class TradeExecutor:
    """
    Executes trading decisions.

    Modes:
    - shadow: Log only, no real trades
    - micro: Real trades, max $10 per position
    - normal: Real trades with full sizing
    """

    # Minimum quantities, step sizes, and notional requirements by symbol
    # min_notional is the minimum order value in USDT required by Binance
    QUANTITY_CONFIG = {
        # Tier 0: Major pairs (higher notional requirements - $20)
        "BTCUSDT": {"min": 0.001, "step": 0.001, "price_step": 0.1, "min_notional": 20},
        "ETHUSDT": {"min": 0.001, "step": 0.001, "price_step": 0.01, "min_notional": 20},
        "BNBUSDT": {"min": 0.01, "step": 0.01, "price_step": 0.01, "min_notional": 20},
        # Tier 1: Low minimums ($5 notional)
        "DOGEUSDT": {"min": 1, "step": 1, "price_step": 0.00001, "min_notional": 5},
        "XRPUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 5},
        "ADAUSDT": {"min": 1, "step": 1, "price_step": 0.0001, "min_notional": 5},
        "1000PEPEUSDT": {"min": 100, "step": 1, "price_step": 0.0000001, "min_notional": 5},
        "TRXUSDT": {"min": 1, "step": 1, "price_step": 0.00001, "min_notional": 5},
        "POLUSDT": {"min": 1, "step": 1, "price_step": 0.0001, "min_notional": 5},
        # Tier 2: Medium minimums
        "DOTUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.001, "min_notional": 5},
        "LINKUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.001, "min_notional": 5},
        "AVAXUSDT": {"min": 1, "step": 1, "price_step": 0.01, "min_notional": 5},
        "NEARUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.001, "min_notional": 5},
        # Tier 3: Popular alts
        "SUIUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 5},
        "APTUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.001, "min_notional": 5},
        "ARBUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 5},
        "OPUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 5},
        "WIFUSDT": {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 5},
        # Other large caps
        "SOLUSDT": {"min": 0.01, "step": 0.01, "price_step": 0.01, "min_notional": 5},
    }

    # Default for unknown symbols (use $20 to be safe)
    DEFAULT_CONFIG = {"min": 0.1, "step": 0.1, "price_step": 0.0001, "min_notional": 20}

    def __init__(
        self,
        binance_client: BinanceClient,
        position_tracker: PositionTracker,
        mode: str = "shadow",
        micro_max_usd: float = 10.0,
    ) -> None:
        """
        Initialize Trade Executor.

        Args:
            binance_client: Binance client for order execution
            position_tracker: Position tracker
            mode: "shadow" / "micro" / "normal"
            micro_max_usd: Maximum position size in micro mode
        """
        self.binance_client = binance_client
        self.position_tracker = position_tracker
        self.mode = mode
        self.micro_max_usd = micro_max_usd

        # Execution statistics
        self._executions_attempted = 0
        self._executions_successful = 0
        self._executions_failed = 0

        log.info(
            f"[TradeExecutor] Initialized in {mode.upper()} mode"
            f"{f' (max ${micro_max_usd})' if mode == 'micro' else ''}"
        )

    async def execute(
        self,
        decision: Decision,
        context_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision from orchestrator
            context_snapshot: Market context for learning

        Returns:
            Execution result dict
        """
        self._executions_attempted += 1

        # Not a trade decision
        if decision.action != "TRADE":
            return {
                "executed": False,
                "reason": f"action_is_{decision.action.lower()}",
            }

        signal = decision.signal
        if not signal:
            return {"executed": False, "reason": "no_signal"}

        # Shadow mode - log only
        if self.mode == "shadow":
            log.info(
                f"[SHADOW] Would execute: {signal.side} {signal.symbol} "
                f"@ ${signal.entry_price:.4f}, size=${decision.position_size_usd:.2f}"
            )
            return {
                "executed": False,
                "reason": "shadow_mode",
                "would_execute": {
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "size_usd": decision.position_size_usd,
                    "entry": signal.entry_price,
                    "sl": signal.stop_loss,
                    "tp1": signal.take_profit_1,
                    "tp2": signal.take_profit_2,
                },
            }

        # Pre-execution checks
        if self.position_tracker.has_position(signal.symbol):
            return {"executed": False, "reason": "position_already_open"}

        if not self.position_tracker.can_open_position():
            return {"executed": False, "reason": "max_positions_reached"}

        # Determine position size
        position_size_usd = decision.position_size_usd or 50.0

        if self.mode == "micro":
            position_size_usd = min(position_size_usd, self.micro_max_usd)
            log.info(f"[MICRO] Size limited to ${position_size_usd:.2f}")

        # Calculate quantity
        current_price = self.binance_client.get_current_price(signal.symbol)
        if current_price <= 0:
            return {"executed": False, "reason": "failed_to_get_price"}

        quantity = self._calculate_quantity(signal.symbol, position_size_usd, current_price)

        if quantity <= 0:
            return {"executed": False, "reason": "quantity_too_small"}

        # Execute with protection
        try:
            result = await self._execute_with_protection(
                signal=signal,
                quantity=quantity,
                current_price=current_price,
                context_snapshot=context_snapshot,
            )

            self._executions_successful += 1
            return {"executed": True, "result": result}

        except Exception as e:
            self._executions_failed += 1
            log.error(f"[TradeExecutor] Execution failed: {e}")
            return {"executed": False, "reason": str(e)}

    def _calculate_quantity(
        self,
        symbol: str,
        size_usd: float,
        price: float,
    ) -> float:
        """
        Calculate quantity with proper precision.

        Args:
            symbol: Trading symbol
            size_usd: Position size in USD
            price: Current price

        Returns:
            Quantity with correct precision
        """
        config = self.QUANTITY_CONFIG.get(symbol, self.DEFAULT_CONFIG)
        min_notional = config.get("min_notional", 5)

        # Check if size meets minimum notional requirement
        if size_usd < min_notional:
            log.warning(
                f"[TradeExecutor] {symbol}: Size ${size_usd:.2f} below min notional ${min_notional}"
            )
            # Increase size to meet minimum
            size_usd = min_notional

        # Base quantity
        quantity = size_usd / price

        # Round to step
        step = config["step"]
        quantity = (quantity // step) * step

        # Enforce minimum quantity
        if quantity < config["min"]:
            # If calculated qty is too small, use minimum
            min_value = config["min"] * price
            if min_value <= size_usd * 1.5:  # Allow some flexibility
                quantity = config["min"]
            else:
                log.warning(
                    f"[TradeExecutor] {symbol}: Min qty ${min_value:.2f} exceeds size ${size_usd:.2f}"
                )
                return 0  # Too expensive

        # Final notional check
        final_notional = quantity * price
        if final_notional < min_notional:
            # Increase quantity to meet notional
            required_qty = min_notional / price
            quantity = ((required_qty // step) + 1) * step
            log.info(
                f"[TradeExecutor] {symbol}: Adjusted qty to {quantity} for min notional ${min_notional}"
            )

        return quantity

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price to correct precision."""
        config = self.QUANTITY_CONFIG.get(symbol, self.DEFAULT_CONFIG)
        step = config["price_step"]
        return round(price / step) * step

    async def _execute_with_protection(
        self,
        signal: Signal,
        quantity: float,
        current_price: float,
        context_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Execute entry with SL and TP orders.

        Args:
            signal: Trading signal
            quantity: Quantity to trade
            current_price: Current market price
            context_snapshot: Market context

        Returns:
            Execution result
        """
        symbol = signal.symbol
        entry_side = "buy" if signal.side == "LONG" else "sell"
        exit_side = "sell" if signal.side == "LONG" else "buy"

        log.info(
            f"[EXECUTE] Opening {signal.side} {symbol}, "
            f"qty={quantity:.4f}, price~${current_price:.4f}"
        )

        # 1. Entry order (market)
        entry_order = self.binance_client.place_market_order(
            symbol=symbol,
            side=entry_side,
            quantity=quantity,
        )

        # entry_order is Order object, not dict - use attributes
        actual_entry_price = float(
            entry_order.average_price or
            entry_order.price or
            current_price
        ) if entry_order else current_price

        log.info(f"[EXECUTE] Entry filled @ ${actual_entry_price:.4f}")

        # Wait for position to register on exchange before placing SL/TP
        # Binance needs time to register the position before GTE orders can be placed
        await asyncio.sleep(1.0)

        # 2. Recalculate levels based on actual entry
        sl_price, tp1_price, tp2_price = self._recalculate_levels(
            signal, actual_entry_price
        )

        # Round prices
        sl_price = self._round_price(symbol, sl_price)
        tp1_price = self._round_price(symbol, tp1_price)
        tp2_price = self._round_price(symbol, tp2_price)

        # 3. Stop Loss order
        sl_order = None
        try:
            sl_order = self.binance_client.place_stop_loss(
                symbol=symbol,
                side=exit_side,
                quantity=quantity,
                stop_price=sl_price,
            )
            log.info(f"[EXECUTE] SL set @ ${sl_price:.4f}")
        except Exception as e:
            log.error(f"[EXECUTE] Failed to place SL: {e}")

        # 4. Take Profit orders (50% each)
        tp_orders: list[dict] = []

        # Round quantities
        config = self.QUANTITY_CONFIG.get(symbol, self.DEFAULT_CONFIG)
        step = config["step"]

        tp1_qty = ((quantity / 2) // step) * step
        tp2_qty = quantity - tp1_qty

        if tp1_qty < config["min"]:
            # If can't split, put all on TP1
            tp1_qty = quantity
            tp2_qty = 0

        # TP1
        if tp1_qty > 0:
            try:
                tp1_order = self.binance_client.place_take_profit(
                    symbol=symbol,
                    side=exit_side,
                    quantity=tp1_qty,
                    tp_price=tp1_price,
                )
                # Only append if order was successfully created
                if tp1_order is not None:
                    tp_orders.append(tp1_order)
                    log.info(f"[EXECUTE] TP1 set @ ${tp1_price:.4f} ({tp1_qty:.4f})")
                else:
                    log.warning(f"[EXECUTE] TP1 order returned None for {symbol}")
            except Exception as e:
                log.error(f"[EXECUTE] Failed to place TP1: {e}")

        # TP2 replaced by trailing stop after TP1 hits
        # Trailing stop with 1% callback will be activated in position_tracker
        if tp2_qty >= config["min"]:
            log.info(f"[EXECUTE] Trailing stop will activate after TP1 (callback=1%)")

        # 5. Add to position tracker
        position = self.position_tracker.add_position(
            signal=signal,
            entry_order=entry_order,
            sl_order=sl_order,
            tp_orders=tp_orders,
            context_snapshot=context_snapshot,
        )

        return {
            "position_id": position.id,
            "entry_price": actual_entry_price,
            "quantity": quantity,
            "position_value": actual_entry_price * quantity,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
        }

    def _recalculate_levels(
        self,
        signal: Signal,
        actual_entry: float,
    ) -> tuple[float, float, float]:
        """
        Recalculate SL/TP levels based on actual entry price.

        Args:
            signal: Original signal
            actual_entry: Actual fill price

        Returns:
            Tuple of (sl, tp1, tp2)
        """
        # Calculate original risk percentage
        original_risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

        # Original R:R ratios
        original_reward1_pct = abs(signal.take_profit_1 - signal.entry_price) / signal.entry_price
        original_reward2_pct = abs(signal.take_profit_2 - signal.entry_price) / signal.entry_price

        # Recalculate with same percentages
        if signal.side == "LONG":
            sl = actual_entry * (1 - original_risk_pct)
            tp1 = actual_entry * (1 + original_reward1_pct)
            tp2 = actual_entry * (1 + original_reward2_pct)
        else:  # SHORT
            sl = actual_entry * (1 + original_risk_pct)
            tp1 = actual_entry * (1 - original_reward1_pct)
            tp2 = actual_entry * (1 - original_reward2_pct)

        return (sl, tp1, tp2)

    async def close_all_positions(self, reason: str = "MANUAL") -> int:
        """
        Close all open positions.

        Args:
            reason: Reason for closing

        Returns:
            Number of positions closed
        """
        closed_count = 0

        for symbol in list(self.position_tracker.positions.keys()):
            if self.position_tracker.has_position(symbol):
                if self.position_tracker.close_position(symbol, reason):
                    closed_count += 1
                    log.info(f"[TradeExecutor] Closed {symbol} ({reason})")

        return closed_count

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "mode": self.mode,
            "executions_attempted": self._executions_attempted,
            "executions_successful": self._executions_successful,
            "executions_failed": self._executions_failed,
            "success_rate": (
                self._executions_successful / self._executions_attempted
                if self._executions_attempted > 0 else 0
            ),
        }
