"""
Position Tracker for AI Trading System V3.
Tracks open positions, their PnL, and exit conditions.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import (
    MarketRegime,
    Order,
    Signal,
    TrackedPosition,
)
from execution.binance_client import BinanceClient

log = get_logger("position_tracker")


class PositionTracker:
    """
    Tracks all open and closed positions.

    Responsibilities:
    - Track position lifecycle
    - Update real-time PnL
    - Detect SL/TP hits
    - Enforce position limits
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        max_positions: int = 3,
    ) -> None:
        """
        Initialize Position Tracker.

        Args:
            binance_client: Binance client for price/order queries
            max_positions: Maximum concurrent positions
        """
        self.binance_client = binance_client
        self.max_positions = max_positions

        # Active positions: symbol -> TrackedPosition
        self.positions: dict[str, TrackedPosition] = {}

        # Closed positions history (for session)
        self.closed_positions: list[TrackedPosition] = []

        log.info(
            f"[PositionTracker] Initialized with max_positions={max_positions}"
        )

    def add_position(
        self,
        signal: Signal,
        entry_order: Order | None,
        sl_order: Order | None,
        tp_orders: list[Order],
        context_snapshot: dict[str, Any] | None = None,
    ) -> TrackedPosition:
        """
        Add a new position after order execution.

        Args:
            signal: Signal that triggered the position
            entry_order: Filled entry order from Binance
            sl_order: Stop-loss order
            tp_orders: Take-profit orders
            context_snapshot: Market context at entry time

        Returns:
            TrackedPosition object
        """
        # entry_order is Order object, not dict - use attributes
        entry_price = float(entry_order.average_price or entry_order.price or 0) if entry_order else 0
        quantity = float(entry_order.filled_quantity or entry_order.quantity or 0) if entry_order else 0

        # CRITICAL: Validate entry_price and quantity before creating position
        if entry_price <= 0 or quantity <= 0:
            log.error(
                f"[PositionTracker] INVALID position data: {signal.symbol} "
                f"entry_price={entry_price}, quantity={quantity} - NOT adding position"
            )
            # Create a dummy position that will be immediately marked as failed
            # This prevents recording trades with invalid data
            raise ValueError(
                f"Invalid position data: entry_price={entry_price}, quantity={quantity}"
            )

        position = TrackedPosition(
            id=str(uuid.uuid4()),
            signal_id=signal.id,
            agent_name=signal.agent_name,
            symbol=signal.symbol,
            side=signal.side,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            quantity=quantity,
            position_value_usd=entry_price * quantity,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            entry_order_id=entry_order.order_id if entry_order else '',
            sl_order_id=sl_order.order_id if sl_order else None,
            tp1_order_id=tp_orders[0].order_id if len(tp_orders) > 0 else None,
            tp2_order_id=tp_orders[1].order_id if len(tp_orders) > 1 else None,
            status="OPEN",
            entry_regime=signal.regime_at_signal,
            entry_context=context_snapshot or {},
            current_price=entry_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
        )

        self.positions[signal.symbol] = position

        log.info(
            f"[PositionTracker] Added position: {signal.side} {signal.symbol} "
            f"@ ${entry_price:.4f}, qty={quantity:.4f}"
        )

        return position

    def update_positions(self) -> None:
        """Update PnL for all open positions."""
        for symbol, position in self.positions.items():
            if position.status == "CLOSED":
                continue

            try:
                # Get current price
                current_price = self.binance_client.get_current_price(symbol)
                if current_price <= 0:
                    continue

                position.current_price = current_price

                # Calculate PnL
                if position.side == "LONG":
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                else:  # SHORT
                    pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

                position.unrealized_pnl_pct = pnl_pct
                position.unrealized_pnl = position.position_value_usd * (pnl_pct / 100)

                # Update max profit for trailing
                if pnl_pct > position.max_profit_pct:
                    position.max_profit_pct = pnl_pct

            except Exception as e:
                log.warning(f"[PositionTracker] Failed to update {symbol}: {e}")

    def check_exits(self) -> tuple[list[TrackedPosition], list[dict]]:
        """
        Check if any positions have been closed (SL/TP triggered).
        Uses ACTUAL execution prices from Binance, not theoretical SL/TP levels.

        Returns:
            Tuple of (closed_positions, tp1_events)
            - closed_positions: List of newly closed positions
            - tp1_events: List of {symbol, price, trailing_active} for TP1 hits this cycle
        """
        closed: list[TrackedPosition] = []
        tp1_events: list[dict] = []

        for symbol, position in list(self.positions.items()):
            if position.status == "CLOSED":
                continue

            try:
                # Get both regular and algo orders for this symbol
                regular_orders, algo_orders = self.binance_client.get_all_position_orders(symbol)
                open_order_ids = {o.order_id for o in regular_orders}
                open_algo_ids = {o.order_id for o in algo_orders}

                # Check SL order (algo order)
                if position.sl_order_id and position.sl_order_id not in open_algo_ids:
                    # SL was triggered - get ACTUAL execution price
                    actual_exit_price = self._get_actual_exit_price(
                        symbol, position.sl_order_id, is_algo=True, fallback_price=position.stop_loss
                    )

                    log.info(
                        f"[PositionTracker] {symbol} SL triggered | "
                        f"Target: ${position.stop_loss:.6f} | Actual: ${actual_exit_price:.6f} | "
                        f"Slippage: {self._calc_slippage(position.stop_loss, actual_exit_price):.3f}%"
                    )

                    # Cancel remaining orders for this position
                    self._cancel_all_position_orders(symbol, position)

                    self._close_position(position, "SL", actual_exit_price)
                    closed.append(position)
                    continue

                # Check TP1 order (algo order)
                if position.tp1_order_id and position.tp1_order_id not in open_algo_ids:
                    if not position.tp1_hit:
                        # Get actual TP1 execution price
                        actual_tp1_price = self._get_actual_exit_price(
                            symbol, position.tp1_order_id, is_algo=True, fallback_price=position.take_profit_1
                        )

                        position.tp1_hit = True
                        position.tp1_actual_price = actual_tp1_price
                        position.status = "PARTIALLY_CLOSED"

                        # Calculate and store TP1 PnL (50% of position closed at TP1)
                        tp1_qty = position.quantity / 2
                        if position.side == "LONG":
                            tp1_pnl = tp1_qty * (actual_tp1_price - position.entry_price)
                        else:
                            tp1_pnl = tp1_qty * (position.entry_price - actual_tp1_price)
                        position.tp1_realized_pnl = tp1_pnl

                        log.info(
                            f"[PositionTracker] {symbol} TP1 PnL: ${tp1_pnl:+.4f} "
                            f"(50% @ ${actual_tp1_price:.6f})"
                        )

                        # Activate trailing stop for remaining 50%
                        position.trailing_stop_active = True
                        position.trailing_peak_price = position.current_price or actual_tp1_price

                        log.info(
                            f"[PositionTracker] {symbol} TP1 hit - 50% closed | "
                            f"Target: ${position.take_profit_1:.6f} | Actual: ${actual_tp1_price:.6f}"
                        )

                        # Place trailing stop on Binance (backup protection)
                        remaining_qty = position.quantity / 2
                        exit_side = "sell" if position.side == "LONG" else "buy"
                        callback_pct = position.trailing_stop_callback * 100  # 0.01 -> 1.0

                        trailing_order = self.binance_client.place_trailing_stop(
                            symbol=symbol,
                            side=exit_side,
                            quantity=remaining_qty,
                            callback_rate=callback_pct,
                        )

                        if trailing_order:
                            position.trailing_stop_order_id = trailing_order['order_id']
                            log.info(
                                f"[PositionTracker] {symbol} Trailing stop on EXCHANGE | "
                                f"Callback: {callback_pct}% | OrderId: {position.trailing_stop_order_id}"
                            )
                        else:
                            log.warning(f"[PositionTracker] {symbol} Failed to place trailing on exchange, using software only")

                        log.info(
                            f"[PositionTracker] {symbol} Software trailing ACTIVATED | "
                            f"Peak: ${position.trailing_peak_price:.6f}"
                        )

                        # Track TP1 event for notification
                        tp1_events.append({
                            "symbol": symbol,
                            "price": actual_tp1_price,
                            "trailing_active": position.trailing_stop_order_id is not None,
                            "trailing_order_id": position.trailing_stop_order_id,
                        })

                # Check trailing stop (replaces TP2)
                if position.trailing_stop_active and position.current_price:
                    current = position.current_price
                    peak = position.trailing_peak_price or current

                    # Update peak price
                    if position.side == "LONG":
                        if current > peak:
                            position.trailing_peak_price = current
                            peak = current
                        # Check callback - close if price drops 1% from peak
                        callback_price = peak * (1 - position.trailing_stop_callback)
                        if current <= callback_price:
                            log.info(
                                f"[PositionTracker] {symbol} TRAILING STOP triggered | "
                                f"Peak: ${peak:.6f} | Current: ${current:.6f} | Callback: 1%"
                            )
                            # Close remaining position via market order
                            self._close_trailing_position(position, current)
                            closed.append(position)
                    else:  # SHORT
                        if current < peak:
                            position.trailing_peak_price = current
                            peak = current
                        # Check callback - close if price rises 1% from bottom
                        callback_price = peak * (1 + position.trailing_stop_callback)
                        if current >= callback_price:
                            log.info(
                                f"[PositionTracker] {symbol} TRAILING STOP triggered | "
                                f"Bottom: ${peak:.6f} | Current: ${current:.6f} | Callback: 1%"
                            )
                            # Close remaining position via market order
                            self._close_trailing_position(position, current)
                            closed.append(position)

            except Exception as e:
                log.warning(f"[PositionTracker] Failed to check exits for {symbol}: {e}")

        return closed, tp1_events

    def _get_actual_exit_price(
        self,
        symbol: str,
        order_id: str,
        is_algo: bool = True,
        fallback_price: float = 0.0
    ) -> float:
        """
        Get actual execution price from Binance order.

        Args:
            symbol: Trading symbol
            order_id: Order ID
            is_algo: True if algo order (SL/TP), False if regular order
            fallback_price: Price to use if actual price not available

        Returns:
            Actual execution price or fallback
        """
        try:
            if is_algo:
                order = self.binance_client.get_algo_order(symbol, order_id)
            else:
                order = self.binance_client.get_order(symbol, order_id)

            if order and order.average_price and order.average_price > 0:
                return order.average_price

            log.warning(
                f"[PositionTracker] Could not get actual price for order {order_id}, "
                f"using fallback: ${fallback_price:.6f}"
            )
            return fallback_price

        except Exception as e:
            log.warning(f"[PositionTracker] Error getting order price: {e}, using fallback")
            return fallback_price

    def _calc_slippage(self, expected: float, actual: float) -> float:
        """Calculate slippage percentage."""
        if expected == 0:
            return 0.0
        return ((actual - expected) / expected) * 100

    def _cancel_all_position_orders(self, symbol: str, position: TrackedPosition) -> None:
        """
        Cancel all remaining orders for a position.

        Args:
            symbol: Trading symbol
            position: Position being closed
        """
        try:
            cancelled = self.binance_client.cancel_all_position_orders(symbol)
            if cancelled > 0:
                log.info(f"[PositionTracker] Cancelled {cancelled} remaining orders for {symbol}")
        except Exception as e:
            log.warning(f"[PositionTracker] Failed to cancel orders for {symbol}: {e}")

    def _close_trailing_position(self, position: TrackedPosition, current_price: float) -> None:
        """
        Close remaining position via market order (trailing stop triggered).

        Args:
            position: Position to close
            current_price: Current market price
        """
        symbol = position.symbol

        try:
            # CRITICAL: First check if position still exists on exchange
            # This prevents race condition with exchange trailing stop
            exchange_positions = self.binance_client.get_positions()
            exchange_pos = None
            for pos in exchange_positions:
                if pos.symbol == symbol and abs(pos.contracts) > 0:
                    exchange_pos = pos
                    break

            if not exchange_pos:
                # Position already closed (likely by exchange trailing stop)
                log.warning(
                    f"[PositionTracker] {symbol} position already closed on exchange "
                    f"(likely by exchange trailing stop) - skipping software close"
                )
                # Still mark as closed in our tracker
                self._close_position(position, "TRAILING", current_price)
                return

            # Get actual remaining quantity from exchange (more accurate than our calculation)
            actual_remaining_qty = abs(exchange_pos.contracts)
            log.info(
                f"[PositionTracker] {symbol} actual remaining on exchange: {actual_remaining_qty} "
                f"(calculated: {position.quantity / 2})"
            )

            # Cancel remaining orders (SL, trailing stop on exchange)
            self._cancel_all_position_orders(symbol, position)

            # Close remaining position via market order
            close_side = "sell" if position.side == "LONG" else "buy"

            order = self.binance_client.place_market_order(
                symbol=symbol,
                side=close_side,
                quantity=actual_remaining_qty,  # Use actual qty from exchange
                reduce_only=True,  # CRITICAL: prevents opening opposite position
            )

            if order:
                actual_price = order.average_price if order.average_price else current_price
                log.info(
                    f"[PositionTracker] Trailing stop closed {symbol} | "
                    f"Qty: {actual_remaining_qty:.4f} @ ${actual_price:.6f}"
                )
                self._close_position(position, "TRAILING", actual_price)
            else:
                # Market order failed - try close_position as fallback
                log.warning(f"[PositionTracker] Market order failed for {symbol}, trying close_position")
                close_order = self.binance_client.close_position(symbol)
                if close_order:
                    actual_price = close_order.average_price if close_order.average_price else current_price
                    log.info(f"[PositionTracker] Position closed via fallback: {symbol} @ ${actual_price:.6f}")
                    self._close_position(position, "TRAILING", actual_price)
                else:
                    log.error(f"[PositionTracker] Failed to close {symbol} - position may still be open!")
                    # Don't mark as closed if we couldn't actually close it
                    # This will cause health check to flag it

        except Exception as e:
            log.error(f"[PositionTracker] Failed to close trailing position {symbol}: {e}")
            # Try fallback close
            try:
                close_order = self.binance_client.close_position(symbol)
                if close_order:
                    actual_price = close_order.average_price if close_order.average_price else current_price
                    self._close_position(position, "TRAILING", actual_price)
                    return
            except Exception:
                pass
            # Mark as error but don't pretend it's closed
            log.error(f"[PositionTracker] {symbol} trailing close FAILED - position still open!")

    def _close_position(
        self,
        position: TrackedPosition,
        reason: str,
        exit_price: float,
    ) -> None:
        """
        Mark a position as closed.

        Args:
            position: Position to close
            reason: Exit reason (SL/TP1/TP2/MANUAL/EMERGENCY)
            exit_price: Exit price
        """
        position.status = "CLOSED"
        position.exit_reason = reason
        position.exit_time = datetime.now(timezone.utc)
        position.exit_price = exit_price

        # Calculate realized PnL percentage
        if position.side == "LONG":
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100

        position.realized_pnl_pct = pnl_pct

        # Calculate realized PnL in USD
        # If TP1 was hit, only 50% of position remains (the other 50% was closed at TP1)
        if position.tp1_hit:
            # Remaining 50% PnL
            remaining_qty = position.quantity / 2
            if position.side == "LONG":
                remaining_pnl = remaining_qty * (exit_price - position.entry_price)
            else:
                remaining_pnl = remaining_qty * (position.entry_price - exit_price)

            # Total PnL = TP1 PnL + remaining PnL
            tp1_pnl = position.tp1_realized_pnl or 0
            position.realized_pnl = tp1_pnl + remaining_pnl

            log.info(
                f"[PositionTracker] {position.symbol} Final PnL breakdown: "
                f"TP1=${tp1_pnl:+.4f} + Trailing=${remaining_pnl:+.4f} = ${position.realized_pnl:+.4f}"
            )
        else:
            # Full position closed at once
            position.realized_pnl = position.quantity * abs(exit_price - position.entry_price)
            if position.side == "LONG":
                position.realized_pnl = position.quantity * (exit_price - position.entry_price)
            else:
                position.realized_pnl = position.quantity * (position.entry_price - exit_price)

        # Move to closed history
        self.closed_positions.append(position)

        # CRITICAL: Remove from active positions to prevent duplicate processing
        symbol = position.symbol
        if symbol in self.positions:
            del self.positions[symbol]
            log.info(f"[PositionTracker] {symbol} removed from active tracking")

        log.info(
            f"[PositionTracker] Position closed: {position.symbol} | "
            f"Reason: {reason} | PnL: {pnl_pct:+.2f}%"
        )

    def close_position(self, symbol: str, reason: str) -> bool:
        """
        Forcefully close a position.

        Args:
            symbol: Symbol to close
            reason: Reason for closure

        Returns:
            True if closed successfully
        """
        if symbol not in self.positions:
            log.warning(f"[PositionTracker] No position found for {symbol}")
            return False

        position = self.positions[symbol]
        if position.status == "CLOSED":
            return False

        try:
            # Cancel ALL orders for this position (regular + algo/SL/TP)
            self._cancel_all_position_orders(symbol, position)

            # Get remaining quantity
            remaining_qty = position.quantity
            if position.tp1_hit:
                remaining_qty = position.quantity / 2

            # Close with market order
            close_side = "SELL" if position.side == "LONG" else "BUY"

            close_order = self.binance_client.place_market_order(
                symbol=symbol,
                side=close_side.lower(),
                quantity=remaining_qty,
                reduce_only=True,  # CRITICAL: bypass minimum notional for position close
            )

            # Use ACTUAL execution price from the close order
            exit_price = float(
                close_order.average_price or
                close_order.price or
                position.current_price or
                position.entry_price
            ) if close_order else (position.current_price or position.entry_price)

            log.info(
                f"[PositionTracker] Manual close {symbol} | "
                f"Reason: {reason} | Exit price: ${exit_price:.6f}"
            )

            self._close_position(position, reason, exit_price)

            return True

        except Exception as e:
            log.error(f"[PositionTracker] Failed to close {symbol}: {e}")
            return False

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self.positions and self.positions[symbol].status != "CLOSED"

    def get_position(self, symbol: str) -> TrackedPosition | None:
        """Get position for a symbol."""
        pos = self.positions.get(symbol)
        if pos and pos.status != "CLOSED":
            return pos
        return None

    def get_open_positions(self) -> list[TrackedPosition]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.status != "CLOSED"]

    def get_total_exposure(self) -> float:
        """Get total USD value of open positions."""
        return sum(
            p.position_value_usd
            for p in self.positions.values()
            if p.status != "CLOSED"
        )

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(
            p.unrealized_pnl or 0
            for p in self.positions.values()
            if p.status != "CLOSED"
        )

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        open_count = sum(1 for p in self.positions.values() if p.status != "CLOSED")
        return open_count < self.max_positions

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for current session."""
        closed = self.closed_positions

        if not closed:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

        wins = [p for p in closed if (p.realized_pnl or 0) > 0]
        losses = [p for p in closed if (p.realized_pnl or 0) <= 0]

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(p.realized_pnl or 0 for p in closed),
            "avg_win": sum(p.realized_pnl or 0 for p in wins) / len(wins) if wins else 0,
            "avg_loss": sum(p.realized_pnl or 0 for p in losses) / len(losses) if losses else 0,
        }

    def sync_with_exchange(self) -> dict[str, Any]:
        """
        Sync local position state with actual Binance positions.
        Detects discrepancies, phantom positions, and orphaned orders.

        Returns:
            Dict with sync results and any discrepancies found
        """
        results = {
            "synced": [],
            "orphaned_positions": [],
            "phantom_removed": [],
            "discrepancies": [],
        }

        try:
            # Get actual positions from Binance
            exchange_positions = self.binance_client.get_positions()

            # Build map of real positions (only those with contracts)
            real_positions = {
                p.symbol.replace('/USDT:USDT', 'USDT'): p
                for p in exchange_positions
                if abs(p.contracts) > 0
            }

            # Check tracked positions against exchange
            for symbol, tracked in list(self.positions.items()):
                if tracked.status == "CLOSED":
                    continue

                # Check for invalid tracked positions (qty=0 or entry_price=0)
                if tracked.quantity == 0 or tracked.entry_price == 0:
                    if symbol not in real_positions:
                        # Phantom position - remove entirely
                        log.warning(
                            f"[PositionTracker] Removing phantom position: {symbol} "
                            f"(qty={tracked.quantity}, entry={tracked.entry_price})"
                        )
                        results["phantom_removed"].append(symbol)
                        del self.positions[symbol]
                        continue

                ccxt_symbol = self.binance_client._to_ccxt_symbol(symbol)

                # Find matching exchange position
                exchange_pos = real_positions.get(symbol)

                if exchange_pos is None:
                    # Position doesn't exist on exchange - remove as phantom
                    log.warning(
                        f"[PositionTracker] {symbol} not found on exchange - removing phantom"
                    )
                    results["phantom_removed"].append(symbol)
                    del self.positions[symbol]
                else:
                    # Position exists - check for discrepancies and auto-fix

                    # Fix quantity mismatch
                    if abs(abs(exchange_pos.contracts) - tracked.quantity) > 0.0001:
                        log.warning(
                            f"[PositionTracker] {symbol} quantity mismatch: "
                            f"tracked={tracked.quantity}, actual={abs(exchange_pos.contracts)}"
                        )
                        results["discrepancies"].append({
                            "symbol": symbol,
                            "type": "quantity",
                            "tracked": tracked.quantity,
                            "actual": abs(exchange_pos.contracts),
                        })
                        tracked.quantity = abs(exchange_pos.contracts)
                        log.info(f"[PositionTracker] {symbol}: Fixed quantity to {tracked.quantity}")

                    # Fix entry_price if zero
                    if tracked.entry_price == 0 or tracked.entry_price is None:
                        if exchange_pos.entry_price > 0:
                            tracked.entry_price = exchange_pos.entry_price
                            log.info(f"[PositionTracker] {symbol}: Fixed entry_price to ${tracked.entry_price:.6f}")

                    # Fix side mismatch
                    exchange_side = exchange_pos.side.upper() if exchange_pos.side else ""
                    if exchange_side and tracked.side != exchange_side:
                        log.warning(
                            f"[PositionTracker] {symbol} side mismatch: "
                            f"tracked={tracked.side}, actual={exchange_side}"
                        )
                        results["discrepancies"].append({
                            "symbol": symbol,
                            "type": "side",
                            "tracked": tracked.side,
                            "actual": exchange_side,
                        })
                        tracked.side = exchange_side
                        log.info(f"[PositionTracker] {symbol}: Fixed side to {tracked.side}")

                    # Fix position value if needed
                    if tracked.position_value_usd == 0 and tracked.entry_price > 0 and tracked.quantity > 0:
                        tracked.position_value_usd = tracked.entry_price * tracked.quantity
                        log.info(f"[PositionTracker] {symbol}: Fixed position_value to ${tracked.position_value_usd:.2f}")

                    results["synced"].append(symbol)

            log.info(
                f"[PositionTracker] Sync complete. "
                f"Synced: {len(results['synced'])}, "
                f"Phantom removed: {len(results['phantom_removed'])}, "
                f"Discrepancies fixed: {len(results['discrepancies'])}"
            )

        except Exception as e:
            log.error(f"[PositionTracker] Sync failed: {e}")
            results["error"] = str(e)

        return results

    def get_position_monitor_data(self) -> list[dict[str, Any]]:
        """
        Get detailed monitoring data for all open positions.

        Returns:
            List of position monitoring snapshots
        """
        monitor_data = []

        for symbol, position in self.positions.items():
            if position.status == "CLOSED":
                continue

            try:
                # Get current orders for this position
                regular_orders, algo_orders = self.binance_client.get_all_position_orders(symbol)

                # Identify order types
                sl_active = any(
                    o.order_id == position.sl_order_id for o in algo_orders
                ) if position.sl_order_id else False

                tp1_active = any(
                    o.order_id == position.tp1_order_id for o in algo_orders
                ) if position.tp1_order_id else False

                tp2_active = any(
                    o.order_id == position.tp2_order_id for o in algo_orders
                ) if position.tp2_order_id else False

                monitor_data.append({
                    "symbol": symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "current_price": position.current_price,
                    "quantity": position.quantity,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_pct": position.unrealized_pnl_pct,
                    "status": position.status,
                    "stop_loss": position.stop_loss,
                    "take_profit_1": position.take_profit_1,
                    "take_profit_2": position.take_profit_2,
                    "sl_active": sl_active,
                    "tp1_active": tp1_active,
                    "tp1_hit": position.tp1_hit,
                    "tp2_active": tp2_active,
                    "total_orders": len(regular_orders) + len(algo_orders),
                    "entry_time": position.entry_time.isoformat(),
                })

            except Exception as e:
                log.warning(f"[PositionTracker] Failed to get monitor data for {symbol}: {e}")

        return monitor_data

    def to_dict(self) -> dict[str, Any]:
        """Serialize positions for persistence."""
        return {
            "positions": {
                symbol: pos.model_dump()
                for symbol, pos in self.positions.items()
            },
            "closed_count": len(self.closed_positions),
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Load positions from saved state."""
        positions_data = data.get("positions", {})
        for symbol, pos_data in positions_data.items():
            try:
                position = TrackedPosition(**pos_data)
                if position.status != "CLOSED":
                    self.positions[symbol] = position
                    log.info(f"[PositionTracker] Restored position: {symbol}")
            except Exception as e:
                log.warning(f"[PositionTracker] Failed to restore {symbol}: {e}")

    def discover_unprotected_positions(self) -> list[dict[str, Any]]:
        """
        Find exchange positions that have no SL order.
        Useful for detecting positions that were left unprotected.

        Returns:
            List of unprotected position details
        """
        unprotected = []

        try:
            # Get all real positions from exchange
            exchange_positions = self.binance_client.get_positions()

            for pos in exchange_positions:
                # Position uses 'contracts' attribute, not 'amount'
                if pos.contracts == 0:
                    continue

                # Convert symbol format
                symbol = pos.symbol.replace('/USDT:USDT', 'USDT')

                # Get orders for this symbol
                regular_orders, algo_orders = self.binance_client.get_all_position_orders(symbol)

                # Check if there's any SL-like order
                # Algo orders with qty=0 and closePosition are SL orders
                has_sl = False
                for o in algo_orders:
                    # SL orders have quantity=0 (closePosition=true) and stop_price set
                    if o.quantity == 0 and o.stop_price > 0:
                        has_sl = True
                        break
                    # Or check order_type if available
                    if o.order_type and o.order_type.lower() in ('stop_market', 'stop'):
                        has_sl = True
                        break

                # Also check regular orders for trailing stops
                for o in regular_orders:
                    if o.order_type and o.order_type.lower() in ('stop_market', 'stop', 'trailing_stop_market'):
                        has_sl = True
                        break

                if not has_sl:
                    # Use pos.side directly (ccxt provides 'long' or 'short')
                    side = pos.side.upper() if hasattr(pos, 'side') else ('LONG' if pos.contracts > 0 else 'SHORT')
                    unprotected.append({
                        'symbol': symbol,
                        'side': side,
                        'amount': abs(pos.contracts),
                        'entry_price': pos.entry_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'orders_count': len(regular_orders) + len(algo_orders),
                    })
                    log.warning(
                        f"[PositionTracker] UNPROTECTED: {symbol} {side} {abs(pos.contracts):.4f} @ ${pos.entry_price:.4f} - NO SL!"
                    )

            if unprotected:
                log.warning(f"[PositionTracker] Found {len(unprotected)} unprotected positions!")
            else:
                log.info("[PositionTracker] All exchange positions have SL orders")

        except Exception as e:
            log.error(f"[PositionTracker] Failed to discover unprotected positions: {e}", exc_info=True)

        return unprotected

    def check_order_health(self) -> dict[str, Any]:
        """
        Check health of all orders for open positions.
        Recreates missing SL/Trailing orders if needed.
        Does NOT recreate TP orders if already filled.

        Returns:
            Health report with issues found and actions taken
        """
        report = {
            "checked": 0,
            "healthy": 0,
            "issues": [],
            "actions": [],
        }

        for symbol, position in list(self.positions.items()):
            if position.status == "CLOSED":
                continue

            # Skip invalid positions (will be cleaned up by sync)
            if position.quantity == 0 or position.entry_price == 0:
                log.warning(
                    f"[OrderHealth] {symbol}: Skipping invalid position "
                    f"(qty={position.quantity}, entry=${position.entry_price})"
                )
                continue

            report["checked"] += 1
            issues_for_symbol = []

            try:
                # Get current orders on exchange (regular + algo/SL/TP)
                regular_orders, algo_orders = self.binance_client.get_all_position_orders(symbol)
                all_orders = regular_orders + algo_orders

                order_ids = {o.order_id for o in all_orders}

                log.info(
                    f"[OrderHealth] {symbol}: Checking position | "
                    f"Side: {position.side} | TP1 hit: {position.tp1_hit} | "
                    f"SL order ID: {position.sl_order_id} | "
                    f"Trailing ID: {position.trailing_stop_order_id} | "
                    f"Found {len(regular_orders)} regular + {len(algo_orders)} algo orders"
                )

                # Log all found orders for debugging
                for o in all_orders:
                    log.info(
                        f"[OrderHealth] {symbol}: Order found: id={o.order_id}, "
                        f"type={o.order_type}, side={o.side}, stop=${o.stop_price or 0:.6f}"
                    )

                # Check SL order
                sl_exists = position.sl_order_id and position.sl_order_id in order_ids

                # Also check if there's ANY stop-market order for this symbol (even with different ID)
                # Order type is now lowercase (e.g., 'stop_market') from our parsing
                has_any_sl = any(
                    (o.order_type and o.order_type.lower() in ('stop_market', 'stop')) or
                    (o.quantity == 0 and o.stop_price > 0)  # closePosition=true SL orders
                    for o in all_orders
                )

                if not sl_exists and not position.tp1_hit:
                    if has_any_sl:
                        # SL exists but with different ID - update our tracking
                        sl_order_found = next(
                            (o for o in all_orders if o.order_type in ('stop_market', 'stop', 'STOP_MARKET', 'STOP')),
                            None
                        )
                        if sl_order_found:
                            log.info(
                                f"[OrderHealth] {symbol}: SL exists with different ID. "
                                f"Expected: {position.sl_order_id}, Found: {sl_order_found.order_id}"
                            )
                            position.sl_order_id = sl_order_found.order_id
                    else:
                        # Need SL order (before TP1)
                        issues_for_symbol.append("Missing SL order")
                        log.warning(f"[OrderHealth] {symbol}: No SL order found - recreating")

                        # Recreate SL
                        exit_side = "sell" if position.side == "LONG" else "buy"
                        sl_order = self.binance_client.place_stop_loss(
                            symbol=symbol,
                            side=exit_side,
                            quantity=position.quantity,
                            stop_price=position.stop_loss,
                        )
                        if sl_order:
                            position.sl_order_id = sl_order.order_id
                            report["actions"].append(f"{symbol}: Recreated SL @ ${position.stop_loss:.6f}")
                            log.info(f"[OrderHealth] {symbol}: Recreated SL order @ ${position.stop_loss:.6f}, ID={sl_order.order_id}")
                        else:
                            report["actions"].append(f"{symbol}: FAILED to recreate SL")
                            log.error(f"[OrderHealth] {symbol}: Failed to recreate SL order")

                # Check trailing stop (only if TP1 hit and trailing active)
                if position.tp1_hit and position.trailing_stop_active:
                    trailing_exists = position.trailing_stop_order_id and position.trailing_stop_order_id in order_ids

                    # Also check if there's ANY trailing stop order
                    has_any_trailing = any(
                        o.order_type and o.order_type.lower() in ('trailing_stop_market',)
                        for o in all_orders
                    )

                    if not trailing_exists:
                        if has_any_trailing:
                            # Trailing exists but with different ID
                            trailing_found = next(
                                (o for o in all_orders if o.order_type in ('trailing_stop_market', 'TRAILING_STOP_MARKET')),
                                None
                            )
                            if trailing_found:
                                log.info(
                                    f"[OrderHealth] {symbol}: Trailing exists with different ID. "
                                    f"Expected: {position.trailing_stop_order_id}, Found: {trailing_found.order_id}"
                                )
                                position.trailing_stop_order_id = trailing_found.order_id
                        else:
                            issues_for_symbol.append("Missing Trailing Stop order")
                            log.warning(f"[OrderHealth] {symbol}: No trailing stop found - recreating")

                            # Recreate trailing stop on exchange
                            remaining_qty = position.quantity / 2
                            exit_side = "sell" if position.side == "LONG" else "buy"
                            callback_pct = position.trailing_stop_callback * 100

                            trailing_order = self.binance_client.place_trailing_stop(
                                symbol=symbol,
                                side=exit_side,
                                quantity=remaining_qty,
                                callback_rate=callback_pct,
                            )

                            if trailing_order:
                                position.trailing_stop_order_id = trailing_order.get('order_id')
                                report["actions"].append(f"{symbol}: Recreated Trailing Stop (callback={callback_pct}%)")
                                log.info(f"[OrderHealth] {symbol}: Recreated trailing stop order, ID={position.trailing_stop_order_id}")
                            else:
                                report["actions"].append(f"{symbol}: FAILED to recreate Trailing Stop")
                                log.error(f"[OrderHealth] {symbol}: Failed to recreate trailing stop")

                # Validate SL price is correct (if order exists)
                if sl_exists and not position.tp1_hit:
                    sl_order = next((o for o in all_orders if o.order_id == position.sl_order_id), None)
                    if sl_order and sl_order.stop_price:
                        price_diff_pct = abs(sl_order.stop_price - position.stop_loss) / position.stop_loss * 100
                        if price_diff_pct > 0.5:  # More than 0.5% difference
                            issues_for_symbol.append(f"SL price mismatch: order=${sl_order.stop_price:.6f} vs expected=${position.stop_loss:.6f}")
                            log.warning(
                                f"[OrderHealth] {symbol}: SL price mismatch - "
                                f"order=${sl_order.stop_price:.6f} vs expected=${position.stop_loss:.6f}"
                            )

                # Report status
                if issues_for_symbol:
                    report["issues"].append({
                        "symbol": symbol,
                        "issues": issues_for_symbol,
                    })
                else:
                    report["healthy"] += 1
                    log.info(f"[OrderHealth] {symbol}: Position is healthy")

            except Exception as e:
                log.error(f"[OrderHealth] Failed to check {symbol}: {e}", exc_info=True)
                report["issues"].append({
                    "symbol": symbol,
                    "issues": [f"Check failed: {e}"],
                })

        # Log summary
        if report["issues"]:
            log.warning(
                f"[OrderHealth] Checked {report['checked']} positions: "
                f"{report['healthy']} healthy, {len(report['issues'])} with issues"
            )
        else:
            log.info(f"[OrderHealth] All {report['checked']} positions healthy")

        return report

    def cleanup_orphan_orders(self) -> dict[str, Any]:
        """
        Find and cancel orphan orders (orders for positions that don't exist)
        and wrong-direction orders (e.g., TP SELL for SHORT position).

        Returns:
            Cleanup report with cancelled orders
        """
        report = {
            "orphan_orders_cancelled": [],
            "wrong_direction_cancelled": [],
            "errors": [],
        }

        try:
            # Get all real positions from exchange
            exchange_positions = self.binance_client.get_positions()

            # Build map: binance_symbol -> position_side
            position_map = {}
            for pos in exchange_positions:
                if abs(pos.contracts) > 0:
                    # Convert symbol format: "POL/USDT:USDT" -> "POLUSDT"
                    binance_symbol = pos.symbol.replace('/USDT:USDT', 'USDT')
                    position_map[binance_symbol] = pos.side.upper()  # "LONG" or "SHORT"

            log.info(f"[OrphanCleanup] Real positions: {list(position_map.keys())}")

            # Get all open algo orders (without symbol filter to get ALL)
            try:
                response = self.binance_client.exchange.fapiPrivateGetOpenAlgoOrders()
                if isinstance(response, list):
                    all_algo_orders = response
                else:
                    all_algo_orders = response.get('orders', [])
            except Exception as e:
                log.error(f"[OrphanCleanup] Failed to get all algo orders: {e}")
                report["errors"].append(f"Failed to get algo orders: {e}")
                return report

            log.info(f"[OrphanCleanup] Found {len(all_algo_orders)} total algo orders")

            for order in all_algo_orders:
                order_symbol = order.get('symbol', '')  # POLUSDT format
                algo_id = str(order.get('algoId', ''))
                order_side = order.get('side', '').upper()  # BUY or SELL
                order_type = (
                    order.get('strategyType', '') or
                    order.get('type', '') or
                    order.get('orderType', '')
                ).upper()
                trigger_price = float(order.get('triggerPrice', 0) or 0)

                log.debug(
                    f"[OrphanCleanup] Checking order: {order_symbol} {order_type} "
                    f"{order_side} @ ${trigger_price:.6f} (ID: {algo_id})"
                )

                # Check if position exists
                if order_symbol not in position_map:
                    # ORPHAN ORDER - position doesn't exist
                    log.warning(
                        f"[OrphanCleanup] ORPHAN: {order_symbol} {order_type} {order_side} @ ${trigger_price:.6f} "
                        f"(no position exists) - cancelling"
                    )
                    ccxt_symbol = self.binance_client._to_ccxt_symbol(order_symbol)
                    try:
                        success = self.binance_client.cancel_algo_order(ccxt_symbol, algo_id)
                        if success:
                            report["orphan_orders_cancelled"].append({
                                "symbol": order_symbol,
                                "type": order_type,
                                "side": order_side,
                                "trigger_price": trigger_price,
                                "algo_id": algo_id,
                            })
                            log.info(f"[OrphanCleanup] Cancelled orphan order {algo_id}")
                    except Exception as cancel_e:
                        report["errors"].append(f"Failed to cancel orphan {algo_id}: {cancel_e}")
                        log.error(f"[OrphanCleanup] Failed to cancel orphan order {algo_id}: {cancel_e}")
                    continue

                # Position exists - check order direction
                position_side = position_map[order_symbol]

                # Determine expected exit side: LONG position exits with SELL, SHORT with BUY
                expected_exit_side = "SELL" if position_side == "LONG" else "BUY"

                # Check if this is an exit order (TP or SL)
                is_exit_order = order_type in (
                    'TAKE_PROFIT_MARKET', 'TAKE_PROFIT',
                    'STOP_MARKET', 'STOP',
                    'TRAILING_STOP_MARKET'
                )

                if is_exit_order and order_side != expected_exit_side:
                    # WRONG DIRECTION - TP/SL is in wrong direction
                    log.warning(
                        f"[OrphanCleanup] WRONG DIRECTION: {order_symbol} {order_type} {order_side} "
                        f"for {position_side} position (should be {expected_exit_side}) - cancelling"
                    )
                    ccxt_symbol = self.binance_client._to_ccxt_symbol(order_symbol)
                    try:
                        success = self.binance_client.cancel_algo_order(ccxt_symbol, algo_id)
                        if success:
                            report["wrong_direction_cancelled"].append({
                                "symbol": order_symbol,
                                "type": order_type,
                                "side": order_side,
                                "position_side": position_side,
                                "expected_side": expected_exit_side,
                                "trigger_price": trigger_price,
                                "algo_id": algo_id,
                            })
                            log.info(f"[OrphanCleanup] Cancelled wrong-direction order {algo_id}")
                    except Exception as cancel_e:
                        report["errors"].append(f"Failed to cancel wrong-direction {algo_id}: {cancel_e}")
                        log.error(f"[OrphanCleanup] Failed to cancel wrong-direction order {algo_id}: {cancel_e}")

            # Summary
            total_cancelled = len(report["orphan_orders_cancelled"]) + len(report["wrong_direction_cancelled"])
            if total_cancelled > 0:
                log.warning(
                    f"[OrphanCleanup] Cleaned up {total_cancelled} orders: "
                    f"{len(report['orphan_orders_cancelled'])} orphan, "
                    f"{len(report['wrong_direction_cancelled'])} wrong-direction"
                )
            else:
                log.info("[OrphanCleanup] No orphan or wrong-direction orders found")

        except Exception as e:
            log.error(f"[OrphanCleanup] Cleanup failed: {e}", exc_info=True)
            report["errors"].append(f"Cleanup failed: {e}")

        return report
