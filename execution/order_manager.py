"""
Order Manager for AI Trading System V3.
Provides safe order execution with retry logic, validation, and slippage calculation.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import Order
from execution.binance_client import BinanceClient

log = get_logger("order_manager")


# Minimum position sizes (in base currency)
MIN_POSITION_SIZES: dict[str, float] = {
    'DOGEUSDT': 10.0,      # ~$1
    'XRPUSDT': 1.0,        # ~$0.5
    'ADAUSDT': 1.0,        # ~$0.4
    '1000PEPEUSDT': 100.0, # ~$1
    'SOLUSDT': 0.1,        # ~$15
    'ETHUSDT': 0.01,       # ~$25
    'BTCUSDT': 0.001,      # ~$65
}

# Default minimum if symbol not in list
DEFAULT_MIN_SIZE = 1.0


class OrderManager:
    """
    Safe order execution wrapper with retry logic and validation.

    Features:
    - Automatic retry with exponential backoff
    - Position size validation
    - Slippage calculation
    - Order logging
    """

    def __init__(
        self,
        client: BinanceClient,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """
        Initialize OrderManager.

        Args:
            client: BinanceClient instance
            max_retries: Maximum retry attempts for failed orders
            base_delay: Base delay for exponential backoff (seconds)
        """
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Track order statistics
        self.stats = {
            'orders_placed': 0,
            'orders_failed': 0,
            'total_slippage': 0.0,
            'retries_used': 0,
        }

    def get_min_quantity(self, symbol: str) -> float:
        """
        Get minimum order quantity for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Minimum quantity
        """
        # Normalize symbol
        clean_symbol = symbol.replace('/', '').replace(':USDT', '')
        return MIN_POSITION_SIZES.get(clean_symbol, DEFAULT_MIN_SIZE)

    def validate_order(
        self,
        symbol: str,
        quantity: float,
        price: float | None = None
    ) -> tuple[bool, str]:
        """
        Validate order parameters before execution.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (for notional calculation)

        Returns:
            Tuple of (is_valid, error_message)
        """
        min_qty = self.get_min_quantity(symbol)

        if quantity < min_qty:
            return False, f"Quantity {quantity} below minimum {min_qty} for {symbol}"

        if price:
            notional = quantity * price
            if notional < 5.0:  # Binance minimum
                return False, f"Notional value ${notional:.2f} below minimum $5.00"

        return True, "OK"

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Order | None:
        """
        Place market order with retry logic.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            reduce_only: Only reduce position

        Returns:
            Order object or None if failed
        """
        # Get current price for validation
        current_price = self.client.get_current_price(symbol)

        if not reduce_only:
            is_valid, error = self.validate_order(symbol, quantity, current_price)
            if not is_valid:
                log.error(f"Order validation failed: {error}")
                return None

        # Retry loop
        last_error = None

        for attempt in range(self.max_retries):
            try:
                order = self.client.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    reduce_only=reduce_only,
                )

                if order:
                    self.stats['orders_placed'] += 1

                    # Calculate slippage if we have average price
                    if order.average_price and current_price:
                        slippage = abs(order.average_price - current_price) / current_price * 100
                        self.stats['total_slippage'] += slippage
                        log.info(f"Order filled with {slippage:.3f}% slippage")

                    if attempt > 0:
                        self.stats['retries_used'] += attempt

                    return order

            except Exception as e:
                last_error = e
                log.warning(f"Order attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    log.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        self.stats['orders_failed'] += 1
        log.error(f"Market order failed after {self.max_retries} attempts: {last_error}")
        return None

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        reduce_only: bool = False
    ) -> Order | None:
        """
        Place limit order with retry logic.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price
            reduce_only: Only reduce position

        Returns:
            Order object or None if failed
        """
        if not reduce_only:
            is_valid, error = self.validate_order(symbol, quantity, price)
            if not is_valid:
                log.error(f"Order validation failed: {error}")
                return None

        last_error = None

        for attempt in range(self.max_retries):
            try:
                order = self.client.place_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    reduce_only=reduce_only,
                )

                if order:
                    self.stats['orders_placed'] += 1
                    if attempt > 0:
                        self.stats['retries_used'] += attempt
                    return order

            except Exception as e:
                last_error = e
                log.warning(f"Limit order attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)

        self.stats['orders_failed'] += 1
        log.error(f"Limit order failed after {self.max_retries} attempts: {last_error}")
        return None

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Order | None:
        """
        Place stop-loss order with retry logic.

        Args:
            symbol: Trading symbol
            side: "buy" for short SL, "sell" for long SL
            quantity: Order quantity
            stop_price: Stop trigger price

        Returns:
            Order object or None if failed
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                order = self.client.place_stop_loss(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    stop_price=stop_price,
                )

                if order:
                    self.stats['orders_placed'] += 1
                    return order

            except Exception as e:
                last_error = e
                log.warning(f"Stop loss attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)

        self.stats['orders_failed'] += 1
        log.error(f"Stop loss order failed: {last_error}")
        return None

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float
    ) -> Order | None:
        """
        Place take-profit order with retry logic.

        Args:
            symbol: Trading symbol
            side: "buy" for short TP, "sell" for long TP
            quantity: Order quantity
            tp_price: Take-profit trigger price

        Returns:
            Order object or None if failed
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                order = self.client.place_take_profit(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    tp_price=tp_price,
                )

                if order:
                    self.stats['orders_placed'] += 1
                    return order

            except Exception as e:
                last_error = e
                log.warning(f"Take profit attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)

        self.stats['orders_failed'] += 1
        log.error(f"Take profit order failed: {last_error}")
        return None

    def close_position(self, symbol: str) -> Order | None:
        """
        Close position with retry logic.

        Args:
            symbol: Trading symbol

        Returns:
            Order object or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                order = self.client.close_position(symbol)
                if order:
                    log.info(f"Position closed for {symbol}")
                    return order
                return None

            except Exception as e:
                log.warning(f"Close position attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)

        log.error(f"Failed to close position for {symbol}")
        return None

    def get_stats(self) -> dict[str, Any]:
        """
        Get order execution statistics.

        Returns:
            Dict with order stats
        """
        avg_slippage = 0.0
        if self.stats['orders_placed'] > 0:
            avg_slippage = self.stats['total_slippage'] / self.stats['orders_placed']

        return {
            'orders_placed': self.stats['orders_placed'],
            'orders_failed': self.stats['orders_failed'],
            'success_rate': self.stats['orders_placed'] / max(1, self.stats['orders_placed'] + self.stats['orders_failed']) * 100,
            'average_slippage_pct': avg_slippage,
            'total_retries': self.stats['retries_used'],
        }
