"""
Binance Futures client for AI Trading System V3.
Handles all exchange connectivity, data fetching, and order execution.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import ccxt
import httpx

from core.logger import get_logger
from core.state import FundingData, OHLCVBar, Order, Position

log = get_logger("binance_client")


class BinanceClient:
    """
    Binance USDT-M Futures client using ccxt.

    Handles:
    - Connection and authentication
    - Account info (balance, positions)
    - Market data (prices, OHLCV, funding rates)
    - Order execution (market, limit, stop-loss, take-profit)
    """

    # Minimum notional value for Binance Futures orders
    MIN_NOTIONAL_USDT = 5.0

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = False,
    ) -> None:
        """
        Initialize Binance Futures client.

        Args:
            api_key: Binance API key (or reads from BINANCE_API_KEY env)
            api_secret: Binance API secret (or reads from BINANCE_API_SECRET env)
            testnet: Use testnet instead of mainnet
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self.testnet = testnet

        # Initialize ccxt binanceusdm (USDT-M Futures specific)
        self.exchange = ccxt.binanceusdm({
            'apiKey': self.api_key.strip(),
            'secret': self.api_secret.strip(),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
                'warnOnFetchOpenOrdersWithoutSymbol': False,
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)
            log.info("Using Binance TESTNET")
        else:
            log.info("Using Binance MAINNET")

        # Cache for market filters
        self._market_filters: dict[str, dict] = {}
        self._leverage_cache: dict[str, int] = {}
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to Binance and verify credentials.

        Returns:
            True if connected successfully

        Raises:
            ConnectionError: If connection fails
        """
        max_retries = 3
        retry_delays = [2, 4, 8]

        for attempt in range(max_retries):
            try:
                # Synchronize time with server
                server_time = self.exchange.fetch_time()
                local_time = int(time.time() * 1000)
                offset = server_time - local_time
                self.exchange.options['timeDifference'] = offset

                if abs(offset) > 5000:
                    log.warning(f"Large time offset: {offset}ms - consider syncing system clock")

                log.info(f"Time synchronized: offset={offset}ms")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect to Binance after {max_retries} attempts: {e}")

        # Load markets
        try:
            markets = self.exchange.load_markets()
            futures_count = 0

            for symbol, market in markets.items():
                if market.get('type') == 'swap' and market.get('quote') == 'USDT':
                    info = market.get('info', {})
                    filters = {f['filterType']: f for f in info.get('filters', [])}

                    self._market_filters[symbol] = {
                        'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('notional', self.MIN_NOTIONAL_USDT)),
                        'min_amount': float(filters.get('LOT_SIZE', {}).get('minQty', 0.001)),
                        'price_precision': market.get('precision', {}).get('price', 2),
                        'amount_precision': market.get('precision', {}).get('amount', 3),
                    }
                    futures_count += 1

            log.info(f"Loaded {futures_count} USDT-M Futures markets")

        except Exception as e:
            log.warning(f"Failed to load markets: {e}. Using defaults.")

        # Verify API credentials
        try:
            account_info = self.exchange.fapiPrivateV2GetAccount()
            can_trade = account_info.get('canTrade', False)
            total_balance = float(account_info.get('totalWalletBalance', 0.0))

            if not can_trade:
                raise ConnectionError("Account cannot trade. Enable Futures in API settings.")

            log.info(f"Connected to Binance Futures. Balance: ${total_balance:.2f} USDT")
            self._connected = True
            return True

        except ccxt.AuthenticationError as e:
            log.error(f"Authentication failed: {e}")
            raise ConnectionError(f"Binance authentication failed: {e}")

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_balance(self) -> float:
        """
        Get USDT balance for futures account.

        Returns:
            Total USDT balance
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})
            usdt_balance = balance.get('USDT', {})
            total = float(usdt_balance.get('total', 0.0))
            log.debug(f"Balance: ${total:.2f} USDT")
            return total
        except Exception as e:
            log.error(f"Failed to fetch balance: {e}")
            return 0.0

    def get_free_balance(self) -> float:
        """
        Get available (free) USDT balance.

        Returns:
            Free USDT balance
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})
            usdt_balance = balance.get('USDT', {})
            free = float(usdt_balance.get('free', 0.0))
            return free
        except Exception as e:
            log.error(f"Failed to fetch free balance: {e}")
            return 0.0

    def get_equity(self) -> float:
        """
        Get total account equity (wallet balance + unrealized PnL).

        This is the true account value including open positions.
        Use this for drawdown calculations instead of wallet balance.

        Returns:
            Total equity in USDT
        """
        try:
            account_info = self.exchange.fapiPrivateV2GetAccount()
            # totalMarginBalance = walletBalance + unrealizedProfit
            equity = float(account_info.get('totalMarginBalance', 0.0))
            log.debug(f"Equity: ${equity:.2f} USDT")
            return equity
        except Exception as e:
            log.error(f"Failed to fetch equity: {e}")
            # Fallback to wallet balance
            return self.get_balance()

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open Position objects
        """
        try:
            params = {}
            if symbol:
                binance_symbol = self._to_binance_symbol(symbol)
                params['symbol'] = binance_symbol

            positions_data = self.exchange.fapiPrivateV2GetPositionRisk(params)
            positions = []

            for pos in positions_data:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt == 0:
                    continue

                side = 'long' if position_amt > 0 else 'short'
                contracts = abs(position_amt)

                # Convert symbol format
                symbol_binance = pos['symbol']
                if symbol_binance.endswith('USDT'):
                    base = symbol_binance[:-4]
                    symbol_ccxt = f"{base}/USDT:USDT"
                else:
                    symbol_ccxt = symbol_binance

                positions.append(Position(
                    symbol=symbol_ccxt,
                    side=side,
                    contracts=contracts,
                    entry_price=float(pos.get('entryPrice', 0)),
                    mark_price=float(pos.get('markPrice', 0)),
                    unrealized_pnl=float(pos.get('unRealizedProfit', 0)),
                    leverage=int(pos.get('leverage', 1)),
                    margin_type=pos.get('marginType', 'isolated').lower(),
                    liquidation_price=float(pos.get('liquidationPrice', 0)),
                ))

            if positions:
                log.debug(f"Found {len(positions)} open position(s)")

            return positions

        except Exception as e:
            log.error(f"Failed to fetch positions: {e}")
            return []

    def get_open_orders(self, symbol: str) -> list[Order]:
        """
        Get open orders for a symbol.

        Args:
            symbol: Trading symbol (e.g., "DOGE/USDT:USDT")

        Returns:
            List of open Order objects
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            open_orders = self.exchange.fetch_open_orders(ccxt_symbol)
            orders = []

            for o in open_orders:
                orders.append(Order(
                    order_id=str(o['id']),
                    symbol=ccxt_symbol,
                    side=o['side'],
                    order_type=o['type'],
                    quantity=float(o.get('amount', 0)),
                    price=float(o.get('price')) if o.get('price') else None,
                    stop_price=float(o.get('stopPrice')) if o.get('stopPrice') else None,
                    status=o['status'],
                    filled_quantity=float(o.get('filled', 0)),
                    average_price=float(o.get('average')) if o.get('average') else None,
                    timestamp=datetime.fromtimestamp(o['timestamp'] / 1000, tz=timezone.utc),
                ))

            return orders

        except Exception as e:
            log.error(f"Failed to fetch open orders for {symbol}: {e}")
            return []

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            price = float(ticker.get('last') or ticker.get('close') or 0)
            return price
        except Exception as e:
            log.error(f"Failed to fetch price for {symbol}: {e}")
            return 0.0

    def get_ticker_24h(self, symbol: str) -> dict[str, Any]:
        """
        Get 24h ticker data for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with price, change, volume info
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            ticker = self.exchange.fetch_ticker(ccxt_symbol)

            return {
                'symbol': symbol,
                'last': float(ticker.get('last', 0)),
                'bid': float(ticker.get('bid', 0)),
                'ask': float(ticker.get('ask', 0)),
                'high': float(ticker.get('high', 0)),
                'low': float(ticker.get('low', 0)),
                'volume': float(ticker.get('baseVolume', 0)),
                'quote_volume': float(ticker.get('quoteVolume', 0)),
                'change_pct': float(ticker.get('percentage', 0)),
            }
        except Exception as e:
            log.error(f"Failed to fetch 24h ticker for {symbol}: {e}")
            return {}

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100
    ) -> list[OHLCVBar]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch

        Returns:
            List of OHLCVBar objects
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)

            bars = []
            for candle in ohlcv:
                bars.append(OHLCVBar(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                ))

            return bars

        except Exception as e:
            log.error(f"Failed to fetch OHLCV for {symbol} {timeframe}: {e}")
            return []

    def get_funding_rate(self, symbol: str) -> FundingData | None:
        """
        Get funding rate data for a futures symbol.

        Args:
            symbol: Trading symbol

        Returns:
            FundingData object or None
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Get current and predicted funding rate
            premium_index = self.exchange.fapiPublicGetPremiumIndex({'symbol': binance_symbol})

            current_rate = float(premium_index.get('lastFundingRate', 0))
            predicted_rate = float(premium_index.get('interestRate', 0))  # Approximation
            next_funding_time = datetime.fromtimestamp(
                int(premium_index.get('nextFundingTime', 0)) / 1000,
                tz=timezone.utc
            )

            # Get historical rates to determine trend
            funding_history = self.exchange.fapiPublicGetFundingRate({
                'symbol': binance_symbol,
                'limit': 5
            })

            rates = [float(f['fundingRate']) for f in funding_history]
            trend = self._determine_funding_trend(rates)

            return FundingData(
                symbol=symbol,
                current_rate=current_rate,
                predicted_rate=predicted_rate,
                next_funding_time=next_funding_time,
                rate_trend=trend,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            log.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    def _determine_funding_trend(self, rates: list[float]) -> str:
        """
        Determine funding rate trend from historical rates.

        Args:
            rates: List of historical funding rates (newest first)

        Returns:
            "rising", "falling", or "stable"
        """
        if len(rates) < 3:
            return "stable"

        # Check last 3 rates
        recent = rates[:3]

        if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            return "rising"
        elif all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            return "falling"
        else:
            return "stable"

    def get_long_short_ratio(self, symbol: str, period: str = "5m") -> dict[str, Any] | None:
        """
        Get long/short ratio for top traders.

        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)

        Returns:
            Dict with long_ratio, short_ratio, long_short_ratio or None
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Use direct API call since ccxt doesn't support this endpoint
            url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
            params = {
                'symbol': binance_symbol,
                'period': period,
                'limit': 1,
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data and len(data) > 0:
                latest = data[0]
                return {
                    'symbol': symbol,
                    'long_ratio': float(latest.get('longAccount', 0.5)),
                    'short_ratio': float(latest.get('shortAccount', 0.5)),
                    'long_short_ratio': float(latest.get('longShortRatio', 1.0)),
                    'timestamp': int(latest.get('timestamp', 0)),
                }

            return None

        except Exception as e:
            log.error(f"Failed to fetch long/short ratio for {symbol}: {e}")
            return None

    def get_open_interest(self, symbol: str) -> dict[str, Any] | None:
        """
        Get open interest data.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with open_interest, open_interest_value or None
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Use direct API call
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {'symbol': binance_symbol}

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data:
                return {
                    'symbol': symbol,
                    'open_interest': float(data.get('openInterest', 0)),
                    'timestamp': int(data.get('time', 0)),
                }

            return None

        except Exception as e:
            log.error(f"Failed to fetch open interest for {symbol}: {e}")
            return None

    def get_open_interest_history(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict[str, Any]]:
        """
        Get historical open interest data.

        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points

        Returns:
            List of open interest records
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Use direct API call
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {
                'symbol': binance_symbol,
                'period': period,
                'limit': limit,
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data:
                return [
                    {
                        'open_interest': float(r.get('sumOpenInterest', 0)),
                        'open_interest_value': float(r.get('sumOpenInterestValue', 0)),
                        'timestamp': int(r.get('timestamp', 0)),
                    }
                    for r in data
                ]

            return []

        except Exception as e:
            log.error(f"Failed to fetch open interest history for {symbol}: {e}")
            return []

    def get_taker_buy_sell_volume(self, symbol: str, period: str = "5m", limit: int = 30) -> list[dict[str, Any]]:
        """
        Get taker buy/sell volume ratio.

        Args:
            symbol: Trading symbol
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points

        Returns:
            List of buy/sell volume records
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Use direct API call
            url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
            params = {
                'symbol': binance_symbol,
                'period': period,
                'limit': limit,
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if data:
                return [
                    {
                        'buy_sell_ratio': float(r.get('buySellRatio', 1.0)),
                        'buy_vol': float(r.get('buyVol', 0)),
                        'sell_vol': float(r.get('sellVol', 0)),
                        'timestamp': int(r.get('timestamp', 0)),
                    }
                    for r in data
                ]

            return []

        except Exception as e:
            log.error(f"Failed to fetch taker buy/sell volume for {symbol}: {e}")
            return []

    # =========================================================================
    # TRADING
    # =========================================================================

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage (1-125)

        Returns:
            True if successful
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            self.exchange.set_leverage(leverage, ccxt_symbol)
            self._leverage_cache[symbol] = leverage
            log.info(f"Set leverage for {symbol}: {leverage}x")
            return True
        except Exception as e:
            log.error(f"Failed to set leverage for {symbol}: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> bool:
        """
        Set margin type for a symbol.

        Args:
            symbol: Trading symbol
            margin_type: "ISOLATED" or "CROSS"

        Returns:
            True if successful
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            margin_mode = 'isolated' if margin_type.upper() == 'ISOLATED' else 'cross'
            self.exchange.set_margin_mode(margin_mode, ccxt_symbol)
            log.info(f"Set margin type for {symbol}: {margin_type}")
            return True
        except Exception as e:
            # Often fails if already set to the same mode
            if 'No need to change margin type' in str(e):
                log.debug(f"Margin type already {margin_type} for {symbol}")
                return True
            log.error(f"Failed to set margin type for {symbol}: {e}")
            return False

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Order | None:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            reduce_only: If True, only reduces position

        Returns:
            Order object or None if failed
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            params = {'reduceOnly': reduce_only} if reduce_only else {}

            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='market',
                side=side,
                amount=quantity,
                params=params,
            )

            log.info(f"Market order: {side.upper()} {quantity} {symbol} - Status: {order.get('status')}")

            return Order(
                order_id=str(order['id']),
                symbol=ccxt_symbol,
                side=side,
                order_type='market',
                quantity=quantity,
                status=order.get('status', 'unknown'),
                filled_quantity=float(order.get('filled', 0)),
                average_price=float(order.get('average')) if order.get('average') else None,
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc),
            )

        except ccxt.InvalidOrder as e:
            log.error(f"Invalid order: {e}")
            return None
        except Exception as e:
            log.error(f"Failed to place market order: {e}")
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
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Limit price
            reduce_only: If True, only reduces position

        Returns:
            Order object or None if failed
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            params = {
                'timeInForce': 'GTC',
            }
            if reduce_only:
                params['reduceOnly'] = True

            order = self.exchange.create_order(
                symbol=ccxt_symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=price,
                params=params,
            )

            log.info(f"Limit order: {side.upper()} {quantity} {symbol} @ ${price:.4f}")

            return Order(
                order_id=str(order['id']),
                symbol=ccxt_symbol,
                side=side,
                order_type='limit',
                quantity=quantity,
                price=price,
                status=order.get('status', 'unknown'),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc),
            )

        except Exception as e:
            log.error(f"Failed to place limit order: {e}")
            return None

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Order | None:
        """
        Place a stop-loss order using Algo Order API.

        Args:
            symbol: Trading symbol
            side: "buy" for short position SL, "sell" for long position SL
            quantity: Order quantity
            stop_price: Stop trigger price

        Returns:
            Order object or None if failed
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            binance_symbol = self._to_binance_symbol(symbol)

            # Format price with correct precision
            formatted_stop_price = self.exchange.price_to_precision(ccxt_symbol, stop_price)

            # Use Algo Order API (required since 2025-12-09)
            algo_params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'type': 'STOP_MARKET',
                'algoType': 'CONDITIONAL',
                'triggerPrice': str(formatted_stop_price),
                'closePosition': 'true',
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE',
            }

            response = self.exchange.fapiPrivatePostAlgoOrder(algo_params)

            if response:
                algo_id = response.get('algoId')
                log.info(f"Stop Loss created: {side.upper()} @ ${formatted_stop_price} (algoId={algo_id})")

                return Order(
                    order_id=str(algo_id),
                    symbol=ccxt_symbol,
                    side=side,
                    order_type='stop_market',
                    quantity=quantity,
                    stop_price=float(formatted_stop_price),
                    status='open',
                    timestamp=datetime.now(timezone.utc),
                )

            return None

        except Exception as e:
            error_str = str(e)
            if '-4130' in error_str:
                # SL already exists - cancel old one and create new
                log.info(f"Stop Loss already exists for {symbol} - cancelling old and creating new")
                self.cancel_algo_orders(symbol)

                # Retry creating SL
                try:
                    response = self.exchange.fapiPrivatePostAlgoOrder(algo_params)
                    if response:
                        algo_id = response.get('algoId')
                        log.info(f"Stop Loss created after cancel: {side.upper()} @ ${formatted_stop_price} (algoId={algo_id})")
                        return Order(
                            order_id=str(algo_id),
                            symbol=ccxt_symbol,
                            side=side,
                            order_type='stop_market',
                            quantity=quantity,
                            stop_price=float(formatted_stop_price),
                            status='open',
                            timestamp=datetime.now(timezone.utc),
                        )
                except Exception as retry_e:
                    log.error(f"Failed to place SL after cancel: {retry_e}")

            log.error(f"Failed to place stop loss: {e}")
            return None

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float
    ) -> Order | None:
        """
        Place a take-profit order using Algo Order API.

        Args:
            symbol: Trading symbol
            side: "buy" for short position TP, "sell" for long position TP
            quantity: Order quantity
            tp_price: Take-profit trigger price

        Returns:
            Order object or None if failed
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            binance_symbol = self._to_binance_symbol(symbol)

            # Round quantity to correct precision
            formatted_quantity = float(self.exchange.amount_to_precision(ccxt_symbol, quantity))
            formatted_tp_price = self.exchange.price_to_precision(ccxt_symbol, tp_price)

            # Use Algo Order API
            algo_params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'type': 'TAKE_PROFIT_MARKET',
                'algoType': 'CONDITIONAL',
                'quantity': str(formatted_quantity),
                'triggerPrice': str(formatted_tp_price),
                'reduceOnly': 'true',
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE',
            }

            response = self.exchange.fapiPrivatePostAlgoOrder(algo_params)

            if response:
                algo_id = response.get('algoId')
                log.info(f"Take Profit created: {side.upper()} {formatted_quantity} @ ${formatted_tp_price}")

                return Order(
                    order_id=str(algo_id),
                    symbol=ccxt_symbol,
                    side=side,
                    order_type='take_profit_market',
                    quantity=formatted_quantity,
                    stop_price=float(formatted_tp_price),
                    status='open',
                    timestamp=datetime.now(timezone.utc),
                )

            return None

        except Exception as e:
            log.error(f"Failed to place take profit: {e}")
            return None

    def get_order(self, symbol: str, order_id: str) -> Order | None:
        """
        Get order details by ID.

        Args:
            symbol: Trading symbol
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            order = self.exchange.fetch_order(order_id, ccxt_symbol)

            if order:
                return Order(
                    order_id=str(order['id']),
                    symbol=ccxt_symbol,
                    side=order['side'],
                    order_type=order['type'],
                    quantity=float(order.get('amount', 0)),
                    price=float(order.get('price')) if order.get('price') else None,
                    stop_price=float(order.get('stopPrice')) if order.get('stopPrice') else None,
                    status=order['status'],
                    filled_quantity=float(order.get('filled', 0)),
                    average_price=float(order.get('average')) if order.get('average') else None,
                    timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc),
                )

            return None

        except Exception as e:
            log.error(f"Failed to fetch order {order_id} for {symbol}: {e}")
            return None

    def get_algo_order(self, symbol: str, algo_id: str) -> Order | None:
        """
        Get algo order details (SL/TP) by ID.

        Args:
            symbol: Trading symbol
            algo_id: Algo order ID

        Returns:
            Order object or None if not found
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)
            ccxt_symbol = self._to_ccxt_symbol(symbol)

            # Check historical algo orders for filled orders
            response = self.exchange.fapiPrivateGetAllAlgoOrders({
                'symbol': binance_symbol,
                'algoId': str(algo_id),
            })

            # Response can be a list directly or a dict with 'orders' key
            if isinstance(response, list):
                orders = response
            else:
                orders = response.get('orders', [])
            if orders:
                order = orders[0]
                status = order.get('algoStatus', '').lower()

                # Map Binance algo status to standard status
                status_map = {
                    'executed': 'closed',
                    'cancelled': 'canceled',
                    'expired': 'expired',
                    'new': 'open',
                    'working': 'open',
                }

                return Order(
                    order_id=str(order.get('algoId')),
                    symbol=ccxt_symbol,
                    side=order.get('side', '').lower(),
                    order_type=order.get('type', '').lower(),
                    quantity=float(order.get('quantity', 0)),
                    stop_price=float(order.get('triggerPrice', 0)),
                    status=status_map.get(status, status),
                    filled_quantity=float(order.get('executedQty', 0)),
                    average_price=float(order.get('avgPrice')) if order.get('avgPrice') else None,
                    timestamp=datetime.fromtimestamp(
                        int(order.get('updateTime', 0)) / 1000, tz=timezone.utc
                    ) if order.get('updateTime') else datetime.now(timezone.utc),
                )

            return None

        except Exception as e:
            log.error(f"Failed to fetch algo order {algo_id} for {symbol}: {e}")
            return None

    def get_all_position_orders(self, symbol: str) -> tuple[list[Order], list[Order]]:
        """
        Get all orders (regular and algo) for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (regular_orders, algo_orders)
        """
        regular_orders = self.get_open_orders(symbol)
        algo_orders = []

        try:
            binance_symbol = self._to_binance_symbol(symbol)
            ccxt_symbol = self._to_ccxt_symbol(symbol)

            # Try the new algo orders endpoint
            try:
                response = self.exchange.fapiPrivateGetOpenAlgoOrders({
                    'symbol': binance_symbol
                })
            except Exception as api_e:
                # Fallback: try without symbol filter and filter manually
                log.warning(f"fapiPrivateGetOpenAlgoOrders failed for {symbol}: {api_e}")
                try:
                    response = self.exchange.fapiPrivateGetOpenAlgoOrders()
                    # Filter by symbol
                    if isinstance(response, dict) and 'orders' in response:
                        response['orders'] = [
                            o for o in response.get('orders', [])
                            if o.get('symbol') == binance_symbol
                        ]
                    elif isinstance(response, list):
                        response = [o for o in response if o.get('symbol') == binance_symbol]
                except Exception:
                    response = {'orders': []}

            # Response can be a list directly or a dict with 'orders' key
            orders_list = response if isinstance(response, list) else response.get('orders', [])

            log.debug(f"[{symbol}] Found {len(orders_list)} algo orders")

            for order in orders_list:
                algo_id = str(order.get('algoId', ''))
                # Binance returns 'strategyType' or 'type' depending on endpoint version
                order_type = (
                    order.get('strategyType', '') or
                    order.get('type', '') or
                    order.get('orderType', '') or
                    ''
                ).lower()
                trigger_price = float(order.get('triggerPrice', 0) or 0)
                quantity = float(order.get('quantity', 0) or 0)

                # Log raw order data for debugging
                log.debug(f"[{symbol}] Raw algo order: {order}")
                log.debug(
                    f"[{symbol}] Algo order: id={algo_id}, type={order_type}, "
                    f"trigger=${trigger_price:.6f}, qty={quantity}"
                )

                algo_orders.append(Order(
                    order_id=algo_id,
                    symbol=ccxt_symbol,
                    side=order.get('side', '').lower(),
                    order_type=order_type,
                    quantity=quantity,
                    stop_price=trigger_price,
                    status='open',
                    timestamp=datetime.fromtimestamp(
                        int(order.get('bookTime', 0)) / 1000, tz=timezone.utc
                    ) if order.get('bookTime') else datetime.now(timezone.utc),
                ))

        except Exception as e:
            log.error(f"Failed to fetch algo orders for {symbol}: {e}")

        log.debug(f"[{symbol}] Total orders: {len(regular_orders)} regular + {len(algo_orders)} algo")
        return regular_orders, algo_orders

    def cancel_all_position_orders(self, symbol: str) -> int:
        """
        Cancel ALL orders for a position (both regular and algo orders).

        Args:
            symbol: Trading symbol

        Returns:
            Total number of orders cancelled
        """
        cancelled = 0

        # Cancel regular orders
        cancelled += self.cancel_all_orders(symbol)

        # Cancel algo orders (SL/TP)
        cancelled += self.cancel_algo_orders(symbol)

        log.info(f"[{symbol}] Cancelled total {cancelled} orders (regular + algo)")
        return cancelled

    def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float = 1.0,
        activation_price: float | None = None,
    ) -> dict | None:
        """
        Place a trailing stop order on Binance Futures.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell" (exit side)
            quantity: Quantity to close
            callback_rate: Callback rate in percent (e.g., 1.0 for 1%)
            activation_price: Price at which trailing starts (optional)

        Returns:
            Order info or None if failed
        """
        try:
            # Ensure markets are loaded
            if not self.exchange.markets:
                self.exchange.load_markets()

            binance_symbol = self._to_binance_symbol(symbol)
            ccxt_symbol = self._to_ccxt_symbol(symbol)

            # Format quantity with correct precision
            formatted_quantity = float(self.exchange.amount_to_precision(ccxt_symbol, quantity))

            # Use Algo Order API for trailing stop (required since 2025-12-09)
            algo_params = {
                'symbol': binance_symbol,
                'side': side.upper(),
                'type': 'TRAILING_STOP_MARKET',
                'algoType': 'CONDITIONAL',
                'quantity': str(formatted_quantity),
                'callbackRate': str(callback_rate),
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE',
                'reduceOnly': 'true',
            }

            if activation_price:
                formatted_price = self.exchange.price_to_precision(ccxt_symbol, activation_price)
                algo_params['activationPrice'] = str(formatted_price)

            response = self.exchange.fapiPrivatePostAlgoOrder(algo_params)

            if response:
                algo_id = response.get('algoId')
                log.info(
                    f"Trailing Stop created: {side.upper()} {quantity} {symbol} | "
                    f"Callback: {callback_rate}% | AlgoId: {algo_id}"
                )

                return {
                    'order_id': str(algo_id),
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'callback_rate': callback_rate,
                    'activation_price': activation_price,
                }

            return None

        except Exception as e:
            log.error(f"Failed to place trailing stop for {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            self.exchange.cancel_order(order_id, ccxt_symbol)
            log.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            log.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Number of orders cancelled
        """
        try:
            ccxt_symbol = self._to_ccxt_symbol(symbol)
            open_orders = self.exchange.fetch_open_orders(ccxt_symbol)

            if open_orders:
                self.exchange.cancel_all_orders(ccxt_symbol)
                log.info(f"Cancelled {len(open_orders)} orders for {symbol}")
                return len(open_orders)

            return 0

        except Exception as e:
            log.error(f"Failed to cancel orders for {symbol}: {e}")
            return 0

    def cancel_algo_order(self, symbol: str, algo_id: str) -> bool:
        """
        Cancel a specific algo order (SL/TP) by ID.

        Args:
            symbol: Trading symbol
            algo_id: Algo order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            self.exchange.fapiPrivateDeleteAlgoOrder({
                'symbol': binance_symbol,
                'algoId': str(algo_id)
            })

            log.info(f"Cancelled algo order {algo_id} for {symbol}")
            return True

        except Exception as e:
            log.error(f"Failed to cancel algo order {algo_id} for {symbol}: {e}")
            return False

    def cancel_algo_orders(self, symbol: str) -> int:
        """
        Cancel all algo orders (SL/TP) for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Number of algo orders cancelled
        """
        try:
            binance_symbol = self._to_binance_symbol(symbol)

            # Get open algo orders
            response = self.exchange.fapiPrivateGetOpenAlgoOrders({
                'symbol': binance_symbol
            })

            # Response can be a list directly or a dict with 'orders' key
            if isinstance(response, list):
                orders = response
            else:
                orders = response.get('orders', [])
            cancelled = 0

            for order in orders:
                algo_id = order.get('algoId')
                if algo_id:
                    try:
                        self.exchange.fapiPrivateDeleteAlgoOrder({
                            'symbol': binance_symbol,
                            'algoId': str(algo_id)
                        })
                        cancelled += 1
                        log.info(f"Cancelled algo order {algo_id} for {symbol}")
                    except Exception as e:
                        log.warning(f"Failed to cancel algo order {algo_id}: {e}")

            return cancelled

        except Exception as e:
            log.error(f"Failed to cancel algo orders for {symbol}: {e}")
            return 0

    def close_position(self, symbol: str) -> Order | None:
        """
        Close an open position by market order.

        Args:
            symbol: Trading symbol

        Returns:
            Order object or None if failed/no position
        """
        positions = self.get_positions(symbol)

        if not positions:
            log.warning(f"No open position for {symbol}")
            return None

        position = positions[0]
        close_side = 'sell' if position.side == 'long' else 'buy'

        return self.place_market_order(
            symbol=symbol,
            side=close_side,
            quantity=position.contracts,
            reduce_only=True,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _to_ccxt_symbol(self, symbol: str) -> str:
        """
        Convert symbol to ccxt format.

        "DOGEUSDT" -> "DOGE/USDT:USDT"
        "DOGE/USDT:USDT" -> "DOGE/USDT:USDT"
        """
        if ':USDT' in symbol:
            return symbol
        if '/USDT' in symbol:
            return f"{symbol}:USDT"

        # Pure symbol like "DOGEUSDT"
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"

        return symbol

    def _to_binance_symbol(self, symbol: str) -> str:
        """
        Convert symbol to Binance API format.

        "DOGE/USDT:USDT" -> "DOGEUSDT"
        "DOGEUSDT" -> "DOGEUSDT"
        """
        return symbol.replace('/', '').replace(':USDT', '')
