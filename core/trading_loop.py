"""
Trading Loop for AI Trading System V3.
Main loop that coordinates all trading components.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from core.logger import get_logger
from core.state import Decision, MarketContext, TradeExperience
from core.position_tracker import PositionTracker
from core.trade_executor import TradeExecutor
from execution.binance_client import BinanceClient
from sensors.aggregator import StateAggregator
from analyzers.market_context import MarketContextBuilder
from orchestrator.orchestrator import Orchestrator
from storage.trade_journal import TradeJournal
from notifications.telegram import TelegramNotifier

log = get_logger("trading_loop")


class TradingLoop:
    """
    Main trading loop that coordinates all components.

    Flow each cycle:
    1. Update positions (PnL)
    2. Check for closed positions (SL/TP)
    3. Collect market data
    4. Build market context
    5. Get decision from Orchestrator
    6. Execute decision
    7. Log status
    """

    def __init__(
        self,
        binance_client: BinanceClient,
        orchestrator: Orchestrator,
        executor: TradeExecutor,
        position_tracker: PositionTracker,
        state_aggregator: StateAggregator,
        context_builder: MarketContextBuilder,
        journal: TradeJournal,
        telegram: TelegramNotifier,
        symbols: list[str],
        interval_seconds: int = 300,
    ) -> None:
        """
        Initialize Trading Loop.

        Args:
            binance_client: Binance client
            orchestrator: Trading orchestrator
            executor: Trade executor
            position_tracker: Position tracker
            state_aggregator: State aggregator for market data
            context_builder: Market context builder
            journal: Trade journal
            telegram: Telegram notifier
            symbols: Symbols to trade
            interval_seconds: Seconds between analysis cycles
        """
        self.binance_client = binance_client
        self.orchestrator = orchestrator
        self.executor = executor
        self.position_tracker = position_tracker
        self.state_aggregator = state_aggregator
        self.context_builder = context_builder
        self.journal = journal
        self.telegram = telegram
        self.symbols = symbols
        self.interval_seconds = interval_seconds

        # State
        self.running = False
        self.cycle_count = 0
        self.start_time: datetime | None = None

        # Statistics
        self._total_decisions = 0
        self._trade_decisions = 0
        self._trades_executed = 0

        log.info(
            f"[TradingLoop] Initialized with {len(symbols)} symbols, "
            f"interval={interval_seconds}s"
        )

    async def run(self) -> None:
        """Main trading loop."""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        log.info("=" * 60)
        log.info("Trading Loop Started")
        log.info(f"Mode: {self.executor.mode.upper()}")
        log.info(f"Symbols: {', '.join(self.symbols)}")
        log.info(f"Interval: {self.interval_seconds}s")
        log.info("=" * 60)

        # Sync with exchange to detect phantom positions
        log.info("Syncing positions with exchange...")
        sync_report = self.position_tracker.sync_with_exchange()
        if sync_report.get('phantom_removed'):
            log.warning(f"Removed phantom positions: {sync_report['phantom_removed']}")

        # Cleanup orphan and wrong-direction orders
        log.info("Cleaning up orphan and wrong-direction orders...")
        cleanup_report = self.position_tracker.cleanup_orphan_orders()
        if cleanup_report.get('orphan_orders_cancelled') or cleanup_report.get('wrong_direction_cancelled'):
            orphan_count = len(cleanup_report.get('orphan_orders_cancelled', []))
            wrong_count = len(cleanup_report.get('wrong_direction_cancelled', []))
            log.warning(f"Cancelled {orphan_count} orphan orders, {wrong_count} wrong-direction orders")

            # Notify about cleanup
            if orphan_count + wrong_count > 0:
                orphan_list = ", ".join(
                    o['symbol'] for o in cleanup_report.get('orphan_orders_cancelled', [])
                )
                wrong_list = ", ".join(
                    f"{o['symbol']} ({o['type']})" for o in cleanup_report.get('wrong_direction_cancelled', [])
                )
                message = "🧹 Startup Cleanup\n"
                if orphan_count > 0:
                    message += f"Orphan orders cancelled ({orphan_count}): {orphan_list}\n"
                if wrong_count > 0:
                    message += f"Wrong-direction orders cancelled ({wrong_count}): {wrong_list}"
                await self.telegram.send_message(message)

        # Check for unprotected positions (no SL)
        log.info("Checking for unprotected exchange positions...")
        unprotected = self.position_tracker.discover_unprotected_positions()
        if unprotected:
            await self._protect_untracked_positions(unprotected)

        # Run order health check on startup
        log.info("Running startup order health check...")
        await self._run_health_checks()

        # Send startup notification
        await self._send_startup_notification()

        while self.running:
            self.cycle_count += 1
            cycle_start = time.time()

            try:
                log.info(f"\n{'='*50}")
                log.info(f"Cycle #{self.cycle_count} - {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
                log.info("=" * 50)

                # 1. Update existing positions
                await self._update_positions()

                # 2. Check for closed positions
                await self._process_closed_positions()

                # 2.5. Check for signal-based exits (reverse signal detection)
                await self._check_signal_based_exits()

                # 3. Collect market data and analyze
                await self._analyze_and_decide()

                # 4. Print status summary
                await self._print_status()

                # 5. Health checks (every 5 cycles)
                if self.cycle_count % 5 == 0:
                    await self._run_health_checks()

            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)
                await self.telegram.send_message(
                    f"⚠️ Error in cycle #{self.cycle_count}: {str(e)[:100]}"
                )

            # Wait for next cycle, but check trailing positions frequently
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.interval_seconds - elapsed)

            if sleep_time > 0 and self.running:
                log.info(f"Sleeping {sleep_time:.0f}s until next cycle...")
                # Check trailing positions every 10 seconds during wait
                await self._wait_with_trailing_checks(sleep_time)

        # Cleanup on exit
        await self._shutdown()

    async def _send_startup_notification(self) -> None:
        """Send startup notification to Telegram."""
        balance = self.binance_client.get_balance()
        exposure = self.position_tracker.get_total_exposure()

        message = (
            f"🚀 Trading Bot Started\n"
            f"Mode: {self.executor.mode.upper()}\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Balance: ${balance:.2f}\n"
            f"Open positions: {len(self.position_tracker.get_open_positions())}\n"
            f"Exposure: ${exposure:.2f}"
        )
        await self.telegram.send_message(message)

    # Signal-based exit configuration
    SIGNAL_EXIT_MIN_PROFIT_PCT = 0.5  # Minimum profit to consider signal exit
    SIGNAL_EXIT_MIN_CONFIDENCE = 0.80  # Minimum confidence for reverse signal

    async def _check_signal_based_exits(self) -> None:
        """
        Check if any open positions should be closed based on reverse signals.

        Conservative strategy:
        - Position must be in profit >= 0.5%
        - Reverse signal must have confidence >= 0.80
        - Closes position to lock in profit before potential reversal
        """
        open_positions = self.position_tracker.get_open_positions()
        if not open_positions:
            return

        # Collect market data for signal generation
        try:
            market_state = await self.state_aggregator.collect_all()
            if not market_state.snapshots:
                return
        except Exception as e:
            log.warning(f"[SignalExit] Failed to collect market data: {e}")
            return

        for position in open_positions:
            symbol = position.symbol

            # Skip if not in profit threshold
            current_pnl_pct = position.unrealized_pnl_pct or 0
            if current_pnl_pct < self.SIGNAL_EXIT_MIN_PROFIT_PCT:
                continue

            # Skip if no market data for this symbol
            if symbol not in market_state.snapshots:
                continue

            try:
                # Build context for this symbol
                snapshot = market_state.snapshots[symbol]
                context = self.context_builder.analyze(snapshot)

                # Get signals from agents
                signals = await self.orchestrator.agent_manager.collect_signals(context)

                # Look for strong reverse signal
                reverse_side = "SHORT" if position.side == "LONG" else "LONG"

                reverse_signals = [
                    s for s in signals
                    if s.side == reverse_side and s.confidence >= self.SIGNAL_EXIT_MIN_CONFIDENCE
                ]

                if reverse_signals:
                    best_reverse = max(reverse_signals, key=lambda s: s.confidence)

                    log.info(
                        f"[SignalExit] {symbol}: Reverse signal detected! "
                        f"Position {position.side} +{current_pnl_pct:.1f}% → "
                        f"Signal {best_reverse.side} conf={best_reverse.confidence:.0%}"
                    )

                    # Close the position
                    current_price = self.binance_client.get_current_price(symbol)
                    success = self.position_tracker.close_position(symbol, "SIGNAL_EXIT")

                    if success:
                        # Send notification
                        message = (
                            f"🔄 Signal Exit\n"
                            f"Symbol: {symbol}\n"
                            f"Closed: {position.side} @ ${current_price:.4f}\n"
                            f"PnL: +{current_pnl_pct:.1f}%\n"
                            f"Reason: Strong {best_reverse.side} signal ({best_reverse.confidence:.0%})\n"
                            f"Agent: {best_reverse.agent_name}"
                        )
                        await self.telegram.send_message(message)

                        log.info(f"[SignalExit] {symbol} closed successfully via signal exit")

            except Exception as e:
                log.warning(f"[SignalExit] Error checking {symbol}: {e}")

    async def _update_positions(self) -> None:
        """Update PnL for all open positions."""
        self.position_tracker.update_positions()

        open_positions = self.position_tracker.get_open_positions()
        if open_positions:
            log.info("Open Positions:")
            for pos in open_positions:
                pnl_str = f"{pos.unrealized_pnl_pct:+.2f}%" if pos.unrealized_pnl_pct else "N/A"
                trailing_info = " [TRAILING]" if pos.trailing_stop_active else ""
                log.info(
                    f"  {pos.symbol}: {pos.side} @ ${pos.entry_price:.4f} | "
                    f"Current: ${pos.current_price:.4f} | PnL: {pnl_str}{trailing_info}"
                )

    async def _wait_with_trailing_checks(self, total_sleep: float) -> None:
        """
        Wait for next cycle but check trailing positions frequently.

        Args:
            total_sleep: Total seconds to wait
        """
        check_interval = 10  # Check trailing every 10 seconds
        remaining = total_sleep

        while remaining > 0 and self.running:
            # Check if any positions have active trailing stops
            trailing_positions = [
                p for p in self.position_tracker.get_open_positions()
                if p.trailing_stop_active
            ]

            if trailing_positions:
                # Update prices and check trailing stops
                self.position_tracker.update_positions()
                closed, tp1_events = self.position_tracker.check_exits()

                # Notify about TP1 hits
                for event in tp1_events:
                    trailing_status = "✅ Trailing Active" if event["trailing_active"] else "⚠️ Software Trailing Only"
                    message = (
                        f"🎯 TP1 Hit!\n"
                        f"Symbol: {event['symbol']}\n"
                        f"Price: ${event['price']:.6f}\n"
                        f"50% closed, 50% trailing\n"
                        f"{trailing_status}"
                    )
                    await self.telegram.send_message(message)

                for position in closed:
                    log.info(
                        f"[TRAILING] Position closed: {position.symbol} | "
                        f"{position.exit_reason} | PnL: {position.realized_pnl_pct:+.2f}%"
                    )
                    # Record in journal
                    self.journal.record_trade_close(position)

                    # Send notification
                    emoji = "✅" if (position.realized_pnl or 0) > 0 else "❌"
                    message = (
                        f"{emoji} Trailing Stop\n"
                        f"Symbol: {position.symbol}\n"
                        f"Side: {position.side}\n"
                        f"PnL: ${position.realized_pnl:+.2f} ({position.realized_pnl_pct:+.2f}%)"
                    )
                    await self.telegram.send_message(message)

                    # Learn from trade
                    await self._learn_from_trade(position)

            # Sleep for interval or remaining time
            sleep_now = min(check_interval, remaining)
            await asyncio.sleep(sleep_now)
            remaining -= sleep_now

    async def _process_closed_positions(self) -> None:
        """Process any positions that were closed (SL/TP triggered)."""
        closed, tp1_events = self.position_tracker.check_exits()

        # Notify about TP1 hits
        for event in tp1_events:
            trailing_status = "✅ Trailing Active" if event["trailing_active"] else "⚠️ Software Trailing Only"
            message = (
                f"🎯 TP1 Hit!\n"
                f"Symbol: {event['symbol']}\n"
                f"Price: ${event['price']:.6f}\n"
                f"50% closed, 50% trailing\n"
                f"{trailing_status}"
            )
            await self.telegram.send_message(message)

        for position in closed:
            log.info(
                f"Position closed: {position.symbol} | {position.exit_reason} | "
                f"PnL: {position.realized_pnl_pct:+.2f}%"
            )

            # Record in journal
            self.journal.record_trade_close(position)

            # If closed by SL, record cooldown to prevent immediate re-entry
            if position.exit_reason == "SL":
                self.orchestrator.rules_decision_maker.record_stop_loss(position.symbol)

            # Send Telegram notification
            emoji = "✅" if (position.realized_pnl or 0) > 0 else "❌"
            message = (
                f"{emoji} Position Closed\n"
                f"Symbol: {position.symbol}\n"
                f"Side: {position.side}\n"
                f"Reason: {position.exit_reason}\n"
                f"PnL: ${position.realized_pnl:+.2f} ({position.realized_pnl_pct:+.2f}%)\n"
                f"Duration: {self._format_duration(position.entry_time, position.exit_time)}"
            )
            await self.telegram.send_message(message)

            # Learn from trade
            await self._learn_from_trade(position)

    async def _analyze_and_decide(self) -> None:
        """Collect data, analyze, and execute decisions."""
        # Collect market data
        log.info("Collecting market data...")
        market_state = await self.state_aggregator.collect_all()

        if not market_state.snapshots:
            log.warning("No market data collected")
            return

        # Process each symbol
        for symbol in self.symbols:
            if symbol not in market_state.snapshots:
                continue

            snapshot = market_state.snapshots[symbol]

            # Build market context
            context = self.context_builder.analyze(snapshot)

            log.info(
                f"\n{symbol}: ${context.current_price:.4f} | "
                f"Regime: {context.regime.regime.value} | "
                f"Bias: {context.suggested_bias}"
            )

            # Get balance and equity for decision making
            balance = self.binance_client.get_balance()
            equity = self.binance_client.get_equity()  # For drawdown calculation
            positions = [
                {"symbol": p.symbol, "side": p.side.lower(), "contracts": p.quantity,
                 "mark_price": p.current_price or p.entry_price,
                 "entry_price": p.entry_price, "unrealized_pnl": p.unrealized_pnl or 0}
                for p in self.position_tracker.get_open_positions()
            ]

            # Get decision from orchestrator
            decision = await self.orchestrator.process_context(
                context=context,
                current_positions=positions,
                account_balance=balance,
                account_equity=equity,
            )

            self._total_decisions += 1

            # Execute decision
            await self._execute_decision(decision, context)

    async def _execute_decision(self, decision: Decision, context: MarketContext) -> None:
        """Execute a trading decision."""
        log.info(f"Decision: {decision.action} | Source: {decision.decision_source}")

        if decision.action == "WAIT":
            log.info(f"  Reason: {decision.reasoning}")
            return

        if decision.action == "CLOSE_ALL":
            log.warning("EMERGENCY: Closing all positions!")
            count = await self.executor.close_all_positions("EMERGENCY")
            await self.telegram.send_message(
                f"🚨 EMERGENCY: Closed {count} positions!"
            )
            return

        if decision.action == "REDUCE_EXPOSURE":
            log.warning(f"Reduce exposure requested: {decision.reasoning}")
            # Could implement partial closes here
            return

        if decision.action == "TRADE":
            self._trade_decisions += 1

            signal = decision.signal
            if not signal:
                return

            log.info(f"  Signal: {signal.side} {signal.symbol}")
            log.info(f"  Entry: ${signal.entry_price:.4f}")
            log.info(f"  Size: ${decision.position_size_usd:.2f}")
            log.info(f"  Confidence: {decision.confidence:.0%}")

            # Create context snapshot for learning
            context_snapshot = {
                "regime": context.regime.regime.value,
                "regime_confidence": context.regime.confidence,
                "trend_strength": context.regime.trend_strength,
                "volatility": context.regime.volatility_level,
                "sentiment": context.sentiment.overall_sentiment,
                "fear_greed": context.sentiment.fear_greed_index,
                "funding_rate": context.funding.current_rate,
                "momentum_score": context.momentum_score,
            }

            # Execute
            result = await self.executor.execute(decision, context_snapshot)

            if result['executed']:
                self._trades_executed += 1

                exec_result = result['result']
                log.info(f"  ✓ Executed @ ${exec_result['entry_price']:.4f}")

                # Get position and record in journal
                position = self.position_tracker.get_position(signal.symbol)
                if position:
                    self.journal.record_trade_open(position, signal, decision)

                # Send Telegram notification
                message = (
                    f"📈 Position Opened\n"
                    f"Symbol: {signal.symbol}\n"
                    f"Side: {signal.side}\n"
                    f"Entry: ${exec_result['entry_price']:.4f}\n"
                    f"Size: ${exec_result['position_value']:.2f}\n"
                    f"SL: ${exec_result['sl']:.4f}\n"
                    f"TP1: ${exec_result['tp1']:.4f}\n"
                    f"TP2: ${exec_result['tp2']:.4f}\n"
                    f"Agent: {signal.agent_name}\n"
                    f"Confidence: {signal.confidence:.0%}"
                )
                await self.telegram.send_message(message)

            else:
                log.info(f"  ✗ Not executed: {result['reason']}")

    async def _learn_from_trade(self, position) -> None:
        """Update learners from closed trade."""
        try:
            # Determine outcome
            outcome = 1 if (position.realized_pnl or 0) > 0 else 0

            # Record in orchestrator for agent learning
            from core.state import Signal
            # Create minimal signal for recording
            self.orchestrator.record_trade_result(
                signal=Signal(
                    id=position.signal_id,
                    timestamp=position.entry_time,
                    symbol=position.symbol,
                    side=position.side,
                    confidence=0.0,  # Not used for recording
                    entry_price=position.entry_price,
                    stop_loss=position.stop_loss,
                    take_profit_1=position.take_profit_1,
                    take_profit_2=position.take_profit_2,
                    risk_reward_ratio=0.0,
                    position_size_recommendation="normal",
                    agent_name=position.agent_name,
                    regime_at_signal=position.entry_regime,
                ),
                pnl_percent=position.realized_pnl_pct or 0,
                outcome=outcome,
            )

            # Add to experience buffer for ML learning
            if position.entry_context:
                self.orchestrator.experience_buffer.add_from_context_dict(
                    context_dict=position.entry_context,
                    symbol=position.symbol,
                    side=position.side.lower(),
                    entry_price=position.entry_price,
                    exit_price=position.exit_price or position.entry_price,
                    exit_reason=position.exit_reason or "unknown",
                    pnl_percent=position.realized_pnl_pct or 0,
                )

                # Train predictor on recent experiences
                buffer_size = len(self.orchestrator.experience_buffer)
                if buffer_size >= 5:
                    learned = self.orchestrator.predictor.learn_from_buffer(
                        n_samples=min(10, buffer_size)
                    )
                    log.info(f"[TradingLoop] Predictor trained on {learned} samples")

            log.info(
                f"[TradingLoop] Learners updated from {position.symbol} trade"
            )

        except Exception as e:
            log.warning(f"[TradingLoop] Failed to update learners: {e}")

    async def _print_status(self) -> None:
        """Print current status summary."""
        balance = self.binance_client.get_balance()
        exposure = self.position_tracker.get_total_exposure()
        exposure_pct = (exposure / balance * 100) if balance > 0 else 0
        unrealized = self.position_tracker.get_total_unrealized_pnl()

        open_positions = self.position_tracker.get_open_positions()

        log.info(f"\nStatus:")
        log.info(f"  Balance: ${balance:.2f}")
        log.info(f"  Exposure: ${exposure:.2f} ({exposure_pct:.1f}%)")
        log.info(f"  Unrealized PnL: ${unrealized:+.2f}")
        log.info(f"  Open positions: {len(open_positions)}")
        log.info(f"  Cycle: {self.cycle_count} | Decisions: {self._total_decisions} | Trades: {self._trades_executed}")

        # Today's stats
        stats = self.journal.get_performance_summary(days=1)
        if stats.get('total_trades', 0) > 0:
            log.info(
                f"  Today: {stats['wins']}W / {stats['losses']}L | "
                f"${stats['total_pnl']:+.2f}"
            )

    async def _run_health_checks(self) -> None:
        """
        Run periodic health checks:
        1. Orphan order cleanup - cancel orders for non-existent positions
        2. Order health - verify SL/Trailing orders exist
        3. Model health - check for degradation, auto-rotate if needed
        """
        log.info("[HealthCheck] Running periodic health checks...")

        try:
            # 1. Cleanup orphan and wrong-direction orders (every 10 cycles)
            if self.cycle_count % 10 == 0 or self.cycle_count == 0:
                cleanup_report = self.position_tracker.cleanup_orphan_orders()
                total_cleaned = (
                    len(cleanup_report.get('orphan_orders_cancelled', [])) +
                    len(cleanup_report.get('wrong_direction_cancelled', []))
                )
                if total_cleaned > 0:
                    log.warning(f"[HealthCheck] Cleaned up {total_cleaned} problematic orders")

            # 2. Order health check
            order_report = self.position_tracker.check_order_health()

            if order_report["issues"]:
                # Send alert for order issues (with deduplication)
                issues_text = "\n".join(
                    f"  • {item['symbol']}: {', '.join(item['issues'])}"
                    for item in order_report["issues"]
                )
                actions_text = "\n".join(f"  → {a}" for a in order_report["actions"]) if order_report["actions"] else "  None"

                # Create dedup key from symbols + issue types (not prices)
                dedup_key = "order_health:" + "|".join(
                    f"{item['symbol']}:{','.join(sorted(item['issues']))}"
                    for item in order_report["issues"]
                )

                await self.telegram.send_message(
                    f"⚠️ Order Health Issues\n"
                    f"Issues found:\n{issues_text}\n"
                    f"Actions taken:\n{actions_text}",
                    dedup_key=dedup_key
                )

            # 2. Model health check (auto-rotation)
            model_report = self.orchestrator.predictor.check_health_and_rotate(
                min_samples=50,
                accuracy_threshold=0.40,
                degradation_window=20,
            )

            if model_report["action"]:
                # Model was reset - notify
                await self.telegram.send_message(
                    f"🔄 Model Auto-Rotation\n"
                    f"{model_report['action']}\n"
                    f"Status: {model_report['status']}"
                )
                log.warning(f"[HealthCheck] Model rotated: {model_report['action']}")
            elif model_report["status"] == "degraded":
                await self.telegram.send_message(
                    f"⚠️ Model Degraded\n"
                    f"Accuracy: {model_report['overall_accuracy']:.1%}\n"
                    f"Rolling (20): {model_report['rolling_accuracy']:.1%}\n"
                    f"Consider manual intervention"
                )

            # Log summary
            log.info(
                f"[HealthCheck] Complete | "
                f"Orders: {order_report['healthy']}/{order_report['checked']} healthy | "
                f"Model: {model_report['status']} ({model_report['rolling_accuracy']:.1%} accuracy)"
            )

        except Exception as e:
            log.error(f"[HealthCheck] Failed: {e}")

    async def _protect_untracked_positions(self, unprotected: list[dict]) -> None:
        """
        Add SL orders to unprotected exchange positions.

        Args:
            unprotected: List of unprotected position info from discover_unprotected_positions
        """
        log.warning(f"[Protection] Found {len(unprotected)} unprotected positions - adding SL orders")

        for pos_info in unprotected:
            symbol = pos_info['symbol']
            side = pos_info['side']
            entry_price = pos_info['entry_price']
            amount = pos_info['amount']

            try:
                # Calculate SL at 2% from entry
                sl_pct = 0.02
                if side == 'LONG':
                    sl_price = entry_price * (1 - sl_pct)
                    exit_side = 'sell'
                else:
                    sl_price = entry_price * (1 + sl_pct)
                    exit_side = 'buy'

                # Place SL order
                sl_order = self.binance_client.place_stop_loss(
                    symbol=symbol,
                    side=exit_side,
                    quantity=amount,
                    stop_price=sl_price,
                )

                if sl_order:
                    log.info(
                        f"[Protection] {symbol}: Added SL @ ${sl_price:.6f} for {side} {amount:.4f}"
                    )
                    await self.telegram.send_message(
                        f"🛡️ Protection Added\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side}\n"
                        f"SL: ${sl_price:.6f} (-2%)"
                    )
                else:
                    log.error(f"[Protection] {symbol}: Failed to add SL order")
                    await self.telegram.send_message(
                        f"⚠️ Protection FAILED\n"
                        f"Symbol: {symbol}\n"
                        f"Could not add SL order!"
                    )

            except Exception as e:
                log.error(f"[Protection] {symbol}: Error adding SL: {e}")

    def _format_duration(self, start: datetime | None, end: datetime | None) -> str:
        """Format duration between two datetimes."""
        if not start or not end:
            return "N/A"

        duration = end - start
        minutes = duration.total_seconds() / 60

        if minutes < 60:
            return f"{minutes:.0f}m"
        elif minutes < 1440:
            return f"{minutes/60:.1f}h"
        else:
            return f"{minutes/1440:.1f}d"

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        log.info("Trading Loop shutting down...")

        # Get final stats
        stats = self.journal.get_performance_summary(days=1)

        # Send shutdown notification
        message = (
            f"🛑 Trading Bot Stopped\n"
            f"Cycles: {self.cycle_count}\n"
            f"Decisions: {self._total_decisions}\n"
            f"Trades executed: {self._trades_executed}\n"
        )

        if stats.get('total_trades', 0) > 0:
            message += (
                f"\nSession Results:\n"
                f"Trades: {stats['total_trades']}\n"
                f"Win Rate: {stats['win_rate']:.1%}\n"
                f"Total PnL: ${stats['total_pnl']:+.2f}"
            )

        await self.telegram.send_message(message)

        # Close orchestrator
        await self.orchestrator.close()

        log.info("Trading Loop stopped.")

    def stop(self) -> None:
        """Stop the trading loop."""
        log.info("Stop requested...")
        self.running = False

    def get_stats(self) -> dict[str, Any]:
        """Get loop statistics."""
        uptime = None
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            "running": self.running,
            "mode": self.executor.mode,
            "cycle_count": self.cycle_count,
            "uptime_seconds": uptime,
            "total_decisions": self._total_decisions,
            "trade_decisions": self._trade_decisions,
            "trades_executed": self._trades_executed,
            "symbols": self.symbols,
            "interval_seconds": self.interval_seconds,
        }
