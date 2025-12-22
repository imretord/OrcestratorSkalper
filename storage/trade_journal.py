"""
Trade Journal for AI Trading System V3.
Detailed trade logging for analysis and learning.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from core.logger import get_logger
from core.state import Decision, Signal, TrackedPosition

log = get_logger("trade_journal")


class TradeJournal:
    """
    Comprehensive trade journal for analysis.

    Records:
    - Trade opens and closes
    - Context at entry
    - Performance metrics
    - Agent performance
    """

    def __init__(
        self,
        journal_dir: str = "data/journal",
    ) -> None:
        """
        Initialize Trade Journal.

        Args:
            journal_dir: Directory for journal files
        """
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        # Current day's journal file
        self._current_date = datetime.now(timezone.utc).date()
        self._entries: list[dict[str, Any]] = []

        log.info(f"[TradeJournal] Initialized at {self.journal_dir}")

    def _get_journal_file(self, date: datetime | None = None) -> Path:
        """Get journal file for a specific date."""
        if date is None:
            date = datetime.now(timezone.utc)
        return self.journal_dir / f"trades_{date.strftime('%Y%m%d')}.json"

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single entry to journal."""
        # Check if we need to rotate to new day
        current_date = datetime.now(timezone.utc).date()
        if current_date != self._current_date:
            self._flush_entries()
            self._current_date = current_date
            self._entries = []

        self._entries.append(entry)

        # Write to file immediately
        self._flush_entries()

    def _flush_entries(self) -> None:
        """Flush entries to disk."""
        if not self._entries:
            return

        file_path = self._get_journal_file()

        # Load existing entries
        existing = []
        if file_path.exists():
            try:
                with open(file_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Merge and save
        all_entries = existing + self._entries

        with open(file_path, 'w') as f:
            json.dump(all_entries, f, indent=2, default=str)

        self._entries = []

    def record_trade_open(
        self,
        position: TrackedPosition,
        signal: Signal,
        decision: Decision,
    ) -> None:
        """
        Record trade opening.

        Args:
            position: Opened position
            signal: Signal that triggered the trade
            decision: Decision from orchestrator
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "OPEN",
            "position_id": position.id,
            "signal_id": signal.id,

            # Position details
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "quantity": position.quantity,
            "position_value_usd": position.position_value_usd,

            # Levels
            "stop_loss": position.stop_loss,
            "take_profit_1": position.take_profit_1,
            "take_profit_2": position.take_profit_2,

            # Signal details
            "agent": signal.agent_name,
            "signal_confidence": signal.confidence,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "ml_prediction": signal.ml_prediction,
            "ml_confidence": signal.ml_confidence,

            # Decision details
            "decision_source": decision.decision_source,
            "decision_confidence": decision.confidence,

            # Context
            "regime": position.entry_regime.value,
            "reasoning": signal.reasoning[:5] if signal.reasoning else [],
            "warnings": signal.warnings[:3] if signal.warnings else [],
        }

        self._write_entry(entry)

        log.info(
            f"[TradeJournal] Recorded OPEN: {position.side} {position.symbol} "
            f"@ ${position.entry_price:.4f}"
        )

    def record_trade_close(self, position: TrackedPosition) -> None:
        """
        Record trade closing.

        Args:
            position: Closed position
        """
        duration_minutes = 0
        if position.exit_time and position.entry_time:
            duration_minutes = (position.exit_time - position.entry_time).total_seconds() / 60

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "CLOSE",
            "position_id": position.id,
            "signal_id": position.signal_id,

            # Position details
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": position.exit_price,
            "quantity": position.quantity,

            # Result
            "exit_reason": position.exit_reason,
            "duration_minutes": round(duration_minutes, 1),
            "realized_pnl": position.realized_pnl,
            "realized_pnl_pct": position.realized_pnl_pct,
            "max_profit_pct": position.max_profit_pct,

            # Context
            "regime": position.entry_regime.value,
            "tp1_hit": position.tp1_hit,
        }

        self._write_entry(entry)

        emoji = "+" if (position.realized_pnl or 0) > 0 else "-"
        log.info(
            f"[TradeJournal] Recorded CLOSE: {position.symbol} | "
            f"{position.exit_reason} | PnL: {emoji}${abs(position.realized_pnl or 0):.2f}"
        )

    def get_trades(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Load trades from the last N days.

        Args:
            days: Number of days to load

        Returns:
            List of trade entries
        """
        all_trades: list[dict[str, Any]] = []

        for i in range(days):
            date = datetime.now(timezone.utc) - timedelta(days=i)
            file_path = self._get_journal_file(date)

            if file_path.exists():
                try:
                    with open(file_path) as f:
                        trades = json.load(f)
                        all_trades.extend(trades)
                except Exception as e:
                    log.warning(f"[TradeJournal] Failed to load {file_path}: {e}")

        return all_trades

    def get_performance_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Get performance summary for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Performance statistics
        """
        trades = self.get_trades(days)
        closed = [t for t in trades if t.get('event') == 'CLOSE']

        if not closed:
            return {
                "trades": 0,
                "period_days": days,
            }

        wins = [t for t in closed if (t.get('realized_pnl_pct') or 0) > 0]
        losses = [t for t in closed if (t.get('realized_pnl_pct') or 0) <= 0]

        return {
            "period_days": days,
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(t.get('realized_pnl', 0) or 0 for t in closed),
            "total_pnl_pct": sum(t.get('realized_pnl_pct', 0) or 0 for t in closed),
            "avg_win": (
                sum(t.get('realized_pnl', 0) or 0 for t in wins) / len(wins)
                if wins else 0
            ),
            "avg_loss": (
                sum(t.get('realized_pnl', 0) or 0 for t in losses) / len(losses)
                if losses else 0
            ),
            "avg_win_pct": (
                sum(t.get('realized_pnl_pct', 0) or 0 for t in wins) / len(wins)
                if wins else 0
            ),
            "avg_loss_pct": (
                sum(t.get('realized_pnl_pct', 0) or 0 for t in losses) / len(losses)
                if losses else 0
            ),
            "best_trade": max(t.get('realized_pnl', 0) or 0 for t in closed),
            "worst_trade": min(t.get('realized_pnl', 0) or 0 for t in closed),
            "avg_duration_minutes": (
                sum(t.get('duration_minutes', 0) or 0 for t in closed) / len(closed)
                if closed else 0
            ),
            "by_agent": self._group_by_agent(closed),
            "by_regime": self._group_by_regime(closed),
            "by_exit_reason": self._group_by_exit_reason(closed),
        }

    def _group_by_agent(self, trades: list[dict]) -> dict[str, dict]:
        """Group performance by agent."""
        # Get opens to find agent info
        opens = {t.get('position_id'): t for t in self.get_trades(30) if t.get('event') == 'OPEN'}

        by_agent: dict[str, list] = {}
        for trade in trades:
            pos_id = trade.get('position_id')
            open_trade = opens.get(pos_id, {})
            agent = open_trade.get('agent', 'Unknown')

            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(trade)

        result = {}
        for agent, agent_trades in by_agent.items():
            wins = [t for t in agent_trades if (t.get('realized_pnl_pct') or 0) > 0]
            result[agent] = {
                "trades": len(agent_trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(agent_trades) if agent_trades else 0,
                "total_pnl": sum(t.get('realized_pnl', 0) or 0 for t in agent_trades),
            }

        return result

    def _group_by_regime(self, trades: list[dict]) -> dict[str, dict]:
        """Group performance by market regime."""
        by_regime: dict[str, list] = {}

        for trade in trades:
            regime = trade.get('regime', 'unknown')
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(trade)

        result = {}
        for regime, regime_trades in by_regime.items():
            wins = [t for t in regime_trades if (t.get('realized_pnl_pct') or 0) > 0]
            result[regime] = {
                "trades": len(regime_trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(regime_trades) if regime_trades else 0,
                "total_pnl": sum(t.get('realized_pnl', 0) or 0 for t in regime_trades),
            }

        return result

    def _group_by_exit_reason(self, trades: list[dict]) -> dict[str, int]:
        """Count trades by exit reason."""
        by_reason: dict[str, int] = {}

        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            by_reason[reason] = by_reason.get(reason, 0) + 1

        return by_reason

    def export_to_csv(self, filepath: str, days: int = 30) -> bool:
        """
        Export trades to CSV for external analysis.

        Args:
            filepath: Output CSV file path
            days: Number of days to export

        Returns:
            True if successful
        """
        try:
            trades = self.get_trades(days)
            if not trades:
                log.warning("[TradeJournal] No trades to export")
                return False

            # Flatten and export
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)

            log.info(f"[TradeJournal] Exported {len(trades)} trades to {filepath}")
            return True

        except Exception as e:
            log.error(f"[TradeJournal] Export failed: {e}")
            return False
