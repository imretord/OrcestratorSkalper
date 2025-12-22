"""
Trade Logger for AI Trading System V3.
Logs trades to JSON files for analysis.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import get_logger

log = get_logger("trade_logger")


class TradeLogger:
    """
    JSON-based trade logging.

    Logs each trade to a daily JSON file for easy analysis and backup.
    """

    def __init__(self, log_dir: str = "logs/trades") -> None:
        """
        Initialize TradeLogger.

        Args:
            log_dir: Directory for trade log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """Get log file path for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"trades_{today}.json"

    def _load_trades(self) -> list[dict[str, Any]]:
        """Load trades from today's file."""
        log_file = self._get_log_file()
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return []

    def _save_trades(self, trades: list[dict[str, Any]]) -> None:
        """Save trades to today's file."""
        log_file = self._get_log_file()
        with open(log_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)

    def log_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int = 1,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reason: str = "",
    ) -> str:
        """
        Log trade entry.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            quantity: Position size
            leverage: Leverage used
            stop_loss: Stop loss price
            take_profit: Take profit price
            reason: Entry reason

        Returns:
            Trade ID
        """
        trades = self._load_trades()

        trade_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        trade = {
            'id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_reason': reason,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'status': 'open',
        }

        trades.append(trade)
        self._save_trades(trades)

        log.info(f"[TradeLogger] Entry: {trade_id}")
        return trade_id

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str = "manual",
    ) -> None:
        """
        Log trade exit.

        Args:
            trade_id: Trade ID from log_entry
            exit_price: Exit price
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            reason: Exit reason
        """
        trades = self._load_trades()

        for trade in trades:
            if trade['id'] == trade_id:
                trade['exit_price'] = exit_price
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                trade['exit_reason'] = reason
                trade['exit_time'] = datetime.now(timezone.utc).isoformat()
                trade['status'] = 'closed'
                break

        self._save_trades(trades)
        log.info(f"[TradeLogger] Exit: {trade_id}, PnL=${pnl:.2f}")

    def get_daily_stats(self) -> dict[str, Any]:
        """
        Get statistics for today's trades.

        Returns:
            Dict with stats
        """
        trades = self._load_trades()
        closed = [t for t in trades if t.get('status') == 'closed']

        if not closed:
            return {
                'total': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
            }

        wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in closed)

        return {
            'total': len(closed),
            'wins': wins,
            'losses': len(closed) - wins,
            'win_rate': (wins / len(closed)) * 100,
            'total_pnl': total_pnl,
        }
