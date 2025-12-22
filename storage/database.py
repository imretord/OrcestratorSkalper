"""
Database storage for AI Trading System V3.
SQLite-based storage for trades and state.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from core.logger import get_logger

log = get_logger("database")

Base = declarative_base()


class TradeRecord(Base):
    """Trade record model."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    status = Column(String(20), default="open")
    entry_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(50), nullable=True)
    metadata_json = Column(Text, nullable=True)


class StateSnapshot(Base):
    """Market state snapshot model."""
    __tablename__ = "state_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    state_json = Column(Text, nullable=False)


class Database:
    """
    SQLite database manager.

    Features:
    - Trade logging
    - State snapshot storage
    - Query historical data
    """

    def __init__(self, db_path: str = "storage/trading.db") -> None:
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        log.info(f"[Database] Initialized: {db_path}")

    def log_trade_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Log trade entry.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            quantity: Position size
            metadata: Additional metadata

        Returns:
            Trade record ID
        """
        with Session(self.engine) as session:
            trade = TradeRecord(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                status="open",
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            session.add(trade)
            session.commit()

            log.info(f"[Database] Trade entry logged: {symbol} {side} @ ${entry_price}")
            return trade.id

    def log_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        """
        Log trade exit.

        Args:
            trade_id: Trade record ID
            exit_price: Exit price
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            exit_reason: Reason for exit
        """
        with Session(self.engine) as session:
            trade = session.query(TradeRecord).filter_by(id=trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.pnl_pct = pnl_pct
                trade.status = "closed"
                trade.exit_time = datetime.now(timezone.utc)
                trade.exit_reason = exit_reason
                session.commit()

                log.info(f"[Database] Trade exit logged: ID={trade_id}, PnL=${pnl:.2f}")

    def save_state_snapshot(self, state_json: str) -> None:
        """
        Save market state snapshot.

        Args:
            state_json: JSON serialized state
        """
        with Session(self.engine) as session:
            snapshot = StateSnapshot(state_json=state_json)
            session.add(snapshot)
            session.commit()

    def get_recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get recent trades.

        Args:
            limit: Maximum number of trades

        Returns:
            List of trade dicts
        """
        with Session(self.engine) as session:
            trades = (
                session.query(TradeRecord)
                .order_by(TradeRecord.entry_time.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    'id': t.id,
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'status': t.status,
                    'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'exit_reason': t.exit_reason,
                }
                for t in trades
            ]

    def get_trade_stats(self) -> dict[str, Any]:
        """
        Calculate trade statistics.

        Returns:
            Dict with stats
        """
        with Session(self.engine) as session:
            closed_trades = session.query(TradeRecord).filter_by(status="closed").all()

            if not closed_trades:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                }

            wins = sum(1 for t in closed_trades if t.pnl and t.pnl > 0)
            losses = sum(1 for t in closed_trades if t.pnl and t.pnl <= 0)
            total_pnl = sum(t.pnl or 0 for t in closed_trades)

            return {
                'total_trades': len(closed_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / len(closed_trades)) * 100 if closed_trades else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            }
