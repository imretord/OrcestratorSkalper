"""
State Persistence for AI Trading System V3.
Saves and restores state on restart.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logger import get_logger
from core.position_tracker import PositionTracker

log = get_logger("state_persistence")


class StatePersistence:
    """
    Persists system state between restarts.

    Saves:
    - Open positions
    - Learner statistics
    - Last cycle information
    """

    def __init__(
        self,
        state_file: str = "data/state.json",
        auto_save: bool = True,
    ) -> None:
        """
        Initialize State Persistence.

        Args:
            state_file: Path to state file
            auto_save: Whether to auto-save after each operation
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        self._state: dict[str, Any] = {}
        self._load()

        log.info(f"[StatePersistence] Initialized at {self.state_file}")

    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self._state = json.load(f)
                log.info(f"[StatePersistence] Loaded state from {self.state_file}")
            except Exception as e:
                log.warning(f"[StatePersistence] Failed to load state: {e}")
                self._state = {}
        else:
            self._state = {}

    def _save(self) -> None:
        """Save state to file."""
        try:
            self._state["saved_at"] = datetime.now(timezone.utc).isoformat()

            with open(self.state_file, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)

            log.debug(f"[StatePersistence] State saved to {self.state_file}")

        except Exception as e:
            log.error(f"[StatePersistence] Failed to save state: {e}")

    def save_positions(self, position_tracker: PositionTracker) -> None:
        """
        Save position tracker state.

        Args:
            position_tracker: Position tracker to save
        """
        self._state["positions"] = position_tracker.to_dict()

        if self.auto_save:
            self._save()

        log.info(
            f"[StatePersistence] Saved {len(position_tracker.positions)} positions"
        )

    def load_positions(self, position_tracker: PositionTracker) -> int:
        """
        Restore positions to tracker.

        Args:
            position_tracker: Position tracker to restore to

        Returns:
            Number of positions restored
        """
        positions_data = self._state.get("positions", {})

        if not positions_data:
            return 0

        position_tracker.load_from_dict(positions_data)

        restored = len(position_tracker.positions)
        log.info(f"[StatePersistence] Restored {restored} positions")

        return restored

    def save_learner_stats(
        self,
        predictor_stats: dict[str, Any],
        meta_learner_state: dict[str, Any] | None = None,
    ) -> None:
        """
        Save learner statistics.

        Args:
            predictor_stats: Online predictor stats
            meta_learner_state: Meta learner state
        """
        self._state["learners"] = {
            "predictor": predictor_stats,
            "meta_learner": meta_learner_state,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.auto_save:
            self._save()

    def get_learner_stats(self) -> dict[str, Any]:
        """Get saved learner statistics."""
        return self._state.get("learners", {})

    def save_cycle_info(
        self,
        cycle_count: int,
        last_cycle_time: datetime,
        decisions_made: int,
        trades_executed: int,
    ) -> None:
        """
        Save cycle information.

        Args:
            cycle_count: Number of cycles completed
            last_cycle_time: Time of last cycle
            decisions_made: Total decisions made
            trades_executed: Total trades executed
        """
        self._state["cycle_info"] = {
            "cycle_count": cycle_count,
            "last_cycle_time": last_cycle_time.isoformat(),
            "decisions_made": decisions_made,
            "trades_executed": trades_executed,
        }

        if self.auto_save:
            self._save()

    def get_cycle_info(self) -> dict[str, Any]:
        """Get saved cycle information."""
        return self._state.get("cycle_info", {})

    def save_full_state(
        self,
        position_tracker: PositionTracker,
        predictor_stats: dict[str, Any],
        cycle_count: int,
        decisions_made: int,
        trades_executed: int,
    ) -> None:
        """
        Save complete system state.

        Args:
            position_tracker: Position tracker
            predictor_stats: Predictor statistics
            cycle_count: Cycle count
            decisions_made: Decisions made
            trades_executed: Trades executed
        """
        self._state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "positions": position_tracker.to_dict(),
            "learners": {
                "predictor": predictor_stats,
            },
            "cycle_info": {
                "cycle_count": cycle_count,
                "last_cycle_time": datetime.now(timezone.utc).isoformat(),
                "decisions_made": decisions_made,
                "trades_executed": trades_executed,
            },
        }

        self._save()
        log.info("[StatePersistence] Full state saved")

    def get_last_saved_time(self) -> datetime | None:
        """Get time of last state save."""
        saved_at = self._state.get("saved_at")
        if saved_at:
            try:
                return datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
            except Exception:
                pass
        return None

    def clear(self) -> None:
        """Clear saved state."""
        self._state = {}
        self._save()
        log.info("[StatePersistence] State cleared")

    def get_summary(self) -> dict[str, Any]:
        """Get summary of saved state."""
        positions_data = self._state.get("positions", {})
        positions = positions_data.get("positions", {})

        return {
            "has_state": bool(self._state),
            "saved_at": self._state.get("saved_at"),
            "positions_count": len(positions),
            "has_learner_data": "learners" in self._state,
            "cycle_count": self._state.get("cycle_info", {}).get("cycle_count", 0),
        }
