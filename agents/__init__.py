"""
Agents module for AI Trading System V3.
Contains specialized trading agents for different market regimes.
"""
from agents.base_agent import BaseAgent
from agents.trend_follower import TrendFollowerAgent
from agents.mean_reversion import MeanReversionAgent
from agents.breakout_catcher import BreakoutCatcherAgent
from agents.agent_manager import AgentManager

__all__ = [
    "BaseAgent",
    "TrendFollowerAgent",
    "MeanReversionAgent",
    "BreakoutCatcherAgent",
    "AgentManager",
]
