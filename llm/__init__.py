"""
LLM module for AI Trading System V3.
"""
from llm.client import LLMClient
from llm.prompts import (
    ANALYZE_MARKET_STATE,
    EXIT_DECISION_PROMPT,
    MARKET_ANALYSIS_SYSTEM,
    RISK_ASSESSMENT_PROMPT,
    TRADE_DECISION_PROMPT,
)

__all__ = [
    "LLMClient",
    "ANALYZE_MARKET_STATE",
    "EXIT_DECISION_PROMPT",
    "MARKET_ANALYSIS_SYSTEM",
    "RISK_ASSESSMENT_PROMPT",
    "TRADE_DECISION_PROMPT",
]
