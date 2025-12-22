"""
LLM prompt templates for AI Trading System V3.
"""
from __future__ import annotations


# System prompt for market analysis
MARKET_ANALYSIS_SYSTEM = """You are an expert cryptocurrency trader and market analyst.
You analyze market data objectively and provide clear, actionable insights.

Your analysis should consider:
1. Technical indicators (RSI, MACD, EMAs, Bollinger Bands)
2. Volume patterns and anomalies
3. Funding rates (positive = overleveraged longs, negative = overleveraged shorts)
4. Price action and momentum

Be concise and specific. Focus on what's actionable for trading decisions."""


# Prompt for analyzing market state
ANALYZE_MARKET_STATE = """Analyze the following market state and identify the most interesting trading opportunity:

{market_summary}

Consider:
1. Which symbol has the strongest setup?
2. What direction (long/short) is favored?
3. What are the key risk factors?

Provide a brief analysis (2-3 sentences) and your recommendation."""


# Prompt for trade decision
TRADE_DECISION_PROMPT = """Based on the current market state, should we open a position?

{market_summary}

Current portfolio:
- Balance: ${balance:.2f}
- Open positions: {open_positions}
- Daily PnL: ${daily_pnl:.2f}

Respond in JSON format:
{{
    "action": "BUY" or "SELL" or "HOLD",
    "symbol": "SYMBOL" or null,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}}"""


# Prompt for risk assessment
RISK_ASSESSMENT_PROMPT = """Assess the risk of the following trade:

Symbol: {symbol}
Direction: {side}
Entry Price: ${entry_price}
Stop Loss: ${stop_loss}
Take Profit: ${take_profit}
Position Size: ${position_size}

Market conditions:
{market_summary}

Respond in JSON format:
{{
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "risk_score": 0.0 to 1.0,
    "concerns": ["list", "of", "concerns"],
    "recommendation": "PROCEED" or "REDUCE_SIZE" or "ABORT"
}}"""


# Prompt for exit decision
EXIT_DECISION_PROMPT = """Should we close or adjust this position?

Position:
- Symbol: {symbol}
- Side: {side}
- Entry: ${entry_price}
- Current: ${current_price}
- PnL: ${pnl:.2f} ({pnl_pct:.2f}%)
- Duration: {duration_minutes:.0f} minutes

Current market state:
{market_summary}

Respond in JSON format:
{{
    "action": "HOLD" or "CLOSE" or "TIGHTEN_STOP",
    "reason": "Brief explanation",
    "new_stop_loss": price or null
}}"""
