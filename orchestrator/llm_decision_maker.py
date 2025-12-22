"""
LLM-based Decision Maker for AI Trading System V3.
Uses Claude as the "brain" for trading decisions.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from core.logger import get_logger
from core.state import (
    Decision,
    MarketContext,
    MarketRegime,
    Signal,
)

log = get_logger("llm_decision_maker")


# System prompt for trading decisions
TRADING_SYSTEM_PROMPT = """You are an expert crypto trading AI assistant.
Your role is to analyze market data and trading signals, then decide whether to execute trades.

You are risk-aware, disciplined, and avoid overtrading.
You always consider position sizing based on confidence and market conditions.

When making decisions, consider:
1. Signal quality (confidence, warnings, ML prediction)
2. Market regime and volatility
3. Current exposure and risk limits
4. Sentiment and funding rates
5. Recent performance

IMPORTANT: You must respond ONLY with a valid JSON object in this exact format:
{
    "action": "TRADE" | "WAIT" | "CLOSE_ALL" | "REDUCE_EXPOSURE",
    "signal_index": <index of chosen signal if TRADE, null otherwise>,
    "position_size_percent": <0.1-1.0 multiplier for recommended size>,
    "reasoning": "<brief explanation>",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks_identified": ["risk1", "risk2"],
    "confidence": <0.0-1.0>
}

Do not include any text outside the JSON object."""


class LLMDecisionMaker:
    """
    LLM-based decision maker using Claude API.

    Uses Claude to analyze signals and market context,
    making intelligent trading decisions.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize LLM decision maker.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum response tokens
            temperature: Temperature for generation
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )

        # Track usage
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0

        log.info(f"[LLMDecisionMaker] Initialized with model: {self.model}")

    async def make_decision(
        self,
        signals: list[Signal],
        context: MarketContext,
        current_positions: list[dict[str, Any]],
        account_balance: float,
        agent_stats: dict[str, Any],
    ) -> Decision | None:
        """
        Make trading decision using LLM.

        Args:
            signals: List of signals from agents
            context: Current market context
            current_positions: List of open positions
            account_balance: Current account balance
            agent_stats: Statistics for all agents

        Returns:
            Decision object or None if LLM fails
        """
        self._total_calls += 1

        # Build the prompt
        user_message = self._build_prompt(
            signals=signals,
            context=context,
            positions=current_positions,
            balance=account_balance,
            agent_stats=agent_stats,
        )

        try:
            # Call Claude API
            response = await self._client.post(
                "/messages",
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "system": TRADING_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_message}
                    ],
                },
            )

            if response.status_code != 200:
                log.error(
                    f"[LLMDecisionMaker] API error: {response.status_code} - {response.text}"
                )
                self._failed_calls += 1
                return None

            data = response.json()
            content = data.get("content", [])

            if not content:
                log.error("[LLMDecisionMaker] Empty response from API")
                self._failed_calls += 1
                return None

            # Extract text response
            text_response = content[0].get("text", "")

            # Parse the JSON response
            decision = self._parse_response(text_response, signals, context)

            if decision:
                self._successful_calls += 1
                log.info(
                    f"[LLMDecisionMaker] Decision: {decision.action}, "
                    f"confidence: {decision.confidence:.2f}"
                )

            return decision

        except httpx.TimeoutException:
            log.error("[LLMDecisionMaker] Request timed out")
            self._failed_calls += 1
            return None
        except Exception as e:
            log.error(f"[LLMDecisionMaker] Error: {e}")
            self._failed_calls += 1
            return None

    def _build_prompt(
        self,
        signals: list[Signal],
        context: MarketContext,
        positions: list[dict[str, Any]],
        balance: float,
        agent_stats: dict[str, Any],
    ) -> str:
        """Build the prompt for LLM."""
        # Format signals
        signals_text = "No signals available."
        if signals:
            signals_list = []
            for i, sig in enumerate(signals):
                signals_list.append(f"""
Signal {i}:
- Agent: {sig.agent_name}
- Side: {sig.side}
- Symbol: {sig.symbol}
- Entry: ${sig.entry_price:.2f}
- Stop Loss: ${sig.stop_loss:.2f}
- TP1: ${sig.take_profit_1:.2f}, TP2: ${sig.take_profit_2:.2f}
- R:R Ratio: {sig.risk_reward_ratio:.2f}
- Confidence: {sig.confidence:.2%}
- Size Recommendation: {sig.position_size_recommendation}
- ML Prediction: {sig.ml_prediction or 'N/A'}
- Reasoning: {'; '.join(sig.reasoning[:3])}
- Warnings: {'; '.join(sig.warnings) if sig.warnings else 'None'}""")
            signals_text = "\n".join(signals_list)

        # Format positions
        positions_text = "No open positions."
        if positions:
            pos_list = []
            for pos in positions:
                pos_list.append(
                    f"- {pos.get('symbol')}: {pos.get('side')} "
                    f"{pos.get('contracts')} @ ${pos.get('entry_price', 0):.2f}, "
                    f"PnL: ${pos.get('unrealized_pnl', 0):.2f}"
                )
            positions_text = "\n".join(pos_list)

        # Format agent performance
        agent_perf = []
        for name, stats in agent_stats.items():
            if stats.get("signals_taken", 0) > 0:
                agent_perf.append(
                    f"- {name}: {stats.get('win_rate', 0):.0%} win rate, "
                    f"{stats.get('avg_pnl_percent', 0):+.2f}% avg PnL"
                )

        agent_text = "\n".join(agent_perf) if agent_perf else "No trading history yet."

        prompt = f"""
MARKET CONTEXT:
- Symbol: {context.symbol}
- Current Price: ${context.current_price:.2f}
- Regime: {context.regime.regime.value}
- Regime Confidence: {context.regime.confidence:.2%}
- Trend Strength: {context.regime.trend_strength:.2%}
- Volatility: {context.regime.volatility_level}
- Fear & Greed Index: {context.sentiment.fear_greed_index} ({context.sentiment.fear_greed_label})
- Overall Sentiment: {context.sentiment.overall_sentiment}
- Funding Rate: {context.funding.current_rate*100:.4f}%
- Long/Short Ratio: {context.sentiment.long_short_ratio:.2f}

ACCOUNT STATUS:
- Balance: ${balance:.2f}
- Current Exposure: ${sum(abs(p.get('contracts', 0) * p.get('mark_price', 0)) for p in positions):.2f}

OPEN POSITIONS:
{positions_text}

AVAILABLE SIGNALS:
{signals_text}

AGENT PERFORMANCE:
{agent_text}

Based on this data, should we execute a trade? Consider risk management and market conditions.
Respond with a JSON decision only."""

        return prompt

    def _parse_response(
        self,
        response: str,
        signals: list[Signal],
        context: MarketContext,
    ) -> Decision | None:
        """
        Parse LLM response into Decision object.

        Args:
            response: Raw LLM response
            signals: Original signals
            context: Market context

        Returns:
            Decision object or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                log.error("[LLMDecisionMaker] No JSON found in response")
                return None

            data = json.loads(json_match.group())

            action = data.get("action", "WAIT")
            signal_index = data.get("signal_index")
            position_size_pct = data.get("position_size_percent", 1.0)
            reasoning = data.get("reasoning", "")
            key_factors = data.get("key_factors", [])
            risks = data.get("risks_identified", [])
            confidence = data.get("confidence", 0.5)

            # Get the selected signal if TRADE
            selected_signal = None
            position_size_usd = None

            if action == "TRADE" and signal_index is not None:
                if 0 <= signal_index < len(signals):
                    selected_signal = signals[signal_index]

                    # Calculate position size
                    base_sizes = {"micro": 50, "small": 100, "normal": 200, "large": 500}
                    base_size = base_sizes.get(
                        selected_signal.position_size_recommendation,
                        100
                    )
                    position_size_usd = base_size * position_size_pct
                else:
                    log.warning(
                        f"[LLMDecisionMaker] Invalid signal index: {signal_index}"
                    )
                    action = "WAIT"
                    reasoning = "Invalid signal selection"

            return Decision(
                timestamp=datetime.now(timezone.utc),
                action=action,
                signal=selected_signal,
                position_size_usd=position_size_usd,
                reasoning=reasoning,
                key_factors=key_factors,
                risks_identified=risks,
                decision_source="LLM",
                confidence=confidence,
            )

        except json.JSONDecodeError as e:
            log.error(f"[LLMDecisionMaker] JSON parse error: {e}")
            log.debug(f"[LLMDecisionMaker] Raw response: {response}")
            return None
        except Exception as e:
            log.error(f"[LLMDecisionMaker] Parse error: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get LLM decision maker statistics."""
        success_rate = 0.0
        if self._total_calls > 0:
            success_rate = self._successful_calls / self._total_calls

        return {
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "success_rate": success_rate,
            "model": self.model,
        }

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
