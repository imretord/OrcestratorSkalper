#!/usr/bin/env python3
"""
AI Trading System V3 - Main Entry Point

Usage:
    Tests:
        python main.py --test-sensors     Run sensor tests
        python main.py --test-connection  Test Binance connection
        python main.py --test-llm         Test LLM connection
        python main.py --test-telegram    Test Telegram notifications
        python main.py --test-analyzers   Test Phase 2 analyzers
        python main.py --test-learners    Test Phase 3 learners
        python main.py --test-agents      Test Phase 4 agents
        python main.py --test-orchestrator Test Phase 5 orchestrator
        python main.py --test-all         Run all tests

    Trading:
        python main.py --run              Start trading loop (normal mode)
        python main.py --run --shadow     Shadow mode (log only, no trades)
        python main.py --run --micro      Micro mode (max $10 per trade)

    Utilities:
        python main.py --status           Show current status
        python main.py --close-all        Close all positions
        python main.py --stats            Show performance statistics
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.logger import setup_logger, get_logger
from core.state import MarketState, MarketRegime
from execution.binance_client import BinanceClient
from llm.client import LLMClient
from notifications.telegram import TelegramNotifier
from sensors.aggregator import StateAggregator
from analyzers.regime_detector import RegimeDetector
from analyzers.sentiment_analyzer import SentimentAnalyzer
from analyzers.market_context import MarketContextBuilder
from learners.experience_buffer import ExperienceBuffer
from learners.online_predictor import OnlinePredictor
from learners.meta_learner import MetaLearner
from agents.agent_manager import AgentManager
from orchestrator.orchestrator import Orchestrator
from core.position_tracker import PositionTracker
from core.trade_executor import TradeExecutor
from core.trading_loop import TradingLoop
from storage.trade_journal import TradeJournal
from storage.state_persistence import StatePersistence

# Initialize logger
setup_logger(log_file="logs/trading.log", level="INFO")
log = get_logger("main")


def print_header(text: str) -> None:
    """Print formatted header."""
    width = 50
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width + "\n")


def print_step(step: int, total: int, text: str) -> None:
    """Print step indicator."""
    print(f"[{step}/{total}] {text}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"✗ {text}")


async def test_binance_connection() -> tuple[bool, BinanceClient | None, float]:
    """
    Test Binance Futures connection.

    Returns:
        Tuple of (success, client, balance)
    """
    print_step(1, 6, "Connecting to Binance...")

    try:
        client = BinanceClient()
        client.connect()

        balance = client.get_balance()

        print_success(f"Connected. Balance: ${balance:.2f}")
        return True, client, balance

    except Exception as e:
        print_error(f"Binance connection failed: {e}")
        return False, None, 0.0


async def test_llm_connection() -> tuple[bool, LLMClient | None]:
    """
    Test LLM connection.

    Returns:
        Tuple of (success, client)
    """
    print_step(2, 6, "Connecting to LLM...")

    try:
        # Auto-detect provider from environment
        client = LLMClient()

        print(f"  Provider: {client.provider}, Model: {client.model}")

        success, message = client.test_connection()

        if success:
            print_success(f"Connected. {message}")
            return True, client
        else:
            print_error(f"LLM test failed: {message}")
            return False, None

    except Exception as e:
        print_error(f"LLM connection failed: {e}")
        return False, None


async def test_data_collection(
    client: BinanceClient,
    symbols: list[str]
) -> tuple[bool, MarketState | None, float]:
    """
    Test market data collection.

    Returns:
        Tuple of (success, market_state, elapsed_time)
    """
    print_step(3, 6, "Collecting market data...")

    try:
        aggregator = StateAggregator(client, symbols)

        start_time = time.time()

        # Collect for each symbol with progress
        for symbol in symbols:
            symbol_start = time.time()
            snapshot = await aggregator.collect_symbol(symbol)
            elapsed = time.time() - symbol_start

            if snapshot:
                print(f"  - {symbol}: ✓ ({elapsed:.1f}s)")
            else:
                print(f"  - {symbol}: ✗ (failed)")

        # Full collection
        market_state = await aggregator.collect_all()
        total_elapsed = time.time() - start_time

        print_success(f"All data collected in {total_elapsed:.1f}s")

        return True, market_state, total_elapsed

    except Exception as e:
        print_error(f"Data collection failed: {e}")
        return False, None, 0.0


def test_summary_generation(state: MarketState, aggregator: StateAggregator) -> tuple[bool, str]:
    """
    Test summary generation.

    Returns:
        Tuple of (success, summary)
    """
    print_step(4, 6, "Generating market summary...")

    try:
        summary = aggregator.get_summary(state)
        char_count = len(summary)

        print_success(f"Summary generated ({char_count} chars)")

        return True, summary

    except Exception as e:
        print_error(f"Summary generation failed: {e}")
        return False, ""


async def test_llm_analysis(llm: LLMClient, summary: str) -> tuple[bool, str]:
    """
    Test LLM market analysis.

    Returns:
        Tuple of (success, analysis)
    """
    print_step(5, 6, "Asking LLM for analysis...")

    try:
        start_time = time.time()

        prompt = f"""Analyze the following market state and identify the most interesting trading opportunity:

{summary}

Which symbol looks most interesting for trading right now and why? Be concise (2-3 sentences)."""

        analysis = llm.complete(
            prompt=prompt,
            system_prompt="You are an expert cryptocurrency trader. Provide brief, actionable analysis.",
        )

        elapsed = time.time() - start_time

        print_success(f"LLM response received in {elapsed:.1f}s")

        return True, analysis

    except Exception as e:
        print_error(f"LLM analysis failed: {e}")
        return False, ""


def save_market_state(state: MarketState) -> bool:
    """Save market state to JSON file."""
    try:
        output_path = Path("logs/market_state_sample.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(state.model_dump(), f, indent=2, default=str)

        log.info(f"Market state saved to {output_path}")
        return True

    except Exception as e:
        log.error(f"Failed to save market state: {e}")
        return False


async def run_sensor_tests() -> bool:
    """
    Run complete sensor test suite.

    Returns:
        True if all tests passed
    """
    print_header("AI Trading System V3 - Sensor Test")

    symbols = ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "1000PEPEUSDT"]
    all_passed = True

    # Test 1: Binance Connection
    binance_ok, client, balance = await test_binance_connection()
    if not binance_ok:
        print("\n✗ Cannot proceed without Binance connection")
        return False

    # Test 2: LLM Connection
    llm_ok, llm_client = await test_llm_connection()
    if not llm_ok:
        print("\n⚠ LLM not available - continuing without AI analysis")
        llm_client = None

    # Test 3: Data Collection
    aggregator = StateAggregator(client, symbols)
    data_ok, market_state, elapsed = await test_data_collection(client, symbols)
    if not data_ok:
        print("\n✗ Data collection failed")
        return False

    # Test 4: Summary Generation
    summary_ok, summary = test_summary_generation(market_state, aggregator)
    if not summary_ok:
        all_passed = False

    # Test 5: LLM Analysis (if available)
    if llm_client and summary:
        llm_analysis_ok, analysis = await test_llm_analysis(llm_client, summary)

        print_step(6, 6, "LLM Analysis:")
        print()
        print(f'"{analysis}"')
        print()

        if not llm_analysis_ok:
            all_passed = False
    else:
        print_step(6, 6, "LLM Analysis: Skipped (LLM not available)")

    # Save state to JSON
    save_market_state(market_state)

    # Final summary
    print_header("Test Results")

    if all_passed:
        print("All tests passed! System ready for Phase 2.")
    else:
        print("Some tests failed. Check logs for details.")

    return all_passed


async def run_connection_test() -> bool:
    """Test only Binance connection."""
    print_header("Binance Connection Test")

    success, client, balance = await test_binance_connection()

    if success:
        # Get some additional info
        print("\nFetching additional data...")

        price = client.get_current_price("DOGEUSDT")
        print(f"  DOGE price: ${price:.5f}")

        funding = client.get_funding_rate("DOGEUSDT")
        if funding:
            funding_pct = funding.current_rate * 100
            print(f"  DOGE funding: {funding_pct:.4f}%")

        positions = client.get_positions()
        print(f"  Open positions: {len(positions)}")

        print_success("All connection tests passed")

    return success


async def run_llm_test() -> bool:
    """Test LLM functionality."""
    print_header("LLM Connection Test")

    success, client = await test_llm_connection()

    if success:
        print("\nTesting JSON parsing...")

        try:
            result = client.complete_json(
                prompt="Return a JSON object with fields: symbol (string), price (number), trend (string). Use BTCUSDT, 65000, and 'up'.",
                system_prompt="You are a helpful assistant that returns valid JSON.",
            )

            if 'symbol' in result and 'price' in result:
                print_success(f"JSON parsing working: {result}")
            else:
                print_error(f"Unexpected JSON structure: {result}")
                return False

        except Exception as e:
            print_error(f"JSON parsing failed: {e}")
            return False

        print("\nTesting rate limiting (5 requests)...")

        for i in range(5):
            start = time.time()
            response = client.complete("Say 'OK'", max_tokens=10)
            elapsed = time.time() - start
            print(f"  Request {i+1}: {elapsed:.2f}s")

        stats = client.get_stats()
        print(f"\nStats: {stats['successful']} successful, avg {stats['avg_response_ms']:.0f}ms")

        print_success("All LLM tests passed")

    return success


async def run_telegram_test() -> bool:
    """Test Telegram notifications."""
    print_header("Telegram Test")

    notifier = TelegramNotifier()

    if not notifier.enabled:
        print_error("Telegram not configured (missing bot_token or chat_id)")
        return False

    print("Sending test message...")

    success = notifier.send_test_message()

    if success:
        print_success("Telegram message sent")
    else:
        print_error("Failed to send Telegram message")

    return success


async def run_analyzer_tests() -> bool:
    """Test Phase 2 analyzers."""
    print_header("AI Trading System V3 - Analyzer Tests (Phase 2)")

    symbols = ["DOGEUSDT", "XRPUSDT"]
    all_passed = True

    # Step 1: Connect to Binance
    print_step(1, 5, "Connecting to Binance...")
    try:
        client = BinanceClient()
        client.connect()
        print_success(f"Connected. Balance: ${client.get_balance():.2f}")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False

    # Step 2: Collect sensor data
    print_step(2, 5, "Collecting sensor data...")
    try:
        aggregator = StateAggregator(client, symbols)
        market_state = await aggregator.collect_all()
        print_success(f"Collected data for {len(market_state.snapshots)} symbols")
    except Exception as e:
        print_error(f"Data collection failed: {e}")
        return False

    # Step 3: Test Regime Detector
    print_step(3, 5, "Testing Regime Detector...")
    try:
        regime_detector = RegimeDetector()

        for symbol, snapshot in market_state.snapshots.items():
            regime = regime_detector.analyze(snapshot)
            print(f"  {symbol}: {regime.regime.value}")
            print(f"    - Confidence: {regime.confidence:.0%}")
            print(f"    - Trend strength: {regime.trend_strength:.0%}")
            print(f"    - Volatility: {regime.volatility_level}")

        print_success("Regime Detector working")
    except Exception as e:
        print_error(f"Regime Detector failed: {e}")
        all_passed = False

    # Step 4: Test Sentiment Analyzer
    print_step(4, 5, "Testing Sentiment Analyzer...")
    try:
        sentiment_analyzer = SentimentAnalyzer(client)

        for symbol, snapshot in market_state.snapshots.items():
            sentiment = sentiment_analyzer.analyze(snapshot)
            print(f"  {symbol}: {sentiment.overall_sentiment}")
            print(f"    - Fear & Greed: {sentiment.fear_greed_index} ({sentiment.fear_greed_label})")
            print(f"    - L/S Ratio: {sentiment.long_short_ratio:.2f}")
            print(f"    - Score: {sentiment.sentiment_score:+.2f}")

        print_success("Sentiment Analyzer working")
    except Exception as e:
        print_error(f"Sentiment Analyzer failed: {e}")
        all_passed = False

    # Step 5: Test Market Context Builder
    print_step(5, 5, "Testing Market Context Builder...")
    try:
        context_builder = MarketContextBuilder(client)

        for symbol, snapshot in market_state.snapshots.items():
            context = context_builder.analyze(snapshot)
            print(f"  {symbol} Market Context:")
            print(f"    - Suggested Bias: {context.suggested_bias.upper()}")
            print(f"    - Confidence: {context.confidence:.0%}")
            print(f"    - Risk Level: {context.risk_level}")
            print(f"    - Momentum: {context.momentum_score:+.2f}")
            print(f"    - Trend Aligned: {context.trend_aligned}")

        # Rank opportunities
        contexts = context_builder.build_for_all(market_state.snapshots)
        ranked = context_builder.rank_opportunities(contexts)

        if ranked:
            print("\n  Top Opportunities:")
            for i, (sym, ctx, score) in enumerate(ranked[:3], 1):
                print(f"    {i}. {sym}: {ctx.suggested_bias.upper()} (score: {score:.2f})")

        print_success("Market Context Builder working")
    except Exception as e:
        print_error(f"Market Context Builder failed: {e}")
        all_passed = False

    # Summary
    print_header("Analyzer Test Results")
    if all_passed:
        print("All Phase 2 analyzer tests passed!")
    else:
        print("Some analyzer tests failed. Check output above.")

    return all_passed


async def run_learner_tests() -> bool:
    """Test Phase 3 learners."""
    print_header("AI Trading System V3 - Learner Tests (Phase 3)")

    symbols = ["DOGEUSDT"]
    all_passed = True

    # Step 1: Connect to Binance
    print_step(1, 5, "Connecting to Binance...")
    try:
        client = BinanceClient()
        client.connect()
        print_success(f"Connected. Balance: ${client.get_balance():.2f}")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False

    # Step 2: Collect sensor data
    print_step(2, 5, "Collecting sensor data...")
    try:
        aggregator = StateAggregator(client, symbols)
        market_state = await aggregator.collect_all()
        print_success(f"Collected data for {len(market_state.snapshots)} symbols")
    except Exception as e:
        print_error(f"Data collection failed: {e}")
        return False

    # Build context for testing
    context_builder = MarketContextBuilder(client)
    snapshot = list(market_state.snapshots.values())[0]
    context = context_builder.analyze(snapshot)

    # Step 3: Test Experience Buffer
    print_step(3, 5, "Testing Experience Buffer...")
    try:
        buffer = ExperienceBuffer(capacity=100)

        # Add synthetic experiences for testing
        from datetime import timedelta
        now = datetime.now(timezone.utc)

        # Simulate adding experiences
        for i in range(5):
            exp = buffer.add_from_context(
                context=context,
                side="long" if i % 2 == 0 else "short",
                entry_price=context.current_price,
                exit_price=context.current_price * (1.01 if i % 3 != 0 else 0.99),
                quantity=10.0,
                exit_reason="take_profit" if i % 3 != 0 else "stop_loss",
                entry_time=now - timedelta(hours=i),
                exit_time=now - timedelta(hours=i-1),
            )
            print(f"  Added experience {exp.trade_id}: PnL={exp.pnl_percent:+.2f}%")

        stats = buffer.get_stats()
        print(f"  Buffer stats: {stats['total_trades']} trades, {stats['win_rate']:.0f}% win rate")

        # Test sampling
        samples = buffer.sample(3, method="random")
        print(f"  Random sample: {len(samples)} experiences")

        print_success("Experience Buffer working")
    except Exception as e:
        print_error(f"Experience Buffer failed: {e}")
        all_passed = False

    # Step 4: Test Online Predictor
    print_step(4, 5, "Testing Online Predictor...")
    try:
        predictor = OnlinePredictor(buffer, model_type="ensemble")

        # Make prediction (will be low confidence without training data)
        prediction = predictor.predict(context, "long")
        print(f"  Prediction: {prediction.predicted_direction}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Features extracted: {len(prediction.features)}")

        # Learn from buffer
        learned = predictor.learn_from_buffer(n_samples=5)
        print(f"  Learned from {learned} experiences")

        # Make another prediction
        prediction2 = predictor.predict(context, "long")
        print(f"  Post-learning prediction: {prediction2.predicted_direction} (conf: {prediction2.confidence:.2f})")

        stats = predictor.get_stats()
        print(f"  Predictor stats: {stats['samples_trained']} trained, {stats['predictions_made']} predictions")

        print_success("Online Predictor working")
    except Exception as e:
        print_error(f"Online Predictor failed: {e}")
        all_passed = False

    # Step 5: Test Meta Learner
    print_step(5, 5, "Testing Meta Learner...")
    try:
        meta = MetaLearner(buffer, predictor)

        # Update from experiences
        for exp in buffer:
            meta.update_from_experience(exp)
        print(f"  Updated from {len(buffer)} experiences")

        # Get combined signal
        signal = meta.get_combined_signal(context)
        print(f"  Combined signal: {signal['final_bias'].upper()}")
        print(f"  Combined score: {signal['combined_score']:+.2f}")
        print(f"  Agreement: {signal['agreement']}")

        # Get regime recommendation
        rec = meta.get_regime_recommendation(context)
        if rec.get("has_data"):
            print(f"  Regime recommendation: {rec['regime_bias']} (win rate: {rec['win_rate']:.0%})")
        else:
            print(f"  Regime recommendation: {rec['message']}")

        # Get insights
        insights = meta.get_insights()
        if insights:
            print("  Insights:")
            for insight in insights[:3]:
                print(f"    - {insight}")

        # Get learner state
        state = meta.get_learner_state()
        print(f"  Learner state: {state.experiences_count} experiences, {state.accuracy_rate:.0%} accuracy")

        print_success("Meta Learner working")
    except Exception as e:
        print_error(f"Meta Learner failed: {e}")
        all_passed = False

    # Summary
    print_header("Learner Test Results")
    if all_passed:
        print("All Phase 3 learner tests passed!")
    else:
        print("Some learner tests failed. Check output above.")

    return all_passed


async def run_agent_tests() -> bool:
    """Test Phase 4 agents."""
    print_header("AI Trading System V3 - Agent Tests (Phase 4)")

    symbols = ["DOGEUSDT"]
    all_passed = True

    # Step 1: Connect to Binance
    print_step(1, 5, "Connecting to Binance...")
    try:
        client = BinanceClient()
        client.connect()
        print_success(f"Connected. Balance: ${client.get_balance():.2f}")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False

    # Step 2: Collect sensor data and build context
    print_step(2, 5, "Collecting market data and building context...")
    try:
        aggregator = StateAggregator(client, symbols)
        market_state = await aggregator.collect_all()

        context_builder = MarketContextBuilder(client)
        snapshot = list(market_state.snapshots.values())[0]
        context = context_builder.analyze(snapshot)

        print(f"  Symbol: {context.symbol}")
        print(f"  Price: ${context.current_price:.4f}")
        print(f"  Regime: {context.regime.regime.value}")
        print_success("Context built")
    except Exception as e:
        print_error(f"Context build failed: {e}")
        return False

    # Step 3: Initialize Agent Manager
    print_step(3, 5, "Initializing Agent Manager...")
    try:
        buffer = ExperienceBuffer(capacity=100)
        predictor = OnlinePredictor(buffer, model_type="ensemble")

        agent_manager = AgentManager(
            predictor=predictor,
            meta_learner=None,
        )

        # Show regime coverage
        coverage = agent_manager.get_regime_coverage()
        print("  Agent coverage by regime:")
        for regime, agents in coverage.items():
            if agents:
                print(f"    {regime}: {', '.join(agents)}")

        print_success(f"Agent Manager initialized with {len(agent_manager.agents)} agents")
    except Exception as e:
        print_error(f"Agent Manager init failed: {e}")
        all_passed = False

    # Step 4: Collect signals from agents
    print_step(4, 5, "Collecting signals from agents...")
    try:
        signals = await agent_manager.collect_signals(context)

        print(f"  Regime: {context.regime.regime.value}")
        print(f"  Suitable agents: {[a.name for a in agent_manager.get_suitable_agents(context.regime.regime)]}")
        print(f"  Signals generated: {len(signals)}")

        if signals:
            for i, sig in enumerate(signals, 1):
                print(f"  Signal {i}:")
                print(f"    - Agent: {sig.agent_name}")
                print(f"    - Side: {sig.side}")
                print(f"    - Entry: ${sig.entry_price:.4f}")
                print(f"    - SL: ${sig.stop_loss:.4f}")
                print(f"    - TP1: ${sig.take_profit_1:.4f}")
                print(f"    - R:R: {sig.risk_reward_ratio:.2f}")
                print(f"    - Confidence: {sig.confidence:.0%}")
                print(f"    - Size: {sig.position_size_recommendation}")
                if sig.reasoning:
                    print(f"    - Reasons: {'; '.join(sig.reasoning[:2])}")
                if sig.warnings:
                    print(f"    - Warnings: {'; '.join(sig.warnings[:2])}")
        else:
            print("  No signals generated (conditions not met)")

        print_success("Signal collection working")
    except Exception as e:
        print_error(f"Signal collection failed: {e}")
        all_passed = False

    # Step 5: Test agent stats
    print_step(5, 5, "Testing agent stats...")
    try:
        stats = agent_manager.get_agent_stats()
        print("  Agent statistics:")
        for name, stat in stats.items():
            print(f"    {name}: {stat['total_signals']} signals, {stat['win_rate']:.0%} win rate")

        print_success("Agent stats working")
    except Exception as e:
        print_error(f"Agent stats failed: {e}")
        all_passed = False

    # Summary
    print_header("Agent Test Results")
    if all_passed:
        print("All Phase 4 agent tests passed!")
    else:
        print("Some agent tests failed. Check output above.")

    return all_passed


async def run_orchestrator_tests() -> bool:
    """Test Phase 5 orchestrator."""
    print_header("AI Trading System V3 - Orchestrator Tests (Phase 5)")

    symbols = ["DOGEUSDT"]
    all_passed = True

    # Step 1: Connect to Binance
    print_step(1, 6, "Connecting to Binance...")
    try:
        client = BinanceClient()
        client.connect()
        balance = client.get_balance()
        print_success(f"Connected. Balance: ${balance:.2f}")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False

    # Step 2: Collect sensor data and build context
    print_step(2, 6, "Collecting market data and building context...")
    try:
        aggregator = StateAggregator(client, symbols)
        market_state = await aggregator.collect_all()

        context_builder = MarketContextBuilder(client)
        snapshot = list(market_state.snapshots.values())[0]
        context = context_builder.analyze(snapshot)

        print(f"  Symbol: {context.symbol}")
        print(f"  Price: ${context.current_price:.4f}")
        print(f"  Regime: {context.regime.regime.value}")
        print(f"  Sentiment: {context.sentiment.overall_sentiment}")
        print_success("Context built")
    except Exception as e:
        print_error(f"Context build failed: {e}")
        return False

    # Step 3: Initialize Orchestrator (rules-based only)
    print_step(3, 6, "Initializing Orchestrator (rules-based)...")
    try:
        buffer = ExperienceBuffer(capacity=100)

        orchestrator = Orchestrator(
            experience_buffer=buffer,
            use_llm=False,  # Rules-based only for testing
            max_exposure_usd=500.0,
        )

        print(f"  LLM enabled: {orchestrator.use_llm}")
        print(f"  Max exposure: ${orchestrator.max_exposure_usd}")
        print_success("Orchestrator initialized")
    except Exception as e:
        print_error(f"Orchestrator init failed: {e}")
        return False

    # Step 4: Process context and make decision
    print_step(4, 6, "Processing context and making decision...")
    try:
        positions = client.get_positions()

        decision = await orchestrator.process_context(
            context=context,
            current_positions=positions,
            account_balance=balance,
        )

        print(f"  Decision: {decision.action}")
        print(f"  Source: {decision.decision_source}")
        print(f"  Confidence: {decision.confidence:.0%}")
        print(f"  Reasoning: {decision.reasoning}")

        if decision.signal:
            print(f"  Signal: {decision.signal.side} @ ${decision.signal.entry_price:.4f}")
            if decision.position_size_usd:
                print(f"  Position size: ${decision.position_size_usd:.2f}")

        if decision.key_factors:
            print(f"  Key factors: {'; '.join(decision.key_factors[:3])}")

        if decision.risks_identified:
            print(f"  Risks: {'; '.join(decision.risks_identified[:3])}")

        print_success("Decision making working")
    except Exception as e:
        print_error(f"Decision making failed: {e}")
        all_passed = False

    # Step 5: Test Risk Guardian
    print_step(5, 6, "Testing Risk Guardian...")
    try:
        rg_stats = orchestrator.risk_guardian.get_stats()
        print(f"  Trades blocked: {rg_stats['trades_blocked']}")
        print(f"  Trades modified: {rg_stats['trades_modified']}")
        print(f"  Daily PnL: {rg_stats['daily_pnl']:+.2f}%")
        print(f"  Max exposure: ${rg_stats['max_exposure_usd']:.2f}")

        print_success("Risk Guardian working")
    except Exception as e:
        print_error(f"Risk Guardian failed: {e}")
        all_passed = False

    # Step 6: Get full orchestrator stats
    print_step(6, 6, "Getting orchestrator stats...")
    try:
        stats = orchestrator.get_stats()
        print(f"  Decisions made: {stats['decisions_made']}")
        print(f"  LLM decisions: {stats['llm_decisions']}")
        print(f"  Rules decisions: {stats['rules_decisions']}")

        print("\n  Agent stats:")
        for name, agent_stat in stats['agent_stats'].items():
            print(f"    {name}: {agent_stat['total_signals']} signals")

        print_success("Orchestrator stats working")
    except Exception as e:
        print_error(f"Stats retrieval failed: {e}")
        all_passed = False

    # Cleanup
    await orchestrator.close()

    # Summary
    print_header("Orchestrator Test Results")
    if all_passed:
        print("All Phase 5 orchestrator tests passed!")
        print("\nSystem is ready for live trading (with caution).")
    else:
        print("Some orchestrator tests failed. Check output above.")

    return all_passed


async def run_all_tests() -> bool:
    """Run all test suites."""
    results = {
        'connection': await run_connection_test(),
        'llm': await run_llm_test(),
        'telegram': await run_telegram_test(),
        'sensors': await run_sensor_tests(),
        'analyzers': await run_analyzer_tests(),
        'learners': await run_learner_tests(),
        'agents': await run_agent_tests(),
        'orchestrator': await run_orchestrator_tests(),
    }

    print_header("All Tests Summary")

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    return all_passed


# ============================================================================
# TRADING FUNCTIONS (Phase 6)
# ============================================================================

# Load symbols from config
def _load_symbols() -> list[str]:
    """Load trading symbols from config/settings.yaml."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("trading", {}).get("symbols", [])
    except Exception:
        return ["DOGEUSDT", "XRPUSDT", "ADAUSDT", "1000PEPEUSDT"]  # Fallback


def _load_leverage() -> int:
    """Load default leverage from config/settings.yaml."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("trading", {}).get("leverage", 5)
    except Exception:
        return 5  # Default


def _load_position_size_config() -> tuple[str, float]:
    """Load position size config from config/settings.yaml."""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            trading = config.get("trading", {})
            mode = trading.get("position_size_mode", "fixed")
            pct = trading.get("position_size_pct", 2.0)
            return mode, pct
    except Exception:
        return "fixed", 2.0  # Defaults


DEFAULT_SYMBOLS = _load_symbols()
DEFAULT_LEVERAGE = _load_leverage()
POSITION_SIZE_MODE, POSITION_SIZE_PCT = _load_position_size_config()


async def run_trading(mode: str = "shadow") -> None:
    """
    Run the trading loop.

    Args:
        mode: "shadow" / "micro" / "normal"
    """
    print_header(f"AI Trading System V3 - {mode.upper()} Mode")

    # 1. Initialize Binance client
    print_step(1, 8, "Connecting to Binance...")
    try:
        client = BinanceClient()
        client.connect()
        balance = client.get_balance()
        print_success(f"Connected. Balance: ${balance:.2f}")
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return

    # 1.5. Set leverage for all symbols
    print_step(2, 9, f"Setting leverage to {DEFAULT_LEVERAGE}x for all symbols...")
    leverage_set = 0
    for symbol in DEFAULT_SYMBOLS:
        try:
            if client.set_leverage(symbol, DEFAULT_LEVERAGE):
                leverage_set += 1
        except Exception as e:
            log.warning(f"Failed to set leverage for {symbol}: {e}")
    print_success(f"Leverage set for {leverage_set}/{len(DEFAULT_SYMBOLS)} symbols")

    # 2. Initialize components
    print_step(2, 8, "Initializing components...")

    # State aggregator
    aggregator = StateAggregator(client, DEFAULT_SYMBOLS)

    # Context builder
    context_builder = MarketContextBuilder(client)

    # Experience buffer
    buffer = ExperienceBuffer(capacity=500, persist_path="data/experiences.json")

    # Online predictor
    predictor = OnlinePredictor(buffer, model_type="ensemble")

    # Train predictor on existing buffer data (if any)
    if len(buffer) > 0:
        learned = predictor.learn_from_buffer(n_samples=len(buffer))
        log.info(f"[Startup] Predictor trained on {learned} historical experiences")

    # Meta learner
    meta_learner = MetaLearner(buffer, predictor)

    # Position tracker
    position_tracker = PositionTracker(client, max_positions=15)

    # Trade executor
    executor = TradeExecutor(
        binance_client=client,
        position_tracker=position_tracker,
        mode=mode,
        micro_max_usd=10.0,
    )

    # Trade journal
    journal = TradeJournal()

    # State persistence
    state_persistence = StatePersistence()

    # Telegram notifier
    telegram = TelegramNotifier()

    print_success("Components initialized")

    # 3. Initialize Orchestrator
    print_step(3, 8, "Initializing orchestrator...")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    use_llm = bool(anthropic_key) and mode != "shadow"

    orchestrator = Orchestrator(
        experience_buffer=buffer,
        predictor=predictor,
        meta_learner=meta_learner,
        anthropic_api_key=anthropic_key if use_llm else None,
        use_llm=use_llm,
        max_exposure_usd=500.0 if mode != "micro" else 300.0,
        position_size_mode=POSITION_SIZE_MODE,
        position_size_pct=POSITION_SIZE_PCT,
        leverage=DEFAULT_LEVERAGE,
    )

    print_success(f"Orchestrator ready (LLM: {use_llm}, Position sizing: {POSITION_SIZE_MODE} {POSITION_SIZE_PCT}% × {DEFAULT_LEVERAGE}x)")

    # 4. Restore state if available
    print_step(4, 8, "Checking saved state...")

    state_summary = state_persistence.get_summary()
    if state_summary["has_state"]:
        print(f"  Found saved state from {state_summary['saved_at']}")
        print(f"  Positions: {state_summary['positions_count']}")

        restored = state_persistence.load_positions(position_tracker)
        if restored > 0:
            print_success(f"Restored {restored} positions")
    else:
        print("  No saved state found")

    # 5. Create trading loop
    print_step(5, 8, "Creating trading loop...")

    loop = TradingLoop(
        binance_client=client,
        orchestrator=orchestrator,
        executor=executor,
        position_tracker=position_tracker,
        state_aggregator=aggregator,
        context_builder=context_builder,
        journal=journal,
        telegram=telegram,
        symbols=DEFAULT_SYMBOLS,
        interval_seconds=300,  # 5 minutes
    )

    print_success("Trading loop ready")

    # 6. Setup signal handlers
    print_step(6, 8, "Setting up signal handlers...")

    import signal

    def shutdown_handler(sig, frame):
        print("\n\nShutdown requested...")
        loop.stop()

        # Save state
        state_persistence.save_full_state(
            position_tracker=position_tracker,
            predictor_stats=predictor.get_stats(),
            cycle_count=loop.cycle_count,
            decisions_made=loop._total_decisions,
            trades_executed=loop._trades_executed,
        )

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print_success("Signal handlers configured")

    # 7. Pre-flight checks
    print_step(7, 8, "Running pre-flight checks...")

    checks_passed = True

    # Check balance
    if balance < 50:
        print_error(f"Low balance: ${balance:.2f} (recommend > $100)")
        if mode == "normal":
            checks_passed = False

    # Check Telegram
    if not telegram.enabled:
        print("  ⚠ Telegram not configured")

    # Check existing positions
    positions = client.get_positions()
    if positions:
        print(f"  Found {len(positions)} existing Binance positions")

    if checks_passed:
        print_success("Pre-flight checks passed")
    else:
        print_error("Pre-flight checks failed")
        return

    # 8. Start trading
    print_step(8, 8, "Starting trading loop...")

    print("\n" + "=" * 60)
    print(f"  MODE: {mode.upper()}")
    print(f"  SYMBOLS: {', '.join(DEFAULT_SYMBOLS)}")
    print(f"  INTERVAL: 5 minutes")
    print(f"  LLM: {'Enabled' if use_llm else 'Disabled'}")
    print("=" * 60)

    if mode == "shadow":
        print("\n⚠️  SHADOW MODE: No real trades will be executed!")
        print("    All signals will be logged for analysis.\n")
    elif mode == "micro":
        print("\n⚠️  MICRO MODE: Max $10 per position!")
        print("    Real trades with minimal risk.\n")
    else:
        print("\n🔴 NORMAL MODE: Real trading with full sizing!")
        print("    Monitor actively!\n")

    try:
        await loop.run()
    finally:
        # Final state save
        state_persistence.save_full_state(
            position_tracker=position_tracker,
            predictor_stats=predictor.get_stats(),
            cycle_count=loop.cycle_count,
            decisions_made=loop._total_decisions,
            trades_executed=loop._trades_executed,
        )

        # Print final stats
        print_header("Session Summary")

        stats = journal.get_performance_summary(days=1)
        loop_stats = loop.get_stats()

        print(f"Cycles completed: {loop_stats['cycle_count']}")
        print(f"Total decisions: {loop_stats['total_decisions']}")
        print(f"Trades executed: {loop_stats['trades_executed']}")

        if stats.get('total_trades', 0) > 0:
            print(f"\nTrades: {stats['total_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1%}")
            print(f"Total PnL: ${stats['total_pnl']:+.2f}")


async def show_status() -> None:
    """Show current system status."""
    print_header("System Status")

    # 1. Binance connection
    print_step(1, 4, "Binance Status:")
    try:
        client = BinanceClient()
        client.connect()
        balance = client.get_balance()
        print(f"  Balance: ${balance:.2f}")

        positions = client.get_positions()
        if positions:
            print(f"  Open positions: {len(positions)}")
            for pos in positions:
                # Position is now a Pydantic model with attributes
                symbol = pos.symbol.replace('/USDT:USDT', 'USDT') if hasattr(pos, 'symbol') else 'Unknown'
                side = pos.side if hasattr(pos, 'side') else 'Unknown'
                pnl = pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0
                print(f"    - {symbol}: {side} | PnL: ${pnl:+.2f}")
        else:
            print("  No open positions")

    except Exception as e:
        print_error(f"Binance connection failed: {e}")

    # 2. Saved state
    print_step(2, 4, "Saved State:")
    state = StatePersistence()
    summary = state.get_summary()

    if summary["has_state"]:
        print(f"  Last saved: {summary['saved_at']}")
        print(f"  Tracked positions: {summary['positions_count']}")
        print(f"  Cycle count: {summary['cycle_count']}")
    else:
        print("  No saved state")

    # 3. Journal stats
    print_step(3, 4, "Today's Performance:")
    journal = TradeJournal()
    stats = journal.get_performance_summary(days=1)

    if stats.get('total_trades', 0) > 0:
        print(f"  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Total PnL: ${stats['total_pnl']:+.2f}")
    else:
        print("  No trades today")

    # 4. System info
    print_step(4, 4, "Configuration:")
    print(f"  Symbols: {', '.join(DEFAULT_SYMBOLS)}")
    print(f"  LLM: {'Available' if os.getenv('ANTHROPIC_API_KEY') else 'Not configured'}")
    print(f"  Telegram: {'Configured' if os.getenv('TELEGRAM_BOT_TOKEN') else 'Not configured'}")


async def close_all_positions() -> None:
    """Close all open positions."""
    print_header("Close All Positions")

    print("⚠️  This will close ALL open positions!")
    confirm = input("Are you sure? (yes/no): ")

    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    try:
        client = BinanceClient()
        client.connect()

        positions = client.get_positions()

        if not positions:
            print("No open positions to close.")
            return

        print(f"Closing {len(positions)} positions...")

        for pos in positions:
            # Position is now a Pydantic model with attributes
            symbol = pos.symbol.replace('/USDT:USDT', 'USDT') if hasattr(pos, 'symbol') else None
            side = pos.side if hasattr(pos, 'side') else 'BOTH'
            amt = pos.contracts if hasattr(pos, 'contracts') else 0

            if amt == 0:
                continue

            close_side = "sell" if amt > 0 else "buy"
            quantity = abs(amt)

            print(f"  Closing {symbol}...")

            try:
                client.cancel_all_orders(symbol)
                client.place_market_order(symbol, close_side, quantity, reduce_only=True)
                print_success(f"  {symbol} closed")
            except Exception as e:
                print_error(f"  Failed to close {symbol}: {e}")

        print_success("All positions closed")

        # Send Telegram notification
        telegram = TelegramNotifier()
        if telegram.enabled:
            telegram.send_message("🚨 All positions manually closed via CLI")

    except Exception as e:
        print_error(f"Failed: {e}")


async def show_stats() -> None:
    """Show performance statistics."""
    print_header("Performance Statistics")

    journal = TradeJournal()

    for days, label in [(1, "Today"), (7, "Last 7 Days"), (30, "Last 30 Days")]:
        stats = journal.get_performance_summary(days=days)

        print(f"\n{label}:")

        if stats.get('total_trades', 0) == 0:
            print("  No trades")
            continue

        print(f"  Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Total PnL: ${stats['total_pnl']:+.2f}")
        print(f"  Avg Win: ${stats['avg_win']:+.2f} ({stats['avg_win_pct']:+.2f}%)")
        print(f"  Avg Loss: ${stats['avg_loss']:+.2f} ({stats['avg_loss_pct']:+.2f}%)")

        if stats.get('by_agent'):
            print("\n  By Agent:")
            for agent, agent_stats in stats['by_agent'].items():
                print(
                    f"    {agent}: {agent_stats['trades']} trades, "
                    f"{agent_stats['win_rate']:.0%} win rate, "
                    f"${agent_stats['total_pnl']:+.2f}"
                )

        if stats.get('by_exit_reason'):
            print("\n  By Exit Reason:")
            for reason, count in stats['by_exit_reason'].items():
                print(f"    {reason}: {count}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Trading System V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test-sensors",
        action="store_true",
        help="Run sensor tests",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test Binance connection",
    )
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="Test LLM connection",
    )
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Test Telegram notifications",
    )
    parser.add_argument(
        "--test-analyzers",
        action="store_true",
        help="Test Phase 2 analyzers",
    )
    parser.add_argument(
        "--test-learners",
        action="store_true",
        help="Test Phase 3 learners",
    )
    parser.add_argument(
        "--test-agents",
        action="store_true",
        help="Test Phase 4 agents",
    )
    parser.add_argument(
        "--test-orchestrator",
        action="store_true",
        help="Test Phase 5 orchestrator",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run all tests",
    )

    # Trading commands (Phase 6)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start trading loop",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Shadow mode (log only, no real trades)",
    )
    parser.add_argument(
        "--micro",
        action="store_true",
        help="Micro mode (max $10 per trade)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current system status",
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="Close all open positions",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show performance statistics",
    )

    args = parser.parse_args()

    # Check if any command specified
    has_test_cmd = any([args.test_sensors, args.test_connection, args.test_llm,
                        args.test_telegram, args.test_analyzers, args.test_learners,
                        args.test_agents, args.test_orchestrator, args.test_all])
    has_trade_cmd = any([args.run, args.status, args.close_all, args.stats])

    # Default to showing help if no args
    if not has_test_cmd and not has_trade_cmd:
        parser.print_help()
        sys.exit(0)

    try:
        # Trading commands (Phase 6)
        if args.run:
            # Determine mode
            if args.shadow:
                mode = "shadow"
            elif args.micro:
                mode = "micro"
            else:
                mode = "normal"
            asyncio.run(run_trading(mode))
            sys.exit(0)

        elif args.status:
            asyncio.run(show_status())
            sys.exit(0)

        elif args.close_all:
            asyncio.run(close_all_positions())
            sys.exit(0)

        elif args.stats:
            asyncio.run(show_stats())
            sys.exit(0)

        # Test commands
        elif args.test_all:
            success = asyncio.run(run_all_tests())
        elif args.test_sensors:
            success = asyncio.run(run_sensor_tests())
        elif args.test_connection:
            success = asyncio.run(run_connection_test())
        elif args.test_llm:
            success = asyncio.run(run_llm_test())
        elif args.test_telegram:
            success = asyncio.run(run_telegram_test())
        elif args.test_analyzers:
            success = asyncio.run(run_analyzer_tests())
        elif args.test_learners:
            success = asyncio.run(run_learner_tests())
        elif args.test_agents:
            success = asyncio.run(run_agent_tests())
        elif args.test_orchestrator:
            success = asyncio.run(run_orchestrator_tests())
        else:
            success = True

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        log.exception(f"Fatal error: {e}")
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
