"""
Core data structures for AI Trading System V3.
All state objects use Pydantic for validation and serialization.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
# ============================================================================

class MarketRegime(str, Enum):
    """Market regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    CHOPPY = "choppy"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    COMPRESSION = "compression"


class OHLCVBar(BaseModel):
    """Single OHLCV candlestick bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IndicatorValues(BaseModel):
    """Technical indicator values for a single timeframe."""
    rsi_14: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    atr_14: float | None = None
    adx_14: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None


class SymbolData(BaseModel):
    """Basic symbol information."""
    symbol: str
    current_price: float
    price_change_24h_pct: float
    volume_24h: float
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PriceFeedData(BaseModel):
    """Complete price feed data for a symbol including OHLCV and indicators."""
    symbol: str
    current_price: float
    ohlcv_1m: list[OHLCVBar] = Field(default_factory=list)
    ohlcv_5m: list[OHLCVBar] = Field(default_factory=list)
    ohlcv_15m: list[OHLCVBar] = Field(default_factory=list)
    ohlcv_1h: list[OHLCVBar] = Field(default_factory=list)
    ohlcv_4h: list[OHLCVBar] = Field(default_factory=list)
    indicators: dict[str, IndicatorValues] = Field(default_factory=dict)  # timeframe -> indicators
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class VolumeData(BaseModel):
    """Volume analysis data for a symbol."""
    symbol: str
    current_volume: float
    volume_sma_20: float
    relative_volume: float  # current / sma20
    volume_spike: bool  # relative > 2.0
    buy_volume_ratio: float  # 0-1, buy pressure estimate
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class FundingData(BaseModel):
    """Funding rate data for a futures symbol."""
    symbol: str
    current_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    predicted_rate: float  # Predicted next funding rate
    next_funding_time: datetime
    rate_trend: str  # "rising" / "falling" / "stable"
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SensorSnapshot(BaseModel):
    """Complete sensor data snapshot for a single symbol."""
    symbol: str
    price_feed: PriceFeedData
    volume: VolumeData
    funding: FundingData
    collected_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MarketState(BaseModel):
    """Complete market state containing all symbol snapshots."""
    snapshots: dict[str, SensorSnapshot] = Field(default_factory=dict)  # symbol -> snapshot
    global_metrics: dict[str, Any] = Field(default_factory=dict)  # BTC dominance, etc.
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Position(BaseModel):
    """Open trading position."""
    symbol: str
    side: str  # "long" or "short"
    contracts: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str  # "isolated" or "cross"
    liquidation_price: float

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Order(BaseModel):
    """Trading order."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop_market", etc.
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: str  # "open", "filled", "cancelled"
    filled_quantity: float = 0.0
    average_price: float | None = None
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# ANALYZER DATA STRUCTURES (Phase 2)
# ============================================================================

class RegimeAnalysis(BaseModel):
    """Market regime analysis result."""
    symbol: str
    regime: MarketRegime
    confidence: float  # 0-1
    trend_strength: float  # 0-1
    volatility_level: str  # "low" / "medium" / "high" / "extreme"
    regime_duration_hours: int
    support_level: float | None = None
    resistance_level: float | None = None
    regime_description: str
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SentimentAnalysis(BaseModel):
    """Market sentiment analysis result."""
    symbol: str
    fear_greed_index: int  # 0-100
    fear_greed_label: str  # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    long_short_ratio: float  # > 1 means more longs
    open_interest_change_24h: float  # percentage
    funding_sentiment: str  # "bullish" / "bearish" / "neutral"
    social_sentiment: str  # "bullish" / "bearish" / "neutral"
    overall_sentiment: str  # "bullish" / "bearish" / "neutral"
    sentiment_score: float  # -1 to +1
    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MarketContext(BaseModel):
    """Complete market context combining all analyses."""
    symbol: str
    current_price: float
    regime: RegimeAnalysis
    sentiment: SentimentAnalysis
    price_feed: PriceFeedData
    volume: VolumeData
    funding: FundingData

    # Derived signals
    trend_aligned: bool  # Multi-timeframe trend agreement
    momentum_score: float  # -1 to +1
    volatility_adjusted_score: float  # -1 to +1

    # Trading recommendations
    suggested_bias: str  # "long" / "short" / "neutral"
    confidence: float  # 0-1
    risk_level: str  # "low" / "medium" / "high"

    timestamp: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# LEARNER DATA STRUCTURES (Phase 3)
# ============================================================================

class TradeExperience(BaseModel):
    """Single trade experience for learning."""
    trade_id: str
    symbol: str
    side: str  # "long" / "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_percent: float
    duration_seconds: int

    # Context at entry
    regime_at_entry: MarketRegime
    sentiment_at_entry: float  # -1 to +1
    volatility_at_entry: str
    rsi_at_entry: float
    macd_histogram_at_entry: float
    volume_relative_at_entry: float
    funding_rate_at_entry: float

    # Context at exit
    exit_reason: str  # "take_profit" / "stop_loss" / "trailing_stop" / "signal_exit" / "timeout"

    # Outcome
    success: bool  # pnl > 0
    reward: float  # normalized reward for learning

    entry_time: datetime
    exit_time: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PredictionRecord(BaseModel):
    """Record of a prediction made by the online learner."""
    prediction_id: str
    symbol: str
    predicted_direction: str  # "up" / "down" / "neutral"
    predicted_magnitude: float  # expected % move
    confidence: float  # 0-1

    # Features used
    features: dict[str, float]

    # Actual outcome (filled after resolution)
    actual_direction: str | None = None
    actual_magnitude: float | None = None
    was_correct: bool | None = None

    prediction_time: datetime
    resolution_time: datetime | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class FeatureImportance(BaseModel):
    """Feature importance from meta-learner."""
    feature_name: str
    importance_score: float  # 0-1
    trend_24h: str  # "increasing" / "decreasing" / "stable"
    regime_specific: dict[str, float] = Field(default_factory=dict)  # regime -> importance

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class LearnerState(BaseModel):
    """Complete state of the learning system."""
    experiences_count: int
    predictions_made: int
    predictions_correct: int
    accuracy_rate: float  # 0-1

    # Performance by regime
    accuracy_by_regime: dict[str, float] = Field(default_factory=dict)

    # Feature importances
    feature_importances: list[FeatureImportance] = Field(default_factory=list)

    # Recent performance
    recent_accuracy_10: float  # last 10 predictions
    recent_accuracy_50: float  # last 50 predictions

    last_update: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# AGENT DATA STRUCTURES (Phase 4)
# ============================================================================

class Signal(BaseModel):
    """Trading signal generated by an agent."""
    id: str  # UUID
    timestamp: datetime

    # Core signal
    symbol: str
    side: str  # "LONG" / "SHORT"
    confidence: float  # 0-1

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_1: float  # TP1 - 50% position
    take_profit_2: float  # TP2 - remaining 50%

    # Risk metrics
    risk_reward_ratio: float
    position_size_recommendation: str  # "micro" / "small" / "normal" / "large"

    # Context
    agent_name: str
    regime_at_signal: MarketRegime
    reasoning: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # ML prediction
    ml_prediction: str | None = None  # "FAVORABLE" / "UNFAVORABLE" / "NEUTRAL"
    ml_confidence: float | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AgentState(BaseModel):
    """State of a trading agent."""
    name: str
    status: str = "ACTIVE"  # "ACTIVE" / "STANDBY" / "DISABLED"
    suitable_regimes: list[MarketRegime] = Field(default_factory=list)

    # Statistics
    total_signals: int = 0
    signals_taken: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_pnl_percent: float = 0.0
    total_pnl_percent: float = 0.0

    # Last activity
    last_signal_time: datetime | None = None
    last_trade_result: float | None = None

    # Adaptive parameters
    current_params: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# ORCHESTRATOR DATA STRUCTURES (Phase 5)
# ============================================================================

class Decision(BaseModel):
    """Trading decision from orchestrator."""
    timestamp: datetime

    # Main decision
    action: str  # "TRADE" / "WAIT" / "CLOSE_ALL" / "REDUCE_EXPOSURE"

    # If TRADE
    signal: Signal | None = None
    position_size_usd: float | None = None

    # Reasoning
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    risks_identified: list[str] = Field(default_factory=list)

    # Decision source
    decision_source: str = "RULES"  # "LLM" / "RULES" / "EMERGENCY"
    confidence: float = 0.0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# POSITION TRACKING (Phase 6)
# ============================================================================

class TrackedPosition(BaseModel):
    """Tracked trading position with full lifecycle data."""
    id: str  # UUID
    signal_id: str  # Link to Signal that opened this position
    agent_name: str = ""  # Agent that generated the signal

    # Core position data
    symbol: str
    side: str  # "LONG" / "SHORT"

    # Entry data
    entry_time: datetime
    entry_price: float
    quantity: float
    position_value_usd: float

    # Price levels
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Binance order IDs
    entry_order_id: str
    sl_order_id: str | None = None
    tp1_order_id: str | None = None
    tp2_order_id: str | None = None

    # Status
    status: str = "OPEN"  # "OPEN" / "PARTIALLY_CLOSED" / "CLOSED"
    tp1_hit: bool = False
    tp1_actual_price: float | None = None  # Actual TP1 execution price
    tp1_realized_pnl: float | None = None  # PnL from TP1 (50% of position)

    # Real-time tracking
    current_price: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    max_profit_pct: float = 0.0  # For trailing stop

    # Trailing stop (activated after TP1)
    trailing_stop_active: bool = False
    trailing_stop_callback: float = 0.01  # 1% callback
    trailing_peak_price: float | None = None  # Best price since TP1
    trailing_stop_order_id: str | None = None  # Binance trailing stop order ID

    # Exit data (when closed)
    exit_time: datetime | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None
    realized_pnl_pct: float | None = None
    exit_reason: str | None = None  # "SL" / "TP1" / "TP2" / "MANUAL" / "EMERGENCY"

    # Context at entry (for learning)
    entry_regime: MarketRegime
    entry_context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
