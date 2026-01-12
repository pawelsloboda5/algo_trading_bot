"""Trading strategies and technical indicators."""

from src.strategy.base_strategy import BaseStrategy, Signal, TradeSignal
from src.strategy.indicators import (
    atr,
    bollinger_bands,
    crossover,
    crossunder,
    ema,
    macd,
    rate_of_change,
    rsi,
    sma,
)
from src.strategy.momentum_strategy import MACDMomentumStrategy, MomentumStrategy
from src.strategy.scalping_indicators import (
    fast_trend,
    momentum_strength,
    volatility_percentile,
    volume_acceleration,
)
from src.strategy.scalping_strategy import ScalpingConfig, ScalpingStrategy
from src.strategy.mean_reversion_strategy import (
    MeanReversionConfig,
    MeanReversionStrategy,
    create_conservative_config,
    create_aggressive_config,
    create_optimized_config,
)
from src.strategy.session_filter import SessionFilter, TradingSession

__all__ = [
    # Base
    "BaseStrategy",
    "Signal",
    "TradeSignal",
    # Strategies
    "MomentumStrategy",
    "MACDMomentumStrategy",
    "ScalpingStrategy",
    "ScalpingConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "create_conservative_config",
    "create_aggressive_config",
    # Session filter
    "SessionFilter",
    "TradingSession",
    # Indicators
    "ema",
    "sma",
    "atr",
    "macd",
    "rsi",
    "bollinger_bands",
    "crossover",
    "crossunder",
    "rate_of_change",
    # Scalping indicators
    "volume_acceleration",
    "volatility_percentile",
    "momentum_strength",
    "fast_trend",
]
