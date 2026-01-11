"""Backtesting framework."""

from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, run_backtest
from src.backtest.metrics import PerformanceMetrics, calculate_metrics
from src.backtest.portfolio import Portfolio, Position, Trade, TradeDirection

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    # Portfolio
    "Portfolio",
    "Position",
    "Trade",
    "TradeDirection",
    # Metrics
    "PerformanceMetrics",
    "calculate_metrics",
]
