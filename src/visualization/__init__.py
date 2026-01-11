"""Visualization module for backtest results.

This module provides interactive charts and dashboards for analyzing
backtest performance, as well as structured result saving for AI-driven
optimization.

Components:
    - ChartGenerator: Creates interactive Plotly charts
    - ResultSaver: Saves structured results to files
    - BacktestViewer: Interactive Dash dashboard
"""

from src.visualization.charts import ChartGenerator
from src.visualization.report import ResultSaver
from src.visualization.window import BacktestViewer, show_backtest_results

__all__ = [
    "ChartGenerator",
    "ResultSaver",
    "BacktestViewer",
    "show_backtest_results",
]
