# Visualization Module

## Purpose

This module provides comprehensive backtest visualization and result saving capabilities for Phase 5 of the algo trading bot. It enables both human analysis through interactive dashboards and AI-driven optimization through structured data files.

## Components

### ChartGenerator (`charts.py`)

Creates interactive Plotly charts for backtest analysis:

- `create_equity_curve()` - Equity curve with trade entry/exit markers
- `create_drawdown_chart()` - Underwater chart showing drawdown periods
- `create_trade_distribution()` - Histogram of trade P&L
- `create_monthly_returns_heatmap()` - Calendar heatmap of monthly returns
- `create_rolling_sharpe()` - Rolling Sharpe ratio over time
- `create_trade_analysis()` - Performance by hour, day of week, duration
- `create_cumulative_pnl()` - Cumulative P&L from trades
- `create_streak_analysis()` - Win/loss streak distribution
- `create_overview_figure()` - Combined equity curve and drawdown

### ResultSaver (`report.py`)

Saves structured backtest results for analysis:

**Output Files:**
- `summary.json` - AI-friendly metrics and recommendations
- `trades.csv` - Complete trade history
- `equity_curve.csv` - Timestamped equity values
- `daily_pnl.csv` - Daily P&L breakdown
- `parameters.json` - Strategy and config parameters
- `report.html` - Human-readable HTML report
- `charts/` - Individual chart HTML files

**Key Methods:**
- `save()` - Save all results to a run directory
- `_generate_recommendations()` - AI-powered optimization suggestions

### BacktestViewer (`window.py`)

Interactive Dash dashboard with four tabs:

1. **Overview Tab** - Key metrics, equity curve, drawdown
2. **Trades Tab** - Trade table, P&L distribution, cumulative P&L
3. **Analysis Tab** - Time analysis, monthly returns, rolling Sharpe, streaks
4. **Parameters Tab** - Strategy params, risk settings, recommendations

## Usage

### Basic Usage

```python
from src.visualization import ChartGenerator, ResultSaver, show_backtest_results

# After running a backtest
result = engine.run(data, strategy)

# Save results to files
saver = ResultSaver("results/backtests")
run_dir = saver.save(result, symbol="MCL_FUT")

# Show interactive dashboard
show_backtest_results(result, symbol="MCL_FUT")
```

### CLI Usage

```bash
# Run backtest and open visualization
python scripts/run_backtest.py --symbol MCL_FUT --show

# Run backtest and save results (default)
python scripts/run_backtest.py --symbol MCL_FUT

# Skip saving results
python scripts/run_backtest.py --symbol MCL_FUT --no-save-results
```

## summary.json Schema

The summary.json file is designed for AI consumption:

```json
{
  "run_id": "2026-01-10_143052_ema_MCL_FUT",
  "timestamp": "2026-01-10T14:30:52Z",
  "symbol": "MCL_FUT",
  "strategy": {
    "name": "EMA Crossover",
    "class": "MomentumStrategy",
    "parameters": {...}
  },
  "performance": {
    "total_return_pct": 12.5,
    "sharpe_ratio": 1.2,
    "max_drawdown_pct": 8.5,
    "win_rate": 55.0,
    ...
  },
  "time_analysis": {
    "best_hour": 10,
    "worst_hour": 15,
    "best_day_of_week": "Tuesday",
    ...
  },
  "recommendations": [
    "Consider adding time filter: avoid trading 15:00-16:00",
    ...
  ]
}
```

## Results Directory Structure

```
results/
├── backtests/
│   └── 2026-01-10_143052_ema_MCL_FUT/
│       ├── summary.json
│       ├── trades.csv
│       ├── equity_curve.csv
│       ├── daily_pnl.csv
│       ├── parameters.json
│       ├── report.html
│       └── charts/
│           ├── equity_curve.html
│           ├── drawdown.html
│           ├── trade_distribution.html
│           └── ...
```

## Dependencies

- `plotly>=5.18.0` - Interactive charts
- `dash>=2.14.0` - Web dashboard
- `dash-bootstrap-components>=1.5.0` - UI components

## Performance Considerations

- Charts use WebGL for large datasets (1M+ points)
- Equity curve sampling for >100k points
- Lazy loading of chart tabs in dashboard
- CDN-hosted Plotly.js for faster HTML reports

## Extending

To add a new chart type:

1. Add method to `ChartGenerator` class
2. Add chart to `_save_charts()` in `ResultSaver`
3. Optionally add to dashboard tabs in `BacktestViewer`

## Common Patterns

```python
# Custom chart theme
charts = ChartGenerator(theme="dark")

# Save without charts (faster)
saver.save(result, symbol="MCL_FUT", save_charts=False)

# Run dashboard on custom port
viewer = BacktestViewer(result, symbol="MCL_FUT")
viewer.show(port=8888, debug=True)
```
