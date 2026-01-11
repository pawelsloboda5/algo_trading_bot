# AI Agent Prompt: Phase 5 - Backtest Visualization & Results Dashboard

## Context

You are continuing development on an algorithmic trading bot for Micro WTI Crude Oil futures (MCL). Phases 1-3 are complete (data pipeline, backtesting, risk management). Historical data (2+ years of OHLCV) has been downloaded and converted to Parquet format.

**Your task:** Implement Phase 5 - a comprehensive backtest visualization and results system.

## Primary Goals

1. **Interactive Visualization Window**: After running `python scripts/run_backtest.py`, open a pop-up window with interactive charts showing backtest results
2. **Structured Results Files**: Save detailed results to files that AI agents can parse to optimize the strategy
3. **Professional UX**: Create a polished, easy-to-analyze dashboard for the user

## Files to Create

### Core Visualization Module (`src/visualization/`)

1. **`src/visualization/charts.py`** - Plotly chart generators:
   - `create_equity_curve(equity_df, trades_df)` - Equity curve with trade entry/exit markers
   - `create_drawdown_chart(equity_df)` - Underwater chart showing drawdown periods
   - `create_trade_distribution(trades_df)` - Histogram of trade P&L
   - `create_monthly_returns_heatmap(equity_df)` - Calendar heatmap of monthly returns
   - `create_rolling_sharpe(equity_df, window=30)` - Rolling Sharpe ratio over time
   - `create_trade_analysis(trades_df)` - Win/loss by hour, day of week, hold time

2. **`src/visualization/report.py`** - Report generation:
   - `ResultSaver` class that saves all results to structured files
   - Generate `summary.json` with AI-friendly metrics
   - Generate `report.html` with embedded charts
   - Save CSVs: trades, equity_curve, daily_pnl, signals

3. **`src/visualization/window.py`** - Pop-up visualization:
   - Use Plotly Dash to create a local web app
   - Open browser automatically when `--show` flag is used
   - Tabs: Overview, Trades, Analysis, Parameters

4. **`src/visualization/__init__.py`** - Exports:
   ```python
   from .charts import ChartGenerator
   from .report import ResultSaver
   from .window import BacktestViewer
   ```

5. **`src/visualization/CLAUDE.md`** - Module documentation

### Results Directory

Create `results/` directory structure (see HANDOFF.md for full spec). Each backtest run creates:
```
results/backtests/{timestamp}_{strategy}_{symbol}/
├── summary.json          # AI-parseable metrics
├── trades.csv            # All trades
├── equity_curve.csv      # Timestamped equity
├── daily_pnl.csv         # Daily P&L
├── parameters.json       # Strategy params used
├── report.html           # Human-readable
└── charts/               # Individual chart HTMLs
```

### CLI Updates

Modify `scripts/run_backtest.py` to add options:
- `--show` - Open visualization window after backtest
- `--save-results` (default: True) - Save detailed results
- `--results-dir` - Custom results directory

## Key Requirements

### summary.json (Critical for AI Analysis)

This file must contain everything an AI needs to optimize the strategy:

```json
{
  "run_id": "2026-01-10_143052_ema_MCL_FUT",
  "timestamp": "2026-01-10T14:30:52Z",
  "symbol": "MCL_FUT",
  "strategy": {
    "name": "EMA Crossover",
    "class": "MomentumStrategy",
    "parameters": {
      "fast_period": 20,
      "slow_period": 50,
      "atr_period": 14
    }
  },
  "data_range": {
    "start": "2023-12-10",
    "end": "2026-01-08",
    "trading_days": 649,
    "total_bars": 1663134
  },
  "performance": {
    "total_return_pct": 12.5,
    "total_return_usd": 625.0,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.8,
    "calmar_ratio": 1.47,
    "max_drawdown_pct": 8.5,
    "max_drawdown_usd": 425.0,
    "win_rate": 0.55,
    "profit_factor": 1.4,
    "avg_trade_pnl": 15.5,
    "avg_winner": 45.0,
    "avg_loser": -25.0,
    "largest_winner": 150.0,
    "largest_loser": -80.0,
    "total_trades": 120,
    "winning_trades": 66,
    "losing_trades": 54,
    "avg_hold_time_minutes": 45,
    "max_consecutive_wins": 8,
    "max_consecutive_losses": 5
  },
  "risk_metrics": {
    "risk_per_trade": 0.02,
    "daily_loss_limit": 0.05,
    "trades_rejected": 5,
    "circuit_breaker_triggered": 0,
    "max_position_size_used": 3
  },
  "time_analysis": {
    "best_hour": 10,
    "worst_hour": 15,
    "best_day_of_week": "Tuesday",
    "worst_day_of_week": "Friday",
    "profitable_hours": [9, 10, 11, 14],
    "unprofitable_hours": [15, 16]
  },
  "recommendations": [
    "Consider adding time filter: avoid trading 15:00-16:00",
    "Win rate drops on Fridays - reduce position size or skip",
    "Avg loser ($25) exceeds 2x target ($20) - tighten stops"
  ]
}
```

### Visualization Window Features

The Dash app should include:

1. **Overview Tab**:
   - Large equity curve with trade markers
   - Key metrics cards (Total Return, Sharpe, Max DD, Win Rate)
   - Drawdown chart below equity curve

2. **Trades Tab**:
   - Interactive trade table (sortable, filterable)
   - Trade P&L distribution histogram
   - Cumulative P&L chart

3. **Analysis Tab**:
   - Monthly returns heatmap
   - Performance by hour of day
   - Performance by day of week
   - Rolling Sharpe ratio
   - Win/Loss streak analysis

4. **Parameters Tab**:
   - Strategy parameters used
   - Risk management settings
   - Suggestions for next optimization

## Technical Constraints

- Use `plotly` for all charts (already in pyproject.toml)
- Use `dash` for the visualization window (add to pyproject.toml)
- Follow existing code patterns (see `config/logging_config.py`, `src/backtest/engine.py`)
- Use structured logging with `structlog`
- Keep charts responsive and performant (1M+ data points)

## Existing Code to Reference

1. **`src/backtest/engine.py`** - `BacktestResult` dataclass contains:
   - `metrics: BacktestMetrics`
   - `trades: pd.DataFrame`
   - `equity_curve: pd.Series`
   - `signals: pd.Series`

2. **`src/backtest/metrics.py`** - `BacktestMetrics` contains:
   - All performance metrics (Sharpe, Sortino, MaxDD, etc.)
   - Has `to_dict()` and `__str__()` methods

3. **`scripts/run_backtest.py`** - Current CLI that needs updating

## Dependencies to Add

Add to `pyproject.toml`:
```toml
"dash>=2.14.0",
"dash-bootstrap-components>=1.5.0",
```

## Success Criteria

1. Running `python scripts/run_backtest.py --show` opens a browser with interactive dashboard
2. Running `python scripts/run_backtest.py` creates `results/backtests/{run_id}/` with all files
3. `summary.json` contains comprehensive metrics that an AI can parse to suggest optimizations
4. Charts are interactive, professional-looking, and load quickly
5. The user can easily understand what worked and what didn't in their backtest

## Example Usage After Implementation

```bash
# Run backtest and open visualization
python scripts/run_backtest.py --symbol MCL_FUT --show

# Run multiple backtests for comparison
python scripts/run_backtest.py --symbol MCL_FUT --strategy ema --fast-period 10 --slow-period 30
python scripts/run_backtest.py --symbol MCL_FUT --strategy ema --fast-period 20 --slow-period 50
python scripts/run_backtest.py --symbol MCL_FUT --strategy macd

# Results saved to results/backtests/ for AI analysis
```

## Start Here

1. Read `HANDOFF.md` for full project context
2. Read `src/backtest/engine.py` to understand `BacktestResult` structure
3. Read `src/backtest/metrics.py` to understand available metrics
4. Create `src/visualization/` module
5. Update `scripts/run_backtest.py` with new options
6. Test with `python scripts/run_backtest.py --symbol MCL_FUT --show`

---

**The goal is to make backtest results so clear and well-structured that both humans and AI agents can quickly understand performance and identify optimization opportunities.**
