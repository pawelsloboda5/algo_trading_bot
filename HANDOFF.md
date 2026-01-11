# AI Agent Handoff Document

## Project Summary

This is an **algorithmic trading bot for Micro WTI Crude Oil futures (MCL)** using momentum/trend-following strategies. The project is in active development with Phase 1 (Data), Phase 2 (Backtesting), and Phase 3 (Risk Management) complete. **Data has been downloaded and converted to Parquet.**

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data pipeline (Databento client, storage) |
| Phase 2 | ✅ Complete | Backtesting framework (strategies, engine, metrics) |
| Phase 3 | ✅ Complete | Risk management (position sizing, circuit breakers) |
| Data | ✅ Complete | 2+ years of OHLCV data converted to Parquet |
| Phase 5 | 🔄 **NEXT** | Backtest visualization & results dashboard |
| Phase 4 | ❌ TODO | Paper trading (IB) - after strategy optimization |
| Phase 6 | ❌ TODO | AWS infrastructure |

## Data Available

Historical data downloaded from Databento and converted to Parquet:

| Schema | Date Range | Rows | Location |
|--------|------------|------|----------|
| OHLCV-1m | 2023-12-10 to 2026-01-08 | 1,663,134 | `data/raw/MCL_FUT/` |
| OHLCV-1h | 2023-12-10 to 2026-01-08 | ~27,000 | `data/raw/MCL_FUT/` |
| OHLCV-1d | 2023-12-10 to 2026-01-08 | ~750 | `data/raw/MCL_FUT/` |
| OHLCV-1s | 2025-01-09 to 2026-01-08 | ~15M | `data/raw/MCL_FUT/` |

File structure: `data/raw/MCL_FUT/{year}/{date}_{schema}.parquet`

## What's Been Built

### Phase 1: Data Pipeline ✅
- `src/data/databento_client.py` - Databento API wrapper
- `src/data/storage.py` - Parquet file storage with date partitioning
- `src/data/schemas.py` - Data models (OHLCV, BBO, OrderBook, ContractSpec)
- `scripts/download_historical.py` - CLI for data download
- `scripts/convert_databento.py` - DBN to Parquet converter

### Phase 2: Backtesting Framework ✅
- `src/strategy/indicators.py` - Numba-accelerated indicators (EMA, SMA, ATR, MACD, RSI, Bollinger)
- `src/strategy/base_strategy.py` - Abstract strategy interface
- `src/strategy/momentum_strategy.py` - EMA crossover and MACD strategies
- `src/backtest/portfolio.py` - Position tracking, trade log
- `src/backtest/metrics.py` - Performance metrics (Sharpe, MaxDD, Win Rate, etc.)
- `src/backtest/engine.py` - Vectorized backtester with risk management integration
- `scripts/run_backtest.py` - CLI for backtesting

### Phase 3: Risk Management ✅
- `src/risk/position_sizer.py` - Fixed fractional, volatility-based, Kelly criterion sizing
- `src/risk/risk_manager.py` - Daily loss limits, circuit breakers, trade validation
- `src/risk/stop_loss.py` - Fixed, trailing, ATR-based stops
- `src/risk/CLAUDE.md` - Module documentation

### Configuration
- `config/settings.py` - Pydantic Settings (loads from .env)
- `config/logging_config.py` - Structured logging with structlog
- `.env.example` - Environment template
- `pyproject.toml` - Dependencies and project config

## NEXT: Phase 5 - Visualization & Results Dashboard

### Goal
Create a comprehensive backtest results system that:
1. **Opens interactive visualization window** after each backtest
2. **Saves detailed results** to structured files for AI analysis
3. **Enables strategy optimization** through data-driven insights

### Files to Create

| File | Purpose |
|------|---------|
| `src/visualization/charts.py` | Interactive Plotly charts (equity curve, drawdown, trades) |
| `src/visualization/report.py` | Generate HTML reports and summary JSON |
| `src/visualization/window.py` | Pop-up visualization window (Plotly Dash) |
| `src/visualization/__init__.py` | Module exports |
| `src/visualization/CLAUDE.md` | Module documentation |

### Results Directory Structure

```
results/
├── backtests/
│   └── {timestamp}_{strategy}_{symbol}/
│       ├── summary.json          # Key metrics for AI analysis
│       ├── trades.csv            # All trades with full details
│       ├── equity_curve.csv      # Timestamped equity values
│       ├── daily_pnl.csv         # Daily P&L breakdown
│       ├── drawdowns.csv         # Drawdown periods
│       ├── signals.csv           # All signals generated
│       ├── parameters.json       # Strategy parameters used
│       ├── report.html           # Human-readable report
│       └── charts/
│           ├── equity_curve.html
│           ├── drawdown.html
│           ├── trade_distribution.html
│           └── monthly_returns.html
├── optimizations/
│   └── {timestamp}_{strategy}/
│       ├── parameter_sweep.csv   # All parameter combinations tested
│       └── best_params.json      # Optimal parameters found
└── comparisons/
    └── strategy_comparison.csv
```

### summary.json Schema (AI-Friendly)

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
  "performance": {
    "total_return_pct": 12.5,
    "total_return_usd": 625.0,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.8,
    "max_drawdown_pct": 8.5,
    "win_rate": 0.55,
    "profit_factor": 1.4,
    "total_trades": 120
  },
  "recommendations": [
    "Consider tightening stops - avg loser exceeds 2x risk target",
    "Performance degrades after 14:00 - consider time filter"
  ]
}
```

### CLI Updates

Update `scripts/run_backtest.py` to add:
```python
@click.option("--show", is_flag=True, help="Open visualization window")
@click.option("--save-results", is_flag=True, default=True, help="Save detailed results")
@click.option("--results-dir", default="results/backtests", help="Results directory")
```

## Key Technical Details

### Contract: MCL (Micro WTI Crude Oil)
- Exchange: NYMEX
- Multiplier: 100 barrels
- Tick size: $0.01 = $1/contract
- Margin: ~$900
- Initial capital: $5,000

### Libraries to Use
- `plotly` - Interactive charts (already in pyproject.toml)
- `dash` - Web-based visualization window
- Existing: `pandas`, `numpy`, `structlog`

## Run Backtest Command

```bash
# Basic backtest (works now)
python scripts/run_backtest.py --symbol MCL_FUT

# With EMA strategy and output
python scripts/run_backtest.py --symbol MCL_FUT --strategy ema --output results/ema.csv

# With MACD strategy
python scripts/run_backtest.py --symbol MCL_FUT --strategy macd --output results/macd.csv
```

## Code Patterns to Follow

### Logging
```python
from config.logging_config import get_logger
logger = get_logger(__name__)
logger.info("event_name", key1=value1, key2=value2)
```

### Settings
```python
from config.settings import get_settings
settings = get_settings()  # Cached singleton
```

### Strategy Interface
```python
class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return Series with values: +1 (long), -1 (short), 0 (flat)
        pass
```

## Important Files to Read

1. `CLAUDE.md` - Main project context
2. `PLAN.md` - Detailed Phase 5 visualization specs
3. `src/backtest/engine.py` - Understand BacktestResult structure
4. `src/backtest/metrics.py` - Available performance metrics
5. `scripts/run_backtest.py` - Current CLI implementation

---

**Continue from here by implementing Phase 5 (Visualization & Results Dashboard).**
