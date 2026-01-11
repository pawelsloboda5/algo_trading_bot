# Scripts Module - Claude Context

## Purpose
CLI scripts for operating the trading bot.

## Available Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `download_historical.py` | Download data from Databento | ✅ Complete |
| `run_backtest.py` | Run backtests on historical data | ✅ Complete |
| `run_paper_trading.py` | Live paper trading with IB | ❌ Not implemented |

## download_historical.py

Download MCL futures data from Databento.

```bash
# Basic usage (30 days of 1-minute bars)
python scripts/download_historical.py

# Specific date range
python scripts/download_historical.py --symbol MCL.FUT --start 2022-01-01 --end 2024-12-31

# Different schema
python scripts/download_historical.py --schema ohlcv-1h
python scripts/download_historical.py --schema mbp-1

# Cost estimate only (no download)
python scripts/download_historical.py --estimate-only

# Available schemas: ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d, trades
```

## run_backtest.py

Run backtest on downloaded data.

```bash
# Basic backtest (EMA crossover)
python scripts/run_backtest.py

# Custom parameters
python scripts/run_backtest.py --fast-period 10 --slow-period 30

# MACD strategy
python scripts/run_backtest.py --strategy macd

# Custom capital
python scripts/run_backtest.py --capital 10000

# Date range filter
python scripts/run_backtest.py --start 2023-06-01 --end 2024-01-01

# Save results
python scripts/run_backtest.py --output results/backtest.csv
```

### Backtest Output

```
==================================================
BACKTEST PERFORMANCE REPORT
==================================================

Total Return ($).................. $1,234.56
Total Return (%).................. 24.69%
CAGR.............................. 18.50%
Sharpe Ratio...................... 1.45
Sortino Ratio..................... 2.10
Max Drawdown ($).................. $500.00
Max Drawdown (%).................. 10.00%
Total Trades...................... 150
Win Rate.......................... 55.00%
Profit Factor..................... 1.80
```

## run_paper_trading.py (TODO)

Will connect to IB TWS/Gateway for paper trading.

Requirements:
- IB TWS or IB Gateway running
- API connections enabled
- Paper trading account configured

## Adding New Scripts

1. Create file in `scripts/`
2. Add `sys.path.insert(0, ...)` for imports
3. Use Click for CLI interface
4. Call `setup_logging()` at start
5. Use `get_settings()` for configuration

## Common Pattern

```python
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging, get_logger
from config.settings import get_settings

logger = get_logger(__name__)

@click.command()
@click.option("--option", default="value", help="Description")
def main(option: str):
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    # ... script logic ...

if __name__ == "__main__":
    main()
```

## Dependencies
- click (CLI framework)
- rich (optional, for formatted output)
