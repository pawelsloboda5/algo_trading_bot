# Algo Trading Bot - Claude Code Instructions

## Project Overview
Algorithmic trading system for Micro WTI Crude Oil futures (MCL) using momentum/trend-following strategies.

## Tech Stack
- **MVP**: Python 3.11+ (research, backtesting, monitoring)
- **Production**: Rust/C++ for live execution (future)
- **Data**: Databento API (GLBX.MDP3 dataset)
- **Broker**: Interactive Brokers (ib-async library)
- **Infrastructure**: Docker + AWS EC2 Chicago Local Zone

## Project Structure
```
config/              - Configuration (Pydantic settings, logging)
src/data/            - Market data ingestion (Databento) and storage (Parquet)
src/strategy/        - Trading strategies and indicators
src/backtest/        - Backtesting framework
src/risk/            - Risk management (position sizing, stop-loss)
src/visualization/   - Charts, reports, and dashboards (Phase 5)
src/execution/       - Order execution (IB integration)
src/monitoring/      - Logging and dashboards
scripts/             - CLI scripts for operations
infrastructure/      - Docker and Terraform IaC
results/             - Backtest results and reports
```

## Key Conventions

### Code Style
- Python 3.11+ features (type hints, match statements, etc.)
- Ruff for linting, MyPy for type checking
- Use `structlog` for structured logging (JSON format in production)
- Pydantic for data validation and settings

### Data Handling
- All timestamps in UTC internally
- Store data in Parquet format (columnar, compressed)
- Use `ts_event` as the primary timestamp column (Databento convention)

### Databento Schemas
We use the following schemas for market data:
- **MBP-10** (`mbp-10`): L2 Market by price, 10 levels of market depth
- **MBP-1** (`mbp-1`): L1 Top of book, trades and quotes
- **OHLCV** (`ohlcv-1m`, `ohlcv-1s`): Aggregated bars for backtesting

### Trading Specifics
- **Primary Symbol**: MCL (Micro WTI Crude Oil Futures)
- MCL contract: 100 barrels multiplier, $0.01 tick = $1/contract
- Use `MCL.FUT` for continuous front-month futures in Databento
- Databento symbol format: `MCLG6` (Feb 2026), `MCLH6` (Mar 2026), etc.
- AWS Chicago Local Zone: `us-east-1-chi-2a` (enabled, NOT us-east-2)

### MCL vs CL Contract Comparison
| Contract | Symbol | Multiplier | Tick Size | Tick Value | Margin (approx) |
|----------|--------|------------|-----------|------------|-----------------|
| Micro WTI | MCL | 100 barrels | $0.01 | $1 | ~$900 |
| Full WTI | CL | 1000 barrels | $0.01 | $10 | ~$9,000 |

### Libraries
- Use `ib-async` (NOT `ib_insync` - it's abandoned)
- Use `databento` for market data
- Use `pandas` for DataFrames, `numpy` for numerical ops
- Use `numba` for performance-critical indicators

## Common Commands

```bash
# Install dependencies
pip install -e .
pip install -e ".[dev]"  # with dev tools

# Download historical data (OHLCV for backtesting)
python scripts/download_historical.py --symbol MCL.FUT --start 2024-01-01

# Download L2 market depth data
python scripts/download_historical.py --symbol MCL.FUT --schema mbp-10 --start 2024-12-01

# Run backtest
python scripts/run_backtest.py --symbol MCL_FUT

# Run backtest with interactive visualization
python scripts/run_backtest.py --symbol MCL_FUT --show

# Run backtest with custom parameters
python scripts/run_backtest.py --symbol MCL_FUT --strategy ema --fast-period 10 --slow-period 30

# Run tests
pytest tests/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Environment Setup
1. Copy `.env.example` to `.env`
2. Add your `DATABENTO_API_KEY`
3. Configure IB connection (port 7497 for paper trading)

## Important Notes
- Never commit `.env` or API keys
- Paper trading first, always verify before live
- All risk limits are enforced in `src/risk/risk_manager.py`

## Data Available

Historical data has been downloaded and converted to Parquet:

| Schema | Date Range | Trading Days | Location |
|--------|------------|--------------|----------|
| OHLCV-1m | 2023-12-10 to 2026-01-08 | 649 | `data/raw/MCL_FUT/` |
| OHLCV-1h | 2023-12-10 to 2026-01-08 | 649 | `data/raw/MCL_FUT/` |
| OHLCV-1d | 2023-12-10 to 2026-01-08 | 649 | `data/raw/MCL_FUT/` |
| OHLCV-1s | 2025-01-09 to 2026-01-08 | ~250 | `data/raw/MCL_FUT/` |

Total: ~1.6M rows of 1-minute data, 35+ MB Parquet files.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data pipeline (Databento client, Parquet storage) |
| Phase 2 | ✅ Complete | Backtesting framework (strategies, engine, metrics) |
| Phase 3 | ✅ Complete | Risk management (position sizing, circuit breakers) |
| Phase 4 | ❌ TODO | Paper trading (IB integration) |
| Phase 5 | ✅ Complete | Backtest visualization & results dashboard |
| Phase 6 | ❌ TODO | AWS infrastructure (Terraform, Docker) |
| Phase 7 | ❌ Future | Rust execution engine (for ms-level trading) |

## Directory Documentation

Each directory has a `CLAUDE.md` file with module-specific context:
- `config/CLAUDE.md` - Settings and logging
- `src/data/CLAUDE.md` - Databento client, storage
- `src/strategy/CLAUDE.md` - Strategies, indicators
- `src/backtest/CLAUDE.md` - Backtest engine, metrics
- `src/risk/CLAUDE.md` - Risk management, position sizing
- `src/visualization/CLAUDE.md` - Charts, reports, dashboards
- `scripts/CLAUDE.md` - CLI scripts
- `infrastructure/CLAUDE.md` - AWS, Docker plans

## Implementation Plan

See `PLAN.md` for detailed implementation plans for remaining phases.

## Latency Architecture

**Current (Python MVP)**: Seconds-level latency
- Suitable for: backtesting, strategy validation, paper trading
- Not suitable for: HFT, millisecond trading

**Future (Rust)**: Millisecond/sub-millisecond
- Required for: production trading at speed
- Co-located on AWS Chicago Local Zone
- Direct FIX protocol connection
