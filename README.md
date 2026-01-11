# Algo Trading Bot

Algorithmic trading system for **Micro WTI Crude Oil futures (MCL)** using momentum/trend-following strategies.

## Contract Details

| Spec | Value |
|------|-------|
| Symbol | MCL (Micro WTI Crude Oil) |
| Exchange | NYMEX |
| Multiplier | 100 barrels |
| Tick Size | $0.01 = $1/contract |
| Margin | ~$900 |
| Starting Capital | $5,000 (testing) |

## Tech Stack

**MVP (Python)** - Current phase
- Research, backtesting, monitoring
- Databento for market data (CME Globex MDP 3.0 dataset)
- Interactive Brokers for paper trading execution
- **Latency**: Seconds-level (sufficient for strategy validation)

**Production (Rust/C++)** - Future phase
- Live execution engine for millisecond/sub-millisecond trading
- Co-located on AWS Chicago Local Zone
- Direct market access via FIX protocol
- **Target Latency**: <5ms round-trip

**Infrastructure**
- Docker containerization
- AWS EC2 Chicago Local Zone (`us-east-1-chi-2a`) - enabled
- Terraform IaC

## Data Strategy

**Dataset**: CME Globex MDP 3.0 (`GLBX.MDP3`)
**Symbol**: `MCL.FUT` (continuous front-month)

### Data to Download

| Schema | Duration | Purpose | Priority |
|--------|----------|---------|----------|
| OHLCV-1m | 3 years (2022-01-01 to present) | Primary backtesting | 1 |
| OHLCV-1h | 3 years | Trend context, multi-timeframe | 2 |
| OHLCV-1d | 3 years | Regime detection | 2 |
| OHLCV-1s | 6 months (recent) | Precise entry/exit timing | 3 |
| MBP-1 | 1 year | Spread/slippage modeling for live trading | 4 |

### Download Commands

```bash
# 1. Three years of 1-minute bars (start here)
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1m --start 2022-01-01

# 2. Three years of hourly and daily
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1h --start 2022-01-01
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1d --start 2022-01-01

# 3. Six months of 1-second bars (large files)
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1s --start 2024-07-01

# 4. One year of L1 top-of-book
python scripts/download_historical.py --symbol MCL.FUT --schema mbp-1 --start 2024-01-01

# Cost estimate only (recommended before large downloads)
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1m --start 2022-01-01 --estimate-only
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env and add DATABENTO_API_KEY

# 3. Download data (start with 1-minute bars)
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1m --start 2022-01-01

# 4. Run backtest
python scripts/run_backtest.py

# 5. Paper trading (requires IB TWS/Gateway running)
python scripts/run_paper_trading.py
```

## Configuration

Copy `.env.example` to `.env` and configure:
- `DATABENTO_API_KEY` - Market data API key (required)
- `IB_HOST`, `IB_PORT` - Interactive Brokers connection
- Risk parameters (initial capital, max contracts, etc.)

## Databento Schemas Reference

| Schema | ID | Level | Use Case |
|--------|-----|-------|----------|
| OHLCV-1s | `ohlcv-1s` | L0 | Second bars, precise timing |
| OHLCV-1m | `ohlcv-1m` | L0 | Minute bars, primary backtest |
| OHLCV-1h | `ohlcv-1h` | L0 | Hourly bars, trend context |
| OHLCV-1d | `ohlcv-1d` | L0 | Daily bars, regime detection |
| MBP-1 | `mbp-1` | L1 | Top of book (BBO) |
| MBP-10 | `mbp-10` | L2 | Market depth (10 levels) |
| Trades | `trades` | L1 | Tick-by-tick trades |

## Project Status

- [x] Phase 1: Data pipeline (Databento)
- [x] Phase 2: Backtesting framework
- [ ] Phase 3: Risk management
- [ ] Phase 4: Paper trading (IB)
- [ ] Phase 5: Monitoring dashboard
- [ ] Phase 6: AWS infrastructure

## Strategies Available

| Strategy | Description | Parameters |
|----------|-------------|------------|
| EMA Crossover | Fast/slow EMA crossover with trend filter | fast_period, slow_period, trend_filter_period |
| MACD | MACD histogram zero-cross | fast_period, slow_period, signal_period |

## Backtest Usage

```bash
# Basic backtest (EMA crossover)
python scripts/run_backtest.py

# Custom parameters
python scripts/run_backtest.py --fast-period 10 --slow-period 30 --capital 10000

# MACD strategy
python scripts/run_backtest.py --strategy macd

# Save results
python scripts/run_backtest.py --output results/backtest.csv
```
