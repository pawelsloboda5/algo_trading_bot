# Data Module - Claude Context

## Purpose
Market data ingestion from Databento and local storage management for MCL futures.

## Files

| File | Description |
|------|-------------|
| `databento_client.py` | Wrapper for Databento Historical API |
| `schemas.py` | Data models (OHLCVBar, BBO, OrderBookSnapshot, ContractSpec) |
| `storage.py` | Parquet file storage manager |

## Databento Configuration

- **Dataset**: `GLBX.MDP3` (CME Globex MDP 3.0)
- **Symbol**: `MCL.FUT` (continuous front-month Micro WTI)
- **API Key**: Set in `.env` as `DATABENTO_API_KEY`

## Schemas Used

| Schema | ID | Description | Data Size |
|--------|-----|-------------|-----------|
| OHLCV-1s | `ohlcv-1s` | 1-second bars | Large |
| OHLCV-1m | `ohlcv-1m` | 1-minute bars | Medium |
| OHLCV-1h | `ohlcv-1h` | 1-hour bars | Small |
| OHLCV-1d | `ohlcv-1d` | Daily bars | Tiny |
| MBP-1 | `mbp-1` | L1 Top of book (BBO) | Large |
| MBP-10 | `mbp-10` | L2 Market depth (10 levels) | Very Large |

## Data Download Plan

| Schema | Duration | Purpose |
|--------|----------|---------|
| OHLCV-1s | 6 months | Precise entry/exit timing |
| OHLCV-1m | 3 years | Primary backtest |
| OHLCV-1h | 3 years | Trend context |
| OHLCV-1d | 3 years | Regime detection |
| MBP-1 | 1 year | Spread/slippage modeling |

## DatabentoClient Usage

```python
from src.data import DatabentoClient

client = DatabentoClient(api_key="db-xxx")

# Cost estimate first!
cost = client.get_cost_estimate(["MCL.FUT"], "ohlcv-1m", start, end)

# Download data
df = client.get_historical_bars(["MCL.FUT"], start, end, schema="ohlcv-1m")
```

## DataStorage Usage

```python
from src.data import DataStorage

storage = DataStorage(Path("data/raw"))

# Save bulk data (splits by date)
storage.save_bulk(df, "MCL_FUT", "ohlcv-1m")

# Load date range
df = storage.load_date_range("MCL_FUT", start_date, end_date, "ohlcv-1m")

# List available dates
dates = storage.list_available_dates("MCL_FUT", "ohlcv-1m")
```

## Data Schemas (Python Classes)

### OHLCVBar
```python
@dataclass
class OHLCVBar:
    timestamp: datetime
    symbol: str
    open, high, low, close: float
    volume: int
```

### BBO (Best Bid/Offer - MBP-1)
```python
@dataclass
class BBO:
    timestamp: datetime
    symbol: str
    bid_price, bid_size: float, int
    ask_price, ask_size: float, int
```

### OrderBookSnapshot (MBP-10)
```python
@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    bids: tuple[MBPLevel, ...]  # Up to 10 levels
    asks: tuple[MBPLevel, ...]
```

## Contract Specifications

```python
MCL_CONTRACT = ContractSpec(
    symbol="MCL",
    exchange="NYMEX",
    multiplier=100.0,      # 100 barrels
    tick_size=0.01,        # $0.01 = $1/contract
    trading_hours="18:00-17:00 ET (Sun-Fri)",
)
```

## Storage Format

- **Format**: Apache Parquet (columnar, compressed)
- **Organization**: `data/raw/{symbol}/{year}/{date}_{schema}.parquet`
- **Timestamp column**: `ts_event` (Databento convention, nanosecond UTC)

## Important Notes

1. Always check cost estimate before downloading large datasets
2. Databento has $125 free credit for new accounts
3. `ts_event` is the exchange timestamp (use this for backtesting)
4. Data is stored in UTC - convert for display only

## Dependencies
- databento
- pandas
- pyarrow (for Parquet)
