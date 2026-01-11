# Config Module - Claude Context

## Purpose
Centralized configuration management using Pydantic Settings.

## Files

| File | Description |
|------|-------------|
| `settings.py` | `Settings` class - loads from `.env` with validation |
| `logging_config.py` | Structured logging setup with structlog |

## Settings Class

All settings loaded from environment variables or `.env` file.

### Key Settings

```python
settings = get_settings()  # Cached singleton

# Databento
settings.databento_api_key      # Required
settings.databento_dataset      # "GLBX.MDP3"
settings.default_schema         # "ohlcv-1m"

# Interactive Brokers
settings.ib_host                # "127.0.0.1"
settings.ib_port                # 7497 (paper), 7496 (live)
settings.ib_client_id           # 1

# Trading Symbol
settings.symbol                 # "MCL"
settings.exchange               # "NYMEX"
settings.contract_multiplier    # 100.0
settings.tick_size              # 0.01

# Risk Management
settings.initial_capital        # 5000.0
settings.max_position_contracts # 5
settings.risk_per_trade         # 0.02 (2%)
settings.daily_loss_limit       # 0.05 (5%)

# Strategy
settings.fast_period            # 20
settings.slow_period            # 50
settings.atr_period             # 14

# Paths
settings.data_dir               # Path("data")
settings.raw_data_dir           # data/raw
settings.processed_data_dir     # data/processed
settings.log_dir                # Path("logs")

# Environment
settings.environment            # "development" | "paper" | "production"
settings.log_level              # "INFO"
settings.is_paper_trading       # True if port is 7497 or 4002
```

## Logging Setup

```python
from config.logging_config import setup_logging, get_logger

# Initialize once at startup
setup_logging(log_level="INFO", log_dir=Path("logs"))

# Get logger in any module
logger = get_logger(__name__)

# Structured logging
logger.info("trade_executed", symbol="MCL", price=58.50, quantity=1)
```

## Environment File (.env)

Required variables:
```bash
DATABENTO_API_KEY=db-xxx
```

Optional (have defaults):
```bash
SYMBOL=MCL
INITIAL_CAPITAL=5000
IB_PORT=7497
LOG_LEVEL=INFO
```

## Validation

- `slow_period` must be > `fast_period`
- `risk_per_trade` must be 0.001-0.10
- `daily_loss_limit` must be 0.01-0.20

## Usage Pattern

```python
from config import get_settings

def my_function():
    settings = get_settings()  # Returns cached instance
    capital = settings.initial_capital
```

## Dependencies
- pydantic
- pydantic-settings
- python-dotenv
- structlog
