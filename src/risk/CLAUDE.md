# Risk Management Module

## Overview

Position sizing, risk limits, and stop-loss management for MCL futures trading.

## Components

### position_sizer.py

`PositionSizer` - Calculate contract quantity based on risk rules.

**Methods:**
- `fixed_fractional(equity, risk_per_trade, entry_price, stop_loss_price)` - Risk X% of equity per trade
- `fixed_contracts(num_contracts)` - Always trade N contracts
- `volatility_adjusted(equity, current_atr, risk_target)` - Size based on ATR volatility
- `kelly_criterion(win_rate, avg_win, avg_loss, fraction)` - Optimal f based on Kelly formula

**Example:**
```python
sizer = PositionSizer(contract_multiplier=100, max_contracts=5)
result = sizer.fixed_fractional(
    equity=5000,
    risk_per_trade=0.02,  # 2%
    entry_price=75.50,
    stop_loss_price=75.30,
)
# result.contracts = 5 (risking $100 with $20/contract stop)
```

### risk_manager.py

`RiskManager` - Central risk enforcement with daily limits and circuit breakers.

**Key Features:**
- Pre-trade validation
- Daily loss limit tracking (default 5%)
- Circuit breaker activation
- Position size calculation

**Usage:**
```python
from src.risk import RiskManager
from src.strategy.base_strategy import Signal

risk_manager = RiskManager(
    initial_capital=5_000,
    risk_per_trade=0.02,
    daily_loss_limit=0.05,
    max_position_contracts=5,
)

validation = risk_manager.validate_trade(
    signal=Signal.LONG,
    entry_price=75.50,
    stop_loss_price=75.30,
    current_equity=5000,
    timestamp=datetime.now(),
)

if validation.allowed:
    # Execute with validation.position_size contracts
    print(f"Trade approved: {validation.position_size} contracts")
else:
    print(f"Trade rejected: {validation.reason}")
```

### stop_loss.py

`StopLossManager` - Stop-loss calculation strategies.

**Methods:**
- `fixed_percentage(entry_price, percentage, direction)` - Stop at X% from entry
- `fixed_dollar(entry_price, dollar_risk, direction)` - Stop at $X distance per contract
- `atr_based(entry_price, current_atr, multiplier, direction)` - ATR-based adaptive stop
- `trailing_stop_update(...)` - Update trailing stop as price moves
- `time_based_exit(entry_time, current_time, max_hold_bars)` - Check max hold time

## MCL Contract Specifications

| Spec | Value |
|------|-------|
| Multiplier | 100 barrels |
| Tick size | $0.01 |
| Tick value | $1/contract |
| Typical margin | ~$900 |

**Example Position Sizing:**
```
Entry: $75.50
Stop: $75.30 (20 tick distance)
Stop distance: $0.20 × 100 = $20/contract

With $5,000 equity, 2% risk = $100
Position size: $100 / $20 = 5 contracts
```

## Integration with Backtest Engine

The backtest engine automatically uses RiskManager when `use_risk_manager=True` (default).

```python
from src.backtest.engine import run_backtest

result = run_backtest(
    data=ohlcv_data,
    strategy=momentum_strategy,
    initial_capital=5_000,
    use_risk_manager=True,  # Enable risk management
    risk_per_trade=0.02,    # 2% per trade
    daily_loss_limit=0.05,  # 5% daily limit
    max_position_contracts=5,
)
```

**With risk manager disabled:**
```python
result = run_backtest(
    data=ohlcv_data,
    strategy=momentum_strategy,
    use_risk_manager=False,  # Fixed position sizing
    position_size=1,         # Always 1 contract
)
```

## Risk Manager Behavior

### Daily Loss Limit
- Tracked as percentage of day-start equity
- Circuit breaker triggers at limit (default 5%)
- Resets automatically on new trading day

### Trade Rejection
Trades are rejected when:
1. Circuit breaker is active
2. Daily loss limit exceeded
3. No stop-loss provided
4. Calculated position size is 0 (stop too wide)

### Position Sizing
- Uses fixed fractional method by default
- Never exceeds `max_position_contracts`
- Returns 0 contracts if risk exceeds limits

## Logging

All risk events are logged with structlog:

```python
logger.info("trade_validated", signal="LONG", position_size=5)
logger.warning("trade_rejected", reason="daily loss limit exceeded")
logger.critical("circuit_breaker_triggered", reason="5% daily limit")
```

## Testing

```bash
# Run risk module tests
pytest tests/unit/test_risk/

# Run with verbose output
pytest tests/unit/test_risk/ -v
```
