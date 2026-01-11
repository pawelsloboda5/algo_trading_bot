# Backtest Module - Claude Context

## Purpose
Backtesting framework for validating trading strategies against historical MCL futures data.

## Files

| File | Description |
|------|-------------|
| `engine.py` | Main `BacktestEngine` class - runs simulations |
| `portfolio.py` | `Portfolio`, `Position`, `Trade` classes for state tracking |
| `metrics.py` | `PerformanceMetrics` and calculation functions |

## Key Classes

### BacktestEngine
```python
engine = BacktestEngine(config)
result = engine.run(data, strategy)
```

### BacktestConfig
```python
@dataclass
class BacktestConfig:
    initial_capital: float = 5_000.0      # Starting cash
    contract_multiplier: float = 100.0     # MCL = 100 barrels
    commission_per_contract: float = 2.25  # IB futures commission
    slippage_ticks: int = 1               # 1 tick = $0.01 = $1 for MCL
    tick_size: float = 0.01
    position_size: int = 1                # Contracts per trade
```

### BacktestResult
Contains:
- `metrics`: PerformanceMetrics object
- `equity_curve`: pd.Series of equity over time
- `trades`: pd.DataFrame of all completed trades
- `signals`: pd.Series of trading signals
- `strategy`: The strategy that was tested
- `config`: The config used

## Portfolio State Tracking

### Portfolio
- Tracks cash, open position, trade history, equity curve
- Methods: `open_position()`, `close_position()`, `get_equity()`

### Position
- Entry time, price, quantity, direction
- Optional stop_loss and take_profit
- `unrealized_pnl(current_price)` method

### Trade (completed)
- Entry/exit times and prices
- Direction, quantity, commission, P&L
- Properties: `return_pct`, `duration`, `is_winner`

## Performance Metrics

| Metric | Description |
|--------|-------------|
| total_return | Absolute $ return |
| total_return_pct | Return as % of initial capital |
| cagr | Compound Annual Growth Rate |
| sharpe_ratio | Risk-adjusted return (annualized) |
| sortino_ratio | Downside risk-adjusted return |
| calmar_ratio | CAGR / Max Drawdown |
| max_drawdown | Largest peak-to-trough decline ($) |
| max_drawdown_pct | Max drawdown as % |
| win_rate | % of winning trades |
| profit_factor | Gross profit / Gross loss |
| avg_trade_pnl | Average P&L per trade |

## Usage Example

```python
from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import MomentumStrategy

config = BacktestConfig(initial_capital=5000)
engine = BacktestEngine(config)
strategy = MomentumStrategy(fast_period=20, slow_period=50)

result = engine.run(data, strategy)
print(result.metrics)
```

## Simulation Logic

1. Generate signals from strategy
2. For each bar:
   - Check stop-loss on open positions
   - If signal changes, close existing position
   - Open new position if signal is non-zero
   - Apply slippage to entry/exit prices
   - Deduct commission
   - Record equity

## Realistic Costs Modeled

- **Commission**: $2.25 per contract (IB futures)
- **Slippage**: 1 tick ($1 for MCL) on entry and exit
- **Round-trip cost**: ~$4.50 per contract

## Limitations

- Uses close price for signals (no lookahead bias)
- Single position at a time (no pyramiding)
- No partial fills modeled
- Designed for bar-level backtesting, not tick-by-tick

## Dependencies
- pandas
- numpy
- Requires `src.strategy` module
