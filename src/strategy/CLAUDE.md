# Strategy Module - Claude Context

## Purpose
Trading strategies and technical indicators for the MCL futures trading bot.

## Files

| File | Description |
|------|-------------|
| `indicators.py` | Numba-accelerated technical indicators (EMA, SMA, ATR, MACD, RSI, Bollinger Bands) |
| `base_strategy.py` | Abstract `BaseStrategy` class - all strategies inherit from this |
| `momentum_strategy.py` | EMA crossover and MACD momentum strategies |

## Key Interfaces

### BaseStrategy (Abstract)
```python
class BaseStrategy(ABC):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Returns Series of Signal values: +1 (long), -1 (short), 0 (flat)"""

    def get_parameters(self) -> dict:
        """Returns strategy parameters for logging"""

    def get_stop_loss(self, data, position, entry_price) -> float | None:
        """Optional: ATR-based stop loss"""
```

### Signal Enum
```python
class Signal(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1
```

## Implemented Strategies

### 1. MomentumStrategy (EMA Crossover)
- **Signal**: Long when fast EMA > slow EMA, short when fast < slow
- **Parameters**: fast_period (20), slow_period (50), atr_period (14), atr_multiplier (2.0)
- **Optional**: trend_filter_period (200) - only trade in direction of long-term trend

### 2. MACDMomentumStrategy
- **Signal**: Long when MACD histogram crosses above zero, short when below
- **Parameters**: fast_period (12), slow_period (26), signal_period (9)

## Indicators Available

| Indicator | Function | Notes |
|-----------|----------|-------|
| EMA | `ema(data, period)` | Numba-accelerated |
| SMA | `sma(data, period)` | Vectorized with cumsum |
| ATR | `atr(high, low, close, period)` | Average True Range |
| MACD | `macd(data, fast, slow, signal)` | Returns (line, signal, histogram) |
| RSI | `rsi(data, period)` | Relative Strength Index |
| Bollinger | `bollinger_bands(data, period, std)` | Returns (upper, middle, lower) |
| Crossover | `crossover(s1, s2)` | Boolean array where s1 crosses above s2 |
| Crossunder | `crossunder(s1, s2)` | Boolean array where s1 crosses below s2 |

## Adding New Strategies

1. Create new file in `src/strategy/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` and `get_parameters()`
4. Optionally override `get_stop_loss()` and `get_take_profit()`
5. Add to `__init__.py` exports

## Performance Notes

- All indicators use Numba JIT compilation for speed
- First call has compilation overhead, subsequent calls are fast
- Designed for backtesting (vectorized), not live tick-by-tick
- For live trading at millisecond scale, these will be rewritten in Rust

## Dependencies
- numpy
- pandas
- numba (for JIT compilation)
