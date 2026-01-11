"""Technical indicators for trading strategies.

Vectorized implementations using NumPy and Numba for performance.
All functions operate on numpy arrays or pandas Series.
"""

import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True, cache=True)
def _ema_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated EMA calculation.

    Args:
        values: Price array
        period: EMA period

    Returns:
        EMA values array
    """
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values)
    result[0] = values[0]

    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]

    return result


def ema(data: pd.Series | np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Args:
        data: Price series
        period: EMA period

    Returns:
        EMA values
    """
    values = data.values if isinstance(data, pd.Series) else data
    return _ema_numba(values.astype(np.float64), period)


def sma(data: pd.Series | np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Args:
        data: Price series
        period: SMA period

    Returns:
        SMA values (NaN for first period-1 values)
    """
    values = data.values if isinstance(data, pd.Series) else data
    result = np.full_like(values, np.nan, dtype=np.float64)

    # Use cumsum for efficient rolling mean
    cumsum = np.cumsum(values)
    result[period - 1 :] = (cumsum[period - 1 :] - np.concatenate([[0], cumsum[:-period]])) / period

    return result


@jit(nopython=True, cache=True)
def _true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Numba-accelerated True Range calculation."""
    n = len(high)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    return tr


def atr(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range - volatility indicator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR values
    """
    h = high.values if isinstance(high, pd.Series) else high
    l = low.values if isinstance(low, pd.Series) else low
    c = close.values if isinstance(close, pd.Series) else close

    tr = _true_range_numba(
        h.astype(np.float64), l.astype(np.float64), c.astype(np.float64)
    )

    # Use EMA of True Range (Wilder's smoothing)
    return _ema_numba(tr, period)


def macd(
    data: pd.Series | np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence.

    Args:
        data: Price series
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    values = data.values if isinstance(data, pd.Series) else data
    values = values.astype(np.float64)

    fast_ema = _ema_numba(values, fast_period)
    slow_ema = _ema_numba(values, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = _ema_numba(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True)
def _rsi_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated RSI calculation."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    # Calculate price changes
    deltas = np.diff(values)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial average (simple mean for first period)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Subsequent values using Wilder's smoothing
    alpha = 1.0 / period
    for i in range(period, n - 1):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def rsi(data: pd.Series | np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index.

    Args:
        data: Price series
        period: RSI period (default: 14)

    Returns:
        RSI values (0-100)
    """
    values = data.values if isinstance(data, pd.Series) else data
    return _rsi_numba(values.astype(np.float64), period)


def bollinger_bands(
    data: pd.Series | np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Args:
        data: Price series
        period: Moving average period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)

    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    values = data.values if isinstance(data, pd.Series) else data
    values = values.astype(np.float64)

    middle = sma(values, period)

    # Rolling standard deviation
    rolling_std = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        rolling_std[i] = np.std(values[i - period + 1 : i + 1])

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    return upper, middle, lower


def crossover(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """Detect crossover events (series1 crosses above series2).

    Args:
        series1: First series
        series2: Second series

    Returns:
        Boolean array, True where crossover occurs
    """
    prev_diff = np.roll(series1 - series2, 1)
    curr_diff = series1 - series2

    # Crossover: was below (prev_diff <= 0), now above (curr_diff > 0)
    cross = (prev_diff <= 0) & (curr_diff > 0)
    cross[0] = False  # First element can't be a crossover

    return cross


def crossunder(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """Detect crossunder events (series1 crosses below series2).

    Args:
        series1: First series
        series2: Second series

    Returns:
        Boolean array, True where crossunder occurs
    """
    prev_diff = np.roll(series1 - series2, 1)
    curr_diff = series1 - series2

    # Crossunder: was above (prev_diff >= 0), now below (curr_diff < 0)
    cross = (prev_diff >= 0) & (curr_diff < 0)
    cross[0] = False  # First element can't be a crossunder

    return cross


def rate_of_change(data: pd.Series | np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change (momentum indicator).

    Args:
        data: Price series
        period: Lookback period (default: 10)

    Returns:
        ROC values as percentage
    """
    values = data.values if isinstance(data, pd.Series) else data
    values = values.astype(np.float64)

    result = np.full_like(values, np.nan)
    result[period:] = (values[period:] - values[:-period]) / values[:-period] * 100

    return result
