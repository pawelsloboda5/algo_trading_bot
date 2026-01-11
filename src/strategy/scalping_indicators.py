"""Advanced indicators for high-frequency scalping strategies.

Numba-accelerated implementations optimized for 1-second data.
All functions operate on numpy arrays for maximum performance.
"""

import numpy as np
import pandas as pd
from numba import jit

from .indicators import _ema_numba, atr


@jit(nopython=True, cache=True)
def _volume_acceleration_numba(volume: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated volume acceleration calculation.

    Volume acceleration = current_volume / EMA(volume, period)
    Values > 1.0 indicate above-average volume.
    Values > 1.5 indicate significant volume surge.

    Args:
        volume: Volume array
        period: EMA period for average volume

    Returns:
        Volume acceleration ratio (current / average)
    """
    n = len(volume)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    # Calculate EMA of volume
    volume_ema = _ema_numba(volume, period)

    # Calculate acceleration ratio
    for i in range(period, n):
        if volume_ema[i] > 0:
            result[i] = volume[i] / volume_ema[i]
        else:
            result[i] = 1.0

    return result


def volume_acceleration(
    volume: pd.Series | np.ndarray, period: int = 20
) -> np.ndarray:
    """Volume Acceleration - detect unusual volume surges.

    Measures current volume relative to recent average.
    Used to confirm entry signals with participation.

    Args:
        volume: Volume series
        period: Lookback period for average (default: 20)

    Returns:
        Volume acceleration ratio (>1.5 = significant surge)
    """
    values = volume.values if isinstance(volume, pd.Series) else volume
    return _volume_acceleration_numba(values.astype(np.float64), period)


@jit(nopython=True, cache=True)
def _volatility_percentile_numba(
    atr_values: np.ndarray, lookback: int
) -> np.ndarray:
    """Numba-accelerated volatility percentile calculation.

    Calculates where current ATR falls in the distribution
    of recent ATR values (0-100 percentile).

    Args:
        atr_values: ATR values array
        lookback: Number of bars for percentile calculation

    Returns:
        Percentile rank (0-100) of current volatility
    """
    n = len(atr_values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < lookback:
        return result

    for i in range(lookback, n):
        # Get lookback window (excluding current bar)
        window = atr_values[i - lookback : i]
        current = atr_values[i]

        # Count how many values in window are less than current
        count_below = 0
        valid_count = 0
        for j in range(lookback):
            if not np.isnan(window[j]):
                valid_count += 1
                if window[j] < current:
                    count_below += 1

        if valid_count > 0:
            result[i] = (count_below / valid_count) * 100.0

    return result


def volatility_percentile(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    atr_period: int = 10,
    lookback: int = 100,
) -> np.ndarray:
    """Volatility Percentile - determine current volatility regime.

    Calculates where current ATR ranks vs recent history.
    Used to filter trades: only trade when volatility is in tradable range.

    Interpretation:
        - < 20: Very low volatility, market is quiet (avoid)
        - 20-80: Normal volatility, good for trading
        - > 80: Very high volatility, chaotic (avoid)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        atr_period: ATR calculation period (default: 10)
        lookback: Percentile lookback period (default: 100)

    Returns:
        Volatility percentile (0-100)
    """
    # Calculate ATR
    atr_values = atr(high, low, close, period=atr_period)

    return _volatility_percentile_numba(atr_values, lookback)


@jit(nopython=True, cache=True)
def _momentum_strength_numba(
    close: np.ndarray, period: int
) -> np.ndarray:
    """Numba-accelerated weighted momentum strength calculation.

    Calculates weighted average of returns over period.
    More recent returns are weighted higher.

    Args:
        close: Close prices
        period: Lookback period for momentum

    Returns:
        Momentum strength as percentage (positive = bullish)
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    # Create weights (1, 2, 3, ..., period) - more recent = higher weight
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = np.sum(weights)

    for i in range(period, n):
        # Calculate weighted returns
        total = 0.0
        for j in range(period):
            idx = i - period + j + 1
            prev_idx = idx - 1
            if close[prev_idx] > 0:
                ret = (close[idx] - close[prev_idx]) / close[prev_idx]
                total += ret * weights[j]

        result[i] = (total / weight_sum) * 100.0  # Convert to percentage

    return result


def momentum_strength(
    close: pd.Series | np.ndarray, period: int = 5
) -> np.ndarray:
    """Momentum Strength - weighted directional momentum.

    Measures price momentum with higher weight on recent bars.
    Used to confirm entry direction with magnitude.

    Interpretation:
        - > 0.015%: Strong bullish momentum
        - < -0.015%: Strong bearish momentum
        - Between: Weak/no momentum

    Args:
        close: Close prices
        period: Lookback period (default: 5)

    Returns:
        Momentum strength as percentage
    """
    values = close.values if isinstance(close, pd.Series) else close
    return _momentum_strength_numba(values.astype(np.float64), period)


@jit(nopython=True, cache=True)
def _fast_trend_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated fast trend calculation using linear regression slope.

    Uses least-squares linear regression slope normalized by price.

    Args:
        close: Close prices
        period: Regression period

    Returns:
        Normalized trend direction (positive = uptrend)
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    # Precompute x values and sums for linear regression
    # y = mx + b, we want slope m
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        # Get y values (prices in window)
        y = close[i - period + 1 : i + 1]
        y_mean = np.mean(y)

        # Calculate covariance
        cov = 0.0
        for j in range(period):
            cov += (x[j] - x_mean) * (y[j] - y_mean)

        # Slope = cov(x,y) / var(x)
        slope = cov / x_var

        # Normalize by price to get direction independent of price level
        if y_mean > 0:
            result[i] = slope / y_mean * 100.0  # As percentage per bar

    return result


def fast_trend(close: pd.Series | np.ndarray, period: int = 10) -> np.ndarray:
    """Fast Trend - quick trend direction using regression slope.

    Uses linear regression slope normalized by price level.
    Faster to respond than EMA crossover.

    Interpretation:
        - > 0: Uptrend
        - < 0: Downtrend
        - Magnitude indicates trend strength

    Args:
        close: Close prices
        period: Regression period (default: 10)

    Returns:
        Normalized trend direction (percentage per bar)
    """
    values = close.values if isinstance(close, pd.Series) else close
    return _fast_trend_numba(values.astype(np.float64), period)


@jit(nopython=True, cache=True)
def _price_velocity_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Numba-accelerated price velocity (rate of change per second).

    Args:
        close: Close prices
        period: Lookback period

    Returns:
        Price velocity (change per bar as percentage)
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period, n):
        if close[i - period] > 0:
            total_change = (close[i] - close[i - period]) / close[i - period]
            result[i] = (total_change / period) * 100.0  # Per-bar velocity

    return result


def price_velocity(
    close: pd.Series | np.ndarray, period: int = 5
) -> np.ndarray:
    """Price Velocity - speed of price movement.

    Measures how fast price is moving per bar.
    Useful for detecting acceleration/deceleration.

    Args:
        close: Close prices
        period: Lookback period (default: 5)

    Returns:
        Price velocity (percentage change per bar)
    """
    values = close.values if isinstance(close, pd.Series) else close
    return _price_velocity_numba(values.astype(np.float64), period)


@jit(nopython=True, cache=True)
def _volume_price_trend_numba(
    close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    """Numba-accelerated Volume Price Trend calculation.

    VPT = previous_VPT + volume * ((close - prev_close) / prev_close)

    Args:
        close: Close prices
        volume: Volume

    Returns:
        VPT cumulative values
    """
    n = len(close)
    result = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if close[i - 1] > 0:
            price_change_pct = (close[i] - close[i - 1]) / close[i - 1]
            result[i] = result[i - 1] + volume[i] * price_change_pct

    return result


def volume_price_trend(
    close: pd.Series | np.ndarray,
    volume: pd.Series | np.ndarray,
) -> np.ndarray:
    """Volume Price Trend - cumulative volume-weighted price change.

    Combines price movement with volume to show buying/selling pressure.
    Rising VPT = accumulation, Falling VPT = distribution.

    Args:
        close: Close prices
        volume: Volume

    Returns:
        VPT cumulative values
    """
    c = close.values if isinstance(close, pd.Series) else close
    v = volume.values if isinstance(volume, pd.Series) else volume
    return _volume_price_trend_numba(
        c.astype(np.float64), v.astype(np.float64)
    )


def candle_strength(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
) -> np.ndarray:
    """Candle Strength - measure of directional conviction in each bar.

    Calculates ratio of body to total range.
    High values indicate strong directional moves.

    Interpretation:
        - > 0.7: Strong bullish candle (close near high)
        - < -0.7: Strong bearish candle (close near low)
        - Near 0: Indecision (doji-like)

    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Candle strength (-1 to +1)
    """
    o = open_.values if isinstance(open_, pd.Series) else open_
    h = high.values if isinstance(high, pd.Series) else high
    l = low.values if isinstance(low, pd.Series) else low
    c = close.values if isinstance(close, pd.Series) else close

    # Total range
    total_range = h - l

    # Body (positive if bullish, negative if bearish)
    body = c - o

    # Calculate strength
    result = np.zeros_like(c, dtype=np.float64)
    mask = total_range > 0
    result[mask] = body[mask] / total_range[mask]

    return result
