"""Momentum/Trend-following strategy using EMA crossover."""

import numpy as np
import pandas as pd

from src.strategy.base_strategy import BaseStrategy, Signal
from src.strategy.indicators import atr, crossover, crossunder, ema


class MomentumStrategy(BaseStrategy):
    """EMA Crossover Momentum Strategy.

    Generates long signals when fast EMA crosses above slow EMA,
    and short signals when fast EMA crosses below slow EMA.

    Features:
    - ATR-based stop loss
    - Optional trend filter using longer-term EMA
    - Position holding until opposite signal
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trend_filter_period: int | None = 200,
    ):
        """Initialize momentum strategy.

        Args:
            fast_period: Fast EMA period (default: 20)
            slow_period: Slow EMA period (default: 50)
            atr_period: ATR period for stop loss calculation (default: 14)
            atr_multiplier: ATR multiplier for stop loss distance (default: 2.0)
            trend_filter_period: Long-term EMA for trend filter, None to disable (default: 200)
        """
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trend_filter_period = trend_filter_period

        # Store calculated values for stop-loss calculation
        self._last_atr: float | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Signal logic:
        - LONG (+1): Fast EMA crosses above Slow EMA
        - SHORT (-1): Fast EMA crosses below Slow EMA
        - Signals persist until opposite crossover

        With trend filter enabled:
        - Only take LONG signals when price > trend EMA
        - Only take SHORT signals when price < trend EMA

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Series of signals (+1, -1, 0)
        """
        self.validate_data(data)

        # Normalize column names to lowercase
        df = data.copy()
        df.columns = df.columns.str.lower()

        close = df["close"].values

        # Calculate EMAs
        fast_ema = ema(close, self.fast_period)
        slow_ema = ema(close, self.slow_period)

        # Calculate ATR for stop-loss
        self._last_atr = atr(
            df["high"].values, df["low"].values, close, self.atr_period
        )[-1]

        # Detect crossovers
        long_cross = crossover(fast_ema, slow_ema)
        short_cross = crossunder(fast_ema, slow_ema)

        # Initialize signals
        signals = np.zeros(len(close), dtype=np.int8)

        # Apply trend filter if enabled
        if self.trend_filter_period is not None:
            trend_ema = ema(close, self.trend_filter_period)
            # Only allow longs when price > trend, shorts when price < trend
            long_cross = long_cross & (close > trend_ema)
            short_cross = short_cross & (close < trend_ema)

        # Generate signals that persist until opposite signal
        position = 0
        for i in range(len(close)):
            if long_cross[i]:
                position = Signal.LONG
            elif short_cross[i]:
                position = Signal.SHORT
            signals[i] = position

        return pd.Series(signals, index=data.index, name="signal")

    def get_stop_loss(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate ATR-based stop loss.

        Args:
            data: OHLCV DataFrame
            position: Current position direction
            entry_price: Entry price

        Returns:
            Stop-loss price
        """
        if position == Signal.FLAT:
            return None

        # Calculate current ATR
        df = data.copy()
        df.columns = df.columns.str.lower()

        current_atr = atr(
            df["high"].values, df["low"].values, df["close"].values, self.atr_period
        )[-1]

        stop_distance = current_atr * self.atr_multiplier

        if position == Signal.LONG:
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
            "trend_filter_period": self.trend_filter_period,
        }


class MACDMomentumStrategy(BaseStrategy):
    """MACD-based Momentum Strategy.

    Uses MACD histogram crossover for signals, with optional
    RSI filter to avoid overbought/oversold conditions.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            atr_period: ATR period for stop loss (default: 14)
            atr_multiplier: ATR multiplier for stop distance (default: 2.0)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals from MACD histogram crossover.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Series of signals
        """
        from src.strategy.indicators import macd

        self.validate_data(data)

        df = data.copy()
        df.columns = df.columns.str.lower()
        close = df["close"].values

        # Calculate MACD
        macd_line, signal_line, histogram = macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )

        # Signals based on histogram zero crossover
        signals = np.zeros(len(close), dtype=np.int8)

        position = 0
        for i in range(1, len(close)):
            # Histogram crosses above zero -> Long
            if histogram[i - 1] <= 0 and histogram[i] > 0:
                position = Signal.LONG
            # Histogram crosses below zero -> Short
            elif histogram[i - 1] >= 0 and histogram[i] < 0:
                position = Signal.SHORT
            signals[i] = position

        return pd.Series(signals, index=data.index, name="signal")

    def get_stop_loss(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate ATR-based stop loss."""
        if position == Signal.FLAT:
            return None

        df = data.copy()
        df.columns = df.columns.str.lower()

        current_atr = atr(
            df["high"].values, df["low"].values, df["close"].values, self.atr_period
        )[-1]

        stop_distance = current_atr * self.atr_multiplier

        if position == Signal.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "atr_period": self.atr_period,
            "atr_multiplier": self.atr_multiplier,
        }
