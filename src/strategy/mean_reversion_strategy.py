"""Mean Reversion Strategy for MCL Scalping.

Based on quantitative analysis showing:
1. Trend following loses money on MCL (25-35% win rate)
2. Mean reversion signals have edge (>70% win rate on EMA deviation)
3. High volume bars tend to reverse, not continue
4. Best hours: 14:00-20:00 UTC (8AM-2PM Chicago)

This strategy FADES exhaustion moves rather than following momentum.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal
from .indicators import atr, ema, rsi
from .session_filter import SessionFilter


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""

    # EMA deviation parameters
    ema_period: int = 20  # EMA for mean calculation
    ema_deviation_threshold: float = 0.3  # 0.3% deviation to trigger signal

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 35.0  # More permissive than classic 30
    rsi_overbought: float = 65.0  # More permissive than classic 70

    # Volume parameters
    volume_ma_period: int = 50  # Rolling volume average
    volume_spike_threshold: float = 2.0  # 2x average = spike

    # ATR and stops
    atr_period: int = 14
    stop_loss_cents: float = 10.0  # Fixed 10 cent stop ($0.10)
    take_profit_cents: float = 8.0  # Fixed 8 cent target ($0.08)

    # Time management
    max_hold_bars: int = 20  # 20 bars max (20 minutes on 1m data)

    # Session filter (14:00-20:00 UTC = 8AM-2PM Chicago)
    session_start_hour: int = 14
    session_end_hour: int = 20
    use_session_filter: bool = True

    # Entry confirmation
    require_confirmation: bool = True  # Wait for reversal bar


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy that fades exhaustion moves.

    Entry Logic:
    ------------
    LONG signal when ANY of these conditions met:
    1. Price > 0.3% below EMA(20) - extended down, expect bounce
    2. RSI(14) < 35 - oversold
    3. High volume (>2x avg) DOWN bar - selling exhaustion

    Plus confirmation (if enabled):
    - Current bar closes above previous bar's close (reversal starting)

    SHORT signal when ANY of these conditions met:
    1. Price > 0.3% above EMA(20) - extended up, expect pullback
    2. RSI(14) > 65 - overbought
    3. High volume (>2x avg) UP bar - buying exhaustion

    Plus confirmation (if enabled):
    - Current bar closes below previous bar's close (reversal starting)

    Exit Logic:
    -----------
    - Stop loss: 10 cents ($10/contract)
    - Take profit: 8 cents ($8/contract)
    - Time stop: 20 bars (20 minutes)

    Session Filter:
    ---------------
    - Only trade 14:00-20:00 UTC (peak MCL liquidity)
    """

    def __init__(self, config: MeanReversionConfig | None = None):
        """Initialize mean reversion strategy."""
        self.config = config or MeanReversionConfig()
        self._atr_values: np.ndarray | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            Series of Signal values
        """
        self.validate_data(data)

        n = len(data)
        signals = np.zeros(n, dtype=np.int8)

        # Need enough data for indicators
        min_bars = max(
            self.config.ema_period,
            self.config.rsi_period,
            self.config.volume_ma_period,
            self.config.atr_period,
        ) + 5

        if n < min_bars:
            return pd.Series(signals, index=data.index, name="signal")

        # Extract price data
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        open_price = data["open"].values
        volume = data["volume"].values

        # Calculate indicators
        ema_values = ema(close, self.config.ema_period)
        rsi_values = rsi(close, self.config.rsi_period)
        self._atr_values = atr(high, low, close, self.config.atr_period)

        # Volume analysis
        vol_ma = pd.Series(volume).rolling(self.config.volume_ma_period).mean().values
        vol_ratio = volume / (vol_ma + 1e-10)
        high_volume = vol_ratio > self.config.volume_spike_threshold

        # Bar direction
        bar_up = close > open_price
        bar_down = close < open_price

        # EMA deviation (percentage)
        ema_dev = (close - ema_values) / (ema_values + 1e-10) * 100

        # Session filter
        if self.config.use_session_filter and hasattr(data.index, 'hour'):
            try:
                # Handle both tz-aware and tz-naive
                hours = data.index.hour
                session_ok = (hours >= self.config.session_start_hour) & (
                    hours < self.config.session_end_hour
                )
            except Exception:
                session_ok = np.ones(n, dtype=bool)
        else:
            session_ok = np.ones(n, dtype=bool)

        # ============ LONG SIGNALS ============
        # Condition 1: Extended below EMA
        extended_down = ema_dev < -self.config.ema_deviation_threshold

        # Condition 2: RSI oversold
        rsi_oversold = rsi_values < self.config.rsi_oversold

        # Condition 3: High volume down bar (selling exhaustion)
        vol_exhaustion_sell = high_volume & bar_down

        # Any primary condition
        long_primary = extended_down | rsi_oversold | vol_exhaustion_sell

        # Confirmation: current close > previous close (bounce starting)
        if self.config.require_confirmation:
            reversal_up = np.zeros(n, dtype=bool)
            reversal_up[1:] = close[1:] > close[:-1]
            long_confirmed = long_primary & reversal_up
        else:
            long_confirmed = long_primary

        # ============ SHORT SIGNALS ============
        # Condition 1: Extended above EMA
        extended_up = ema_dev > self.config.ema_deviation_threshold

        # Condition 2: RSI overbought
        rsi_overbought = rsi_values > self.config.rsi_overbought

        # Condition 3: High volume up bar (buying exhaustion)
        vol_exhaustion_buy = high_volume & bar_up

        # Any primary condition
        short_primary = extended_up | rsi_overbought | vol_exhaustion_buy

        # Confirmation: current close < previous close (pullback starting)
        if self.config.require_confirmation:
            reversal_down = np.zeros(n, dtype=bool)
            reversal_down[1:] = close[1:] < close[:-1]
            short_confirmed = short_primary & reversal_down
        else:
            short_confirmed = short_primary

        # Apply session filter
        long_final = long_confirmed & session_ok
        short_final = short_confirmed & session_ok

        # Generate signals (LONG takes priority if both true)
        signals = np.where(
            long_final,
            Signal.LONG,
            np.where(short_final, Signal.SHORT, Signal.FLAT),
        )

        return pd.Series(signals, index=data.index, name="signal")

    def get_stop_loss(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate stop-loss price (fixed cents).

        Args:
            data: OHLCV DataFrame
            position: Position direction
            entry_price: Entry price

        Returns:
            Stop-loss price
        """
        stop_cents = self.config.stop_loss_cents / 100  # Convert to dollars

        if position == Signal.LONG:
            return entry_price - stop_cents
        elif position == Signal.SHORT:
            return entry_price + stop_cents

        return None

    def get_take_profit(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate take-profit price (fixed cents).

        Args:
            data: OHLCV DataFrame
            position: Position direction
            entry_price: Entry price

        Returns:
            Take-profit price
        """
        target_cents = self.config.take_profit_cents / 100  # Convert to dollars

        if position == Signal.LONG:
            return entry_price + target_cents
        elif position == Signal.SHORT:
            return entry_price - target_cents

        return None

    def get_parameters(self) -> dict:
        """Return current strategy parameters."""
        return {
            "ema_period": self.config.ema_period,
            "ema_deviation_threshold": f"{self.config.ema_deviation_threshold}%",
            "rsi_period": self.config.rsi_period,
            "rsi_levels": f"{self.config.rsi_oversold}/{self.config.rsi_overbought}",
            "volume_spike_threshold": f"{self.config.volume_spike_threshold}x",
            "stop_loss_cents": self.config.stop_loss_cents,
            "take_profit_cents": self.config.take_profit_cents,
            "max_hold_bars": self.config.max_hold_bars,
            "session_hours": f"{self.config.session_start_hour}-{self.config.session_end_hour} UTC",
            "require_confirmation": self.config.require_confirmation,
        }

    @property
    def current_atr(self) -> float:
        """Get most recent ATR value."""
        if self._atr_values is not None and len(self._atr_values) > 0:
            return self._atr_values[-1]
        return 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MeanReversionStrategy("
            f"EMA{self.config.ema_period}@{self.config.ema_deviation_threshold}%, "
            f"RSI{self.config.rsi_period}@{self.config.rsi_oversold}/{self.config.rsi_overbought}, "
            f"Stop={self.config.stop_loss_cents}c, TP={self.config.take_profit_cents}c)"
        )


# Alternative configurations for different market conditions


def create_conservative_config() -> MeanReversionConfig:
    """More conservative configuration with tighter filters."""
    return MeanReversionConfig(
        ema_deviation_threshold=0.4,  # Require larger deviation
        rsi_oversold=30.0,  # Classic levels
        rsi_overbought=70.0,
        volume_spike_threshold=2.5,  # Higher volume requirement
        stop_loss_cents=8.0,  # Tighter stop
        take_profit_cents=6.0,  # Smaller target
        require_confirmation=True,
    )


def create_aggressive_config() -> MeanReversionConfig:
    """More aggressive configuration with looser filters."""
    return MeanReversionConfig(
        ema_deviation_threshold=0.2,  # Smaller deviation triggers
        rsi_oversold=40.0,  # More permissive
        rsi_overbought=60.0,
        volume_spike_threshold=1.5,  # Lower volume threshold
        stop_loss_cents=12.0,  # Wider stop
        take_profit_cents=10.0,  # Larger target
        require_confirmation=False,  # No confirmation needed
    )
