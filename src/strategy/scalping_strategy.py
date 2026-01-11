"""High-frequency scalping strategy for 1-second data.

Multi-factor entry with volume, momentum, and volatility confirmation.
Multi-level exits with profit targets, breakeven, and time stops.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal
from .indicators import atr, ema
from .scalping_indicators import (
    fast_trend,
    momentum_strength,
    volatility_percentile,
    volume_acceleration,
)
from .session_filter import SessionFilter, create_conservative_filter


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy."""

    # Momentum parameters
    momentum_period: int = 8  # 8 seconds of momentum
    momentum_threshold: float = 0.02  # 0.02% minimum momentum (balanced)

    # Volume parameters
    volume_ma_period: int = 20  # 20-second volume average
    volume_accel_threshold: float = 1.3  # 30% above average (more opportunities)

    # Volatility parameters
    volatility_lookback: int = 100  # ~1.5 minutes lookback
    volatility_min_pct: float = 15.0  # Minimum volatility percentile (more permissive)
    volatility_max_pct: float = 85.0  # Maximum volatility percentile

    # Trend parameters
    trend_period: int = 15  # 15-second trend
    trend_threshold: float = 0.0005  # Lower trend threshold

    # ATR and stops
    atr_period: int = 14  # 14-second ATR
    atr_stop_multiplier: float = 2.0  # Wider stop for scalping

    # Multi-level exit targets (in ATR units)
    target_1_atr: float = 0.75  # First target (easier to hit)
    target_2_atr: float = 1.5  # Second target
    target_3_atr: float = 2.5  # Final target
    breakeven_trigger_atr: float = 0.4  # Move to breakeven after this

    # Target allocation (must sum to 100)
    target_1_pct: float = 60.0  # Close 60% at target 1 (lock in profits faster)
    target_2_pct: float = 25.0  # Close 25% at target 2
    target_3_pct: float = 15.0  # Close remaining 15% at target 3

    # Time management
    max_hold_seconds: int = 120  # Maximum 2 minutes hold time

    # Session filter
    use_session_filter: bool = True


class ScalpingStrategy(BaseStrategy):
    """High-frequency scalping strategy for 1-second MCL data.

    Entry conditions (ALL must be true):
    1. momentum_strength > threshold (direction confirmed)
    2. volume_acceleration > 1.5 (participation confirmed)
    3. volatility_percentile between 20-80 (tradable regime)
    4. fast_trend aligns with direction
    5. Within allowed trading session (14:00-20:00 UTC)

    Exit system (ANY triggers exit):
    - Stop Loss: -1.5 ATR → Close 100%
    - Target 1: +1.0 ATR → Close 50%, move stop to breakeven
    - Target 2: +2.0 ATR → Close 25%, trail stop at 1 ATR
    - Target 3: +3.0 ATR → Close remaining 25%
    - Time Stop: 60 seconds max hold → Close 100%
    - Breakeven: +0.5 ATR → Move stop to entry + 1 tick
    """

    def __init__(self, config: ScalpingConfig | None = None):
        """Initialize scalping strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self.config = config or ScalpingConfig()
        self.session_filter = (
            create_conservative_filter() if self.config.use_session_filter else None
        )

        # Internal state for exit management (populated during generate_signals)
        self._atr_values: np.ndarray | None = None
        self._current_atr: float = 0.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume
                  Index should be datetime

        Returns:
            Series of Signal values (+1 long, -1 short, 0 flat)
        """
        self.validate_data(data)

        n = len(data)
        signals = np.zeros(n, dtype=np.int8)

        if n < max(
            self.config.momentum_period,
            self.config.volume_ma_period,
            self.config.volatility_lookback,
            self.config.trend_period,
            self.config.atr_period,
        ) + 10:
            return pd.Series(signals, index=data.index, name="signal")

        # Extract price data
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values

        # Calculate all indicators
        momentum = momentum_strength(close, period=self.config.momentum_period)
        vol_accel = volume_acceleration(volume, period=self.config.volume_ma_period)
        vol_pct = volatility_percentile(
            high,
            low,
            close,
            atr_period=self.config.atr_period,
            lookback=self.config.volatility_lookback,
        )
        trend = fast_trend(close, period=self.config.trend_period)

        # Store ATR for exit calculations
        self._atr_values = atr(high, low, close, period=self.config.atr_period)

        # Session filter
        if self.session_filter and self.config.use_session_filter:
            session_mask = self.session_filter.filter_signals_vectorized(data.index)
        else:
            session_mask = np.ones(n, dtype=bool)

        # Entry conditions (vectorized)
        # Long conditions
        long_momentum = momentum > self.config.momentum_threshold
        long_trend = trend > self.config.trend_threshold

        # Short conditions
        short_momentum = momentum < -self.config.momentum_threshold
        short_trend = trend < -self.config.trend_threshold

        # Common filters
        volume_confirmed = vol_accel > self.config.volume_accel_threshold
        volatility_ok = (vol_pct >= self.config.volatility_min_pct) & (
            vol_pct <= self.config.volatility_max_pct
        )

        # Combine conditions
        long_entry = (
            long_momentum
            & long_trend
            & volume_confirmed
            & volatility_ok
            & session_mask
        )

        short_entry = (
            short_momentum
            & short_trend
            & volume_confirmed
            & volatility_ok
            & session_mask
        )

        # Generate signals (without persistence - each bar is independent)
        signals = np.where(
            long_entry,
            Signal.LONG,
            np.where(short_entry, Signal.SHORT, Signal.FLAT),
        )

        return pd.Series(signals, index=data.index, name="signal")

    def get_stop_loss(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate initial stop-loss price.

        Args:
            data: OHLCV DataFrame
            position: Position direction
            entry_price: Entry price

        Returns:
            Stop-loss price
        """
        if self._atr_values is None or len(self._atr_values) == 0:
            return None

        # Use most recent ATR
        current_atr = self._atr_values[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            # Fallback: use percentage of price
            current_atr = entry_price * 0.002  # 0.2% fallback

        self._current_atr = current_atr
        stop_distance = current_atr * self.config.atr_stop_multiplier

        if position == Signal.LONG:
            return entry_price - stop_distance
        elif position == Signal.SHORT:
            return entry_price + stop_distance

        return None

    def get_take_profit(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate take-profit price (first target).

        Args:
            data: OHLCV DataFrame
            position: Position direction
            entry_price: Entry price

        Returns:
            Take-profit price (first target)
        """
        if self._atr_values is None or len(self._atr_values) == 0:
            return None

        current_atr = self._atr_values[-1]
        if np.isnan(current_atr) or current_atr <= 0:
            current_atr = entry_price * 0.002

        target_distance = current_atr * self.config.target_1_atr

        if position == Signal.LONG:
            return entry_price + target_distance
        elif position == Signal.SHORT:
            return entry_price - target_distance

        return None

    def get_exit_targets(
        self, entry_price: float, position: Signal, current_atr: float
    ) -> dict:
        """Get all exit targets for multi-level exit management.

        Args:
            entry_price: Position entry price
            position: Position direction
            current_atr: Current ATR value

        Returns:
            Dictionary with all targets and stop levels
        """
        if position == Signal.LONG:
            return {
                "stop_loss": entry_price - current_atr * self.config.atr_stop_multiplier,
                "breakeven_trigger": entry_price + current_atr * self.config.breakeven_trigger_atr,
                "target_1": entry_price + current_atr * self.config.target_1_atr,
                "target_2": entry_price + current_atr * self.config.target_2_atr,
                "target_3": entry_price + current_atr * self.config.target_3_atr,
            }
        elif position == Signal.SHORT:
            return {
                "stop_loss": entry_price + current_atr * self.config.atr_stop_multiplier,
                "breakeven_trigger": entry_price - current_atr * self.config.breakeven_trigger_atr,
                "target_1": entry_price - current_atr * self.config.target_1_atr,
                "target_2": entry_price - current_atr * self.config.target_2_atr,
                "target_3": entry_price - current_atr * self.config.target_3_atr,
            }

        return {}

    def check_exit_conditions(
        self,
        position_direction: Signal,
        entry_price: float,
        entry_time: datetime,
        current_price: float,
        current_high: float,
        current_low: float,
        current_time: datetime,
        current_atr: float,
        position_state: str,
        current_stop: float,
        tick_size: float = 0.01,
    ) -> dict | None:
        """Check all exit conditions for current position.

        Args:
            position_direction: LONG or SHORT
            entry_price: Position entry price
            entry_time: Position entry time
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            current_time: Current timestamp
            current_atr: Current ATR value
            position_state: Current exit state ("INITIAL", "BREAKEVEN", "TARGET_1", "TARGET_2")
            current_stop: Current stop loss price
            tick_size: Minimum price increment

        Returns:
            Dictionary with exit action or None if no exit
            {
                "action": "close_full" | "close_partial" | "update_stop",
                "price": exit_price,
                "percentage": percentage to close (for partial),
                "exit_type": type of exit,
                "new_stop": new stop price (for update_stop),
                "new_state": new position state,
            }
        """
        targets = self.get_exit_targets(entry_price, position_direction, current_atr)

        # Calculate current profit in ATR units
        if position_direction == Signal.LONG:
            profit = current_price - entry_price
            profit_atr = profit / current_atr if current_atr > 0 else 0

            # Check stop loss hit (use low for LONG)
            if current_low <= current_stop:
                return {
                    "action": "close_full",
                    "price": current_stop,
                    "exit_type": "stop_loss",
                }

            # Check target 3 (final exit)
            if profit_atr >= self.config.target_3_atr:
                return {
                    "action": "close_full",
                    "price": current_price,
                    "exit_type": "target_3",
                }

            # Check target 2
            if profit_atr >= self.config.target_2_atr and position_state not in ["TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_2_pct,
                    "exit_type": "target_2",
                    "new_stop": entry_price + current_atr,  # Trail stop at 1 ATR
                    "new_state": "TARGET_2",
                }

            # Check target 1
            if profit_atr >= self.config.target_1_atr and position_state not in ["TARGET_1", "TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_1_pct,
                    "exit_type": "target_1",
                    "new_stop": entry_price + tick_size,  # Move to breakeven
                    "new_state": "TARGET_1",
                }

            # Check breakeven trigger
            if profit_atr >= self.config.breakeven_trigger_atr and position_state == "INITIAL":
                return {
                    "action": "update_stop",
                    "new_stop": entry_price + tick_size,
                    "new_state": "BREAKEVEN",
                }

        elif position_direction == Signal.SHORT:
            profit = entry_price - current_price
            profit_atr = profit / current_atr if current_atr > 0 else 0

            # Check stop loss hit (use high for SHORT)
            if current_high >= current_stop:
                return {
                    "action": "close_full",
                    "price": current_stop,
                    "exit_type": "stop_loss",
                }

            # Check target 3 (final exit)
            if profit_atr >= self.config.target_3_atr:
                return {
                    "action": "close_full",
                    "price": current_price,
                    "exit_type": "target_3",
                }

            # Check target 2
            if profit_atr >= self.config.target_2_atr and position_state not in ["TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_2_pct,
                    "exit_type": "target_2",
                    "new_stop": entry_price - current_atr,  # Trail stop at 1 ATR
                    "new_state": "TARGET_2",
                }

            # Check target 1
            if profit_atr >= self.config.target_1_atr and position_state not in ["TARGET_1", "TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_1_pct,
                    "exit_type": "target_1",
                    "new_stop": entry_price - tick_size,  # Move to breakeven
                    "new_state": "TARGET_1",
                }

            # Check breakeven trigger
            if profit_atr >= self.config.breakeven_trigger_atr and position_state == "INITIAL":
                return {
                    "action": "update_stop",
                    "new_stop": entry_price - tick_size,
                    "new_state": "BREAKEVEN",
                }

        # Check time stop
        hold_duration = (current_time - entry_time).total_seconds()
        if hold_duration >= self.config.max_hold_seconds:
            return {
                "action": "close_full",
                "price": current_price,
                "exit_type": "time_stop",
            }

        return None

    def get_parameters(self) -> dict:
        """Return current strategy parameters."""
        return {
            "momentum_period": self.config.momentum_period,
            "momentum_threshold": self.config.momentum_threshold,
            "volume_ma_period": self.config.volume_ma_period,
            "volume_accel_threshold": self.config.volume_accel_threshold,
            "volatility_lookback": self.config.volatility_lookback,
            "volatility_range": f"{self.config.volatility_min_pct}-{self.config.volatility_max_pct}",
            "trend_period": self.config.trend_period,
            "atr_period": self.config.atr_period,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "targets_atr": f"{self.config.target_1_atr}/{self.config.target_2_atr}/{self.config.target_3_atr}",
            "max_hold_seconds": self.config.max_hold_seconds,
            "use_session_filter": self.config.use_session_filter,
        }

    @property
    def current_atr(self) -> float:
        """Get most recent ATR value."""
        return self._current_atr

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ScalpingStrategy("
            f"momentum={self.config.momentum_period}@{self.config.momentum_threshold}%, "
            f"volume={self.config.volume_ma_period}@{self.config.volume_accel_threshold}x, "
            f"ATR={self.config.atr_period}@{self.config.atr_stop_multiplier}x, "
            f"targets={self.config.target_1_atr}/{self.config.target_2_atr}/{self.config.target_3_atr}ATR)"
        )
