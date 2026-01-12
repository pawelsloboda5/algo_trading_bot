"""Mean Reversion Strategy for MCL Scalping - Optimized Version.

Based on quantitative analysis showing:
1. Trend following loses money on MCL (25-35% win rate)
2. Mean reversion signals have edge (>70% win rate on EMA deviation)
3. High volume bars tend to reverse, not continue
4. Best hours: 14:00-20:00 UTC (8AM-2PM Chicago)

This strategy FADES exhaustion moves rather than following momentum.

Optimizations (v2):
- Multi-level scale-out exits (50%/30%/20%)
- Dynamic trailing stops after first target
- Volatility regime filter (avoid extreme vol environments)
- Stricter signal quality filters (require 2+ conditions)
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal
from .indicators import atr, ema, rsi
from .scalping_indicators import volatility_percentile


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""

    # === EMA Deviation Parameters ===
    ema_period: int = 20  # EMA for mean calculation
    ema_deviation_threshold: float = 0.3  # 0.3% deviation to trigger signal

    # === RSI Parameters ===
    rsi_period: int = 14
    rsi_oversold: float = 35.0  # More permissive than classic 30
    rsi_overbought: float = 65.0  # More permissive than classic 70

    # === Volume Parameters ===
    volume_ma_period: int = 50  # Rolling volume average
    volume_spike_threshold: float = 2.0  # 2x average = spike

    # === ATR Period ===
    atr_period: int = 14

    # === Multi-Level Exit Targets (in cents) ===
    stop_loss_cents: float = 11.0  # 11 cents stop ($11/contract)
    target_1_cents: float = 5.0  # First scale-out target
    target_2_cents: float = 10.0  # Second scale-out target
    target_3_cents: float = 15.0  # Final runner target

    # === Scale-Out Percentages ===
    target_1_pct: float = 50.0  # Close 50% at target 1
    target_2_pct: float = 30.0  # Close 30% at target 2
    target_3_pct: float = 20.0  # Let 20% run to target 3

    # === Breakeven & Trailing Stop ===
    breakeven_trigger_cents: float = 4.0  # Move stop to BE at +4 cents
    trail_after_target_1: bool = True  # Start trailing after target 1
    trail_offset_cents: float = 3.0  # Trail 3 cents behind price

    # === Time Management ===
    max_hold_bars: int = 20  # 20 bars max (20 minutes on 1m data)

    # === Session Filter (14:00-20:00 UTC = 8AM-2PM Chicago) ===
    session_start_hour: int = 14
    session_end_hour: int = 20
    use_session_filter: bool = True

    # === Entry Confirmation ===
    require_confirmation: bool = False  # Disabled for more signals

    # === Volatility Regime Filter ===
    use_volatility_filter: bool = True
    atr_percentile_lookback: int = 100  # Bars for percentile calculation
    min_volatility_percentile: float = 20.0  # Avoid dead markets
    max_volatility_percentile: float = 80.0  # Avoid chaotic markets

    # === Signal Quality Filters ===
    require_multiple_conditions: int = 1  # Require N of 3 conditions (1=any, 2=stricter)
    min_volume_threshold: int = 0  # Minimum volume per bar (0=disabled)

    # === Legacy compatibility ===
    take_profit_cents: float = 8.0  # Used by simple backtest mode


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy that fades exhaustion moves.

    Entry Logic:
    ------------
    LONG signal when N of these conditions met (configurable):
    1. Price > 0.3% below EMA(20) - extended down, expect bounce
    2. RSI(14) < 35 - oversold
    3. High volume (>2x avg) DOWN bar - selling exhaustion

    Plus optional filters:
    - Volatility regime (20-80 percentile)
    - Session filter (14:00-20:00 UTC)
    - Confirmation bar (close > previous close)

    Exit Logic (Multi-Level Scale-Out):
    ------------------------------------
    - Target 1: 5 cents → Close 50%, move stop to breakeven
    - Target 2: 10 cents → Close 30%, trail stop at 3c offset
    - Target 3: 15 cents → Close remaining 20%
    - Stop Loss: 11 cents
    - Time Stop: 20 bars max
    """

    def __init__(self, config: MeanReversionConfig | None = None):
        """Initialize mean reversion strategy."""
        self.config = config or MeanReversionConfig()
        self._atr_values: np.ndarray | None = None
        self._volatility_percentile: np.ndarray | None = None

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
            self.config.atr_percentile_lookback if self.config.use_volatility_filter else 0,
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

        # === VOLATILITY REGIME FILTER ===
        if self.config.use_volatility_filter:
            self._volatility_percentile = volatility_percentile(
                high, low, close,
                atr_period=self.config.atr_period,
                lookback=self.config.atr_percentile_lookback,
            )
            volatility_ok = (
                (self._volatility_percentile >= self.config.min_volatility_percentile) &
                (self._volatility_percentile <= self.config.max_volatility_percentile)
            )
        else:
            volatility_ok = np.ones(n, dtype=bool)

        # === VOLUME THRESHOLD FILTER ===
        if self.config.min_volume_threshold > 0:
            volume_ok = volume >= self.config.min_volume_threshold
        else:
            volume_ok = np.ones(n, dtype=bool)

        # === SESSION FILTER ===
        if self.config.use_session_filter and hasattr(data.index, 'hour'):
            try:
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

        # Count conditions met
        long_condition_count = (
            extended_down.astype(int) +
            rsi_oversold.astype(int) +
            vol_exhaustion_sell.astype(int)
        )

        # Apply multi-condition filter
        long_primary = long_condition_count >= self.config.require_multiple_conditions

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

        # Count conditions met
        short_condition_count = (
            extended_up.astype(int) +
            rsi_overbought.astype(int) +
            vol_exhaustion_buy.astype(int)
        )

        # Apply multi-condition filter
        short_primary = short_condition_count >= self.config.require_multiple_conditions

        # Confirmation: current close < previous close (pullback starting)
        if self.config.require_confirmation:
            reversal_down = np.zeros(n, dtype=bool)
            reversal_down[1:] = close[1:] < close[:-1]
            short_confirmed = short_primary & reversal_down
        else:
            short_confirmed = short_primary

        # === APPLY ALL FILTERS ===
        long_final = long_confirmed & session_ok & volatility_ok & volume_ok
        short_final = short_confirmed & session_ok & volatility_ok & volume_ok

        # Generate signals (LONG takes priority if both true)
        signals = np.where(
            long_final,
            Signal.LONG,
            np.where(short_final, Signal.SHORT, Signal.FLAT),
        )

        return pd.Series(signals, index=data.index, name="signal")

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

        Multi-level exit system:
        1. Stop loss → close 100%
        2. Target 3 → close remaining 100%
        3. Target 2 → close 30%, trail stop
        4. Target 1 → close 50%, move to breakeven
        5. Breakeven trigger → update stop only
        6. Trailing stop updates

        Args:
            position_direction: LONG or SHORT
            entry_price: Position entry price
            entry_time: Position entry time
            current_price: Current close price
            current_high: Current bar high
            current_low: Current bar low
            current_time: Current timestamp
            current_atr: Current ATR value (unused, kept for interface compatibility)
            position_state: Current exit state ("INITIAL", "BREAKEVEN", "TARGET_1", "TARGET_2")
            current_stop: Current stop loss price
            tick_size: Minimum price increment

        Returns:
            Dictionary with exit action or None if no exit:
            {
                "action": "close_full" | "close_partial" | "update_stop",
                "price": exit_price,
                "percentage": percentage to close (for partial),
                "exit_type": type of exit,
                "new_stop": new stop price (for update_stop),
                "new_state": new position state,
            }
        """
        # Convert cents to dollars
        target_1 = self.config.target_1_cents / 100
        target_2 = self.config.target_2_cents / 100
        target_3 = self.config.target_3_cents / 100
        breakeven_trigger = self.config.breakeven_trigger_cents / 100
        trail_offset = self.config.trail_offset_cents / 100

        if position_direction == Signal.LONG:
            profit = current_price - entry_price

            # 1. Check stop loss hit (use low for LONG)
            if current_low <= current_stop:
                return {
                    "action": "close_full",
                    "price": current_stop,
                    "exit_type": "stop_loss",
                }

            # 2. Check target 3 (final exit)
            if profit >= target_3:
                return {
                    "action": "close_full",
                    "price": current_price,
                    "exit_type": "target_3",
                }

            # 3. Check target 2
            if profit >= target_2 and position_state not in ["TARGET_2", "TARGET_3"]:
                # Calculate trailing stop
                new_stop = current_price - trail_offset
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_2_pct,
                    "exit_type": "target_2",
                    "new_stop": max(new_stop, entry_price + tick_size),  # Never below breakeven
                    "new_state": "TARGET_2",
                }

            # 4. Check target 1
            if profit >= target_1 and position_state not in ["TARGET_1", "TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_1_pct,
                    "exit_type": "target_1",
                    "new_stop": entry_price + tick_size,  # Move to breakeven
                    "new_state": "TARGET_1",
                }

            # 5. Check breakeven trigger
            if profit >= breakeven_trigger and position_state == "INITIAL":
                return {
                    "action": "update_stop",
                    "new_stop": entry_price + tick_size,
                    "new_state": "BREAKEVEN",
                }

            # 6. Update trailing stop if in TARGET_1 or TARGET_2 state
            if self.config.trail_after_target_1 and position_state in ["TARGET_1", "TARGET_2"]:
                trail_stop = current_price - trail_offset
                if trail_stop > current_stop:
                    return {
                        "action": "update_stop",
                        "new_stop": trail_stop,
                        "new_state": position_state,
                    }

        elif position_direction == Signal.SHORT:
            profit = entry_price - current_price

            # 1. Check stop loss hit (use high for SHORT)
            if current_high >= current_stop:
                return {
                    "action": "close_full",
                    "price": current_stop,
                    "exit_type": "stop_loss",
                }

            # 2. Check target 3 (final exit)
            if profit >= target_3:
                return {
                    "action": "close_full",
                    "price": current_price,
                    "exit_type": "target_3",
                }

            # 3. Check target 2
            if profit >= target_2 and position_state not in ["TARGET_2", "TARGET_3"]:
                # Calculate trailing stop
                new_stop = current_price + trail_offset
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_2_pct,
                    "exit_type": "target_2",
                    "new_stop": min(new_stop, entry_price - tick_size),  # Never above breakeven
                    "new_state": "TARGET_2",
                }

            # 4. Check target 1
            if profit >= target_1 and position_state not in ["TARGET_1", "TARGET_2", "TARGET_3"]:
                return {
                    "action": "close_partial",
                    "price": current_price,
                    "percentage": self.config.target_1_pct,
                    "exit_type": "target_1",
                    "new_stop": entry_price - tick_size,  # Move to breakeven
                    "new_state": "TARGET_1",
                }

            # 5. Check breakeven trigger
            if profit >= breakeven_trigger and position_state == "INITIAL":
                return {
                    "action": "update_stop",
                    "new_stop": entry_price - tick_size,
                    "new_state": "BREAKEVEN",
                }

            # 6. Update trailing stop if in TARGET_1 or TARGET_2 state
            if self.config.trail_after_target_1 and position_state in ["TARGET_1", "TARGET_2"]:
                trail_stop = current_price + trail_offset
                if trail_stop < current_stop:
                    return {
                        "action": "update_stop",
                        "new_stop": trail_stop,
                        "new_state": position_state,
                    }

        return None

    def get_exit_targets(
        self, entry_price: float, position: Signal
    ) -> dict:
        """Get all exit targets for multi-level exit management.

        Args:
            entry_price: Position entry price
            position: Position direction

        Returns:
            Dictionary with all target and stop levels
        """
        t1 = self.config.target_1_cents / 100
        t2 = self.config.target_2_cents / 100
        t3 = self.config.target_3_cents / 100
        sl = self.config.stop_loss_cents / 100
        be = self.config.breakeven_trigger_cents / 100

        if position == Signal.LONG:
            return {
                "stop_loss": entry_price - sl,
                "breakeven_trigger": entry_price + be,
                "target_1": entry_price + t1,
                "target_2": entry_price + t2,
                "target_3": entry_price + t3,
            }
        elif position == Signal.SHORT:
            return {
                "stop_loss": entry_price + sl,
                "breakeven_trigger": entry_price - be,
                "target_1": entry_price - t1,
                "target_2": entry_price - t2,
                "target_3": entry_price - t3,
            }
        return {}

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
        stop_cents = self.config.stop_loss_cents / 100  # Convert to dollars

        if position == Signal.LONG:
            return entry_price - stop_cents
        elif position == Signal.SHORT:
            return entry_price + stop_cents

        return None

    def get_take_profit(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate take-profit price (first target for compatibility).

        Args:
            data: OHLCV DataFrame
            position: Position direction
            entry_price: Entry price

        Returns:
            Take-profit price (target 1)
        """
        target_cents = self.config.target_1_cents / 100  # Use target 1 for simple mode

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
            "targets_cents": f"{self.config.target_1_cents}/{self.config.target_2_cents}/{self.config.target_3_cents}",
            "target_pcts": f"{self.config.target_1_pct}/{self.config.target_2_pct}/{self.config.target_3_pct}",
            "stop_loss_cents": self.config.stop_loss_cents,
            "breakeven_trigger": self.config.breakeven_trigger_cents,
            "trail_offset": self.config.trail_offset_cents,
            "max_hold_bars": self.config.max_hold_bars,
            "session_hours": f"{self.config.session_start_hour}-{self.config.session_end_hour} UTC",
            "require_confirmation": self.config.require_confirmation,
            "volatility_filter": f"{self.config.min_volatility_percentile}-{self.config.max_volatility_percentile}%" if self.config.use_volatility_filter else "disabled",
            "require_conditions": self.config.require_multiple_conditions,
        }

    @property
    def current_atr(self) -> float:
        """Get most recent ATR value."""
        if self._atr_values is not None and len(self._atr_values) > 0:
            val = self._atr_values[-1]
            if not np.isnan(val):
                return val
        return 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MeanReversionStrategy("
            f"EMA{self.config.ema_period}@{self.config.ema_deviation_threshold}%, "
            f"RSI{self.config.rsi_period}@{self.config.rsi_oversold}/{self.config.rsi_overbought}, "
            f"Targets={self.config.target_1_cents}/{self.config.target_2_cents}/{self.config.target_3_cents}c, "
            f"Stop={self.config.stop_loss_cents}c)"
        )


# ============ Configuration Factory Functions ============


def create_conservative_config() -> MeanReversionConfig:
    """More conservative configuration with tighter filters."""
    return MeanReversionConfig(
        ema_deviation_threshold=0.4,  # Require larger deviation
        rsi_oversold=30.0,  # Classic levels
        rsi_overbought=70.0,
        volume_spike_threshold=2.5,  # Higher volume requirement
        stop_loss_cents=8.0,  # Tighter stop
        target_1_cents=4.0,
        target_2_cents=7.0,
        target_3_cents=10.0,
        require_confirmation=True,
        require_multiple_conditions=2,  # Require 2 of 3 conditions
    )


def create_aggressive_config() -> MeanReversionConfig:
    """More aggressive configuration with looser filters."""
    return MeanReversionConfig(
        ema_deviation_threshold=0.2,  # Smaller deviation triggers
        rsi_oversold=40.0,  # More permissive
        rsi_overbought=60.0,
        volume_spike_threshold=1.5,  # Lower volume threshold
        stop_loss_cents=12.0,  # Wider stop
        target_1_cents=6.0,
        target_2_cents=12.0,
        target_3_cents=18.0,
        require_confirmation=False,
        use_volatility_filter=False,  # No volatility filter
        require_multiple_conditions=1,  # Any condition
    )


def create_optimized_config() -> MeanReversionConfig:
    """Optimized configuration based on backtest research.

    Settings derived from:
    - 11 tick stop / 10 tick trailing works for crude oil
    - Scale-out 50%/30%/20% at targets
    - Volatility filter 20-80 percentile
    """
    return MeanReversionConfig(
        # Entry parameters
        ema_deviation_threshold=0.3,
        rsi_oversold=35.0,
        rsi_overbought=65.0,
        volume_spike_threshold=2.0,

        # Multi-level targets
        stop_loss_cents=11.0,
        target_1_cents=5.0,
        target_2_cents=10.0,
        target_3_cents=15.0,
        target_1_pct=50.0,
        target_2_pct=30.0,
        target_3_pct=20.0,

        # Breakeven and trailing
        breakeven_trigger_cents=4.0,
        trail_after_target_1=True,
        trail_offset_cents=3.0,

        # Filters
        use_volatility_filter=True,
        min_volatility_percentile=20.0,
        max_volatility_percentile=80.0,
        require_multiple_conditions=1,  # Start with any condition
        require_confirmation=False,

        # Session
        use_session_filter=True,
        session_start_hour=14,
        session_end_hour=20,

        # Time management
        max_hold_bars=20,
    )
