"""Stop-loss strategies for risk management."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from config.logging_config import get_logger
from src.backtest.portfolio import TradeDirection

logger = get_logger(__name__)


class StopLossType(str, Enum):
    """Stop-loss strategy type."""

    FIXED_PERCENT = "fixed_percent"
    FIXED_DOLLAR = "fixed_dollar"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"


@dataclass
class StopLossResult:
    """Result of stop-loss calculation."""

    price: float
    type: StopLossType
    distance: float  # Price distance from entry


class StopLossManager:
    """Stop-loss calculation strategies for MCL futures trading.

    Provides multiple stop-loss methods:
    - Fixed percentage: Stop at X% from entry
    - Fixed dollar: Stop at $X distance (per contract)
    - ATR-based: Stop at X * ATR from entry
    - Trailing: Dynamic stop that follows price
    """

    def __init__(
        self,
        contract_multiplier: float = 100.0,
        tick_size: float = 0.01,
        default_atr_multiplier: float = 2.0,
    ):
        """Initialize stop-loss manager.

        Args:
            contract_multiplier: Contract multiplier (MCL = 100 barrels)
            tick_size: Minimum price increment (MCL = $0.01)
            default_atr_multiplier: Default ATR multiplier for stops
        """
        if contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if default_atr_multiplier <= 0:
            raise ValueError("default_atr_multiplier must be positive")

        self.contract_multiplier = contract_multiplier
        self.tick_size = tick_size
        self.default_atr_multiplier = default_atr_multiplier

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size."""
        return round(price / self.tick_size) * self.tick_size

    def fixed_percentage(
        self,
        entry_price: float,
        percentage: float,
        direction: TradeDirection,
    ) -> StopLossResult:
        """Calculate stop-loss at fixed percentage from entry.

        Example: 2% stop on $75.50 entry
        - LONG: stop at $75.50 * (1 - 0.02) = $73.99
        - SHORT: stop at $75.50 * (1 + 0.02) = $77.01

        Args:
            entry_price: Trade entry price
            percentage: Stop percentage (e.g., 0.02 for 2%)
            direction: Trade direction (LONG or SHORT)

        Returns:
            StopLossResult with calculated stop price
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if percentage <= 0 or percentage >= 1:
            raise ValueError("percentage must be between 0 and 1")

        distance = entry_price * percentage

        if direction == TradeDirection.LONG:
            stop_price = entry_price - distance
        else:
            stop_price = entry_price + distance

        stop_price = self._round_to_tick(stop_price)

        logger.debug(
            "fixed_percentage_stop",
            entry_price=entry_price,
            percentage=percentage,
            direction=direction.value,
            stop_price=stop_price,
        )

        return StopLossResult(
            price=stop_price,
            type=StopLossType.FIXED_PERCENT,
            distance=abs(entry_price - stop_price),
        )

    def fixed_dollar(
        self,
        entry_price: float,
        dollar_risk: float,
        direction: TradeDirection,
    ) -> StopLossResult:
        """Calculate stop-loss at fixed dollar amount per contract.

        Example: $20 risk on MCL (100 multiplier)
        - Price distance: $20 / 100 = $0.20
        - LONG at $75.50: stop at $75.30
        - SHORT at $75.50: stop at $75.70

        Args:
            entry_price: Trade entry price
            dollar_risk: Dollar risk per contract
            direction: Trade direction (LONG or SHORT)

        Returns:
            StopLossResult with calculated stop price
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if dollar_risk <= 0:
            raise ValueError("dollar_risk must be positive")

        price_distance = dollar_risk / self.contract_multiplier

        if direction == TradeDirection.LONG:
            stop_price = entry_price - price_distance
        else:
            stop_price = entry_price + price_distance

        stop_price = self._round_to_tick(stop_price)

        logger.debug(
            "fixed_dollar_stop",
            entry_price=entry_price,
            dollar_risk=dollar_risk,
            price_distance=price_distance,
            direction=direction.value,
            stop_price=stop_price,
        )

        return StopLossResult(
            price=stop_price,
            type=StopLossType.FIXED_DOLLAR,
            distance=abs(entry_price - stop_price),
        )

    def atr_based(
        self,
        entry_price: float,
        current_atr: float,
        multiplier: float | None = None,
        direction: TradeDirection = TradeDirection.LONG,
    ) -> StopLossResult:
        """Calculate ATR-based stop-loss.

        Formula: stop_distance = ATR * multiplier

        Example: ATR = $0.30, multiplier = 2.0
        - Stop distance: $0.60
        - LONG at $75.50: stop at $74.90
        - SHORT at $75.50: stop at $76.10

        Args:
            entry_price: Trade entry price
            current_atr: Current Average True Range value
            multiplier: ATR multiplier (uses default if None)
            direction: Trade direction (LONG or SHORT)

        Returns:
            StopLossResult with calculated stop price
        """
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if current_atr <= 0:
            raise ValueError("current_atr must be positive")

        mult = multiplier if multiplier is not None else self.default_atr_multiplier
        if mult <= 0:
            raise ValueError("multiplier must be positive")

        stop_distance = current_atr * mult

        if direction == TradeDirection.LONG:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        stop_price = self._round_to_tick(stop_price)

        logger.debug(
            "atr_based_stop",
            entry_price=entry_price,
            current_atr=current_atr,
            multiplier=mult,
            stop_distance=stop_distance,
            direction=direction.value,
            stop_price=stop_price,
        )

        return StopLossResult(
            price=stop_price,
            type=StopLossType.ATR_BASED,
            distance=abs(entry_price - stop_price),
        )

    def trailing_stop_update(
        self,
        current_stop: float,
        high_since_entry: float,
        low_since_entry: float,
        atr: float,
        multiplier: float | None = None,
        direction: TradeDirection = TradeDirection.LONG,
    ) -> StopLossResult:
        """Update trailing stop based on price movement.

        For LONG positions:
        - Trail below the highest high since entry
        - new_stop = high_since_entry - (ATR * multiplier)
        - Only move stop up, never down

        For SHORT positions:
        - Trail above the lowest low since entry
        - new_stop = low_since_entry + (ATR * multiplier)
        - Only move stop down, never up

        Args:
            current_stop: Current stop-loss price
            high_since_entry: Highest high since position entry
            low_since_entry: Lowest low since position entry
            atr: Current Average True Range
            multiplier: ATR multiplier (uses default if None)
            direction: Trade direction (LONG or SHORT)

        Returns:
            StopLossResult with updated stop price
        """
        if atr <= 0:
            raise ValueError("atr must be positive")

        mult = multiplier if multiplier is not None else self.default_atr_multiplier
        if mult <= 0:
            raise ValueError("multiplier must be positive")

        trail_distance = atr * mult

        if direction == TradeDirection.LONG:
            new_stop = high_since_entry - trail_distance
            new_stop = self._round_to_tick(new_stop)
            # Only move stop up for long positions
            final_stop = max(current_stop, new_stop)
        else:
            new_stop = low_since_entry + trail_distance
            new_stop = self._round_to_tick(new_stop)
            # Only move stop down for short positions
            final_stop = min(current_stop, new_stop)

        logger.debug(
            "trailing_stop_update",
            current_stop=current_stop,
            high_since_entry=high_since_entry,
            low_since_entry=low_since_entry,
            trail_distance=trail_distance,
            direction=direction.value,
            new_stop=new_stop,
            final_stop=final_stop,
        )

        return StopLossResult(
            price=final_stop,
            type=StopLossType.TRAILING,
            distance=trail_distance,
        )

    def time_based_exit(
        self,
        entry_time: datetime,
        current_time: datetime,
        max_hold_bars: int,
        bar_duration_minutes: int = 1,
    ) -> bool:
        """Check if position has exceeded maximum hold time.

        Useful for time-based exits to avoid holding losing positions
        indefinitely.

        Args:
            entry_time: Position entry timestamp
            current_time: Current timestamp
            max_hold_bars: Maximum number of bars to hold
            bar_duration_minutes: Duration of each bar in minutes

        Returns:
            True if max hold time exceeded, False otherwise
        """
        if max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive")
        if bar_duration_minutes <= 0:
            raise ValueError("bar_duration_minutes must be positive")

        hold_duration = current_time - entry_time
        hold_minutes = hold_duration.total_seconds() / 60
        max_minutes = max_hold_bars * bar_duration_minutes

        exceeded = hold_minutes >= max_minutes

        if exceeded:
            logger.info(
                "time_based_exit_triggered",
                entry_time=entry_time.isoformat(),
                current_time=current_time.isoformat(),
                hold_minutes=hold_minutes,
                max_minutes=max_minutes,
            )

        return exceeded
