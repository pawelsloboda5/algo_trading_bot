"""Portfolio tracking for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class TradeDirection(str, Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_time: datetime
    exit_time: datetime
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: int
    commission: float
    pnl: float  # Net P&L after commission

    @property
    def gross_pnl(self) -> float:
        """P&L before commission."""
        return self.pnl + self.commission

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry value."""
        entry_value = self.entry_price * self.quantity
        if entry_value == 0:
            return 0.0
        return (self.pnl / entry_value) * 100

    @property
    def duration(self) -> pd.Timedelta:
        """Trade duration."""
        return pd.Timestamp(self.exit_time) - pd.Timestamp(self.entry_time)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0


@dataclass
class PartialExit:
    """Record of a partial position exit."""

    timestamp: datetime
    exit_price: float
    quantity_closed: int
    pnl: float
    commission: float
    exit_type: str  # "target_1", "target_2", "target_3", "stop", "time_stop", etc.


@dataclass
class Position:
    """Current open position."""

    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    quantity: int
    stop_loss: float | None = None
    take_profit: float | None = None

    # Multi-level exit tracking
    initial_quantity: int = field(default=0, init=False)
    partial_exits: list[PartialExit] = field(default_factory=list, init=False)
    state: str = field(default="INITIAL", init=False)  # INITIAL, BREAKEVEN, TARGET_1, TARGET_2

    def __post_init__(self) -> None:
        """Store initial quantity for tracking."""
        self.initial_quantity = self.quantity

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from partial exits."""
        return sum(pe.pnl for pe in self.partial_exits)

    @property
    def total_commission(self) -> float:
        """Total commission from partial exits."""
        return sum(pe.commission for pe in self.partial_exits)

    def unrealized_pnl(self, current_price: float, multiplier: float = 1.0) -> float:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price
            multiplier: Contract multiplier

        Returns:
            Unrealized P&L in dollars
        """
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) * self.quantity * multiplier
        else:
            return (self.entry_price - current_price) * self.quantity * multiplier


@dataclass
class Portfolio:
    """Portfolio state tracker for backtesting.

    Tracks cash, positions, equity curve, and trade history.
    """

    initial_capital: float
    contract_multiplier: float = 100.0  # MCL = 100 barrels
    commission_per_contract: float = 2.25  # IB futures commission

    # State
    cash: float = field(init=False)
    position: Position | None = field(default=None, init=False)
    trades: list[Trade] = field(default_factory=list, init=False)

    # Equity tracking
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize cash to starting capital."""
        self.cash = self.initial_capital

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + unrealized P&L)."""
        if self.position is None:
            return self.cash
        # Note: Need current price to calculate unrealized P&L
        # This property returns cash only; use get_equity(price) for full equity
        return self.cash

    def get_equity(self, current_price: float) -> float:
        """Get total equity including unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Total equity
        """
        if self.position is None:
            return self.cash

        unrealized = self.position.unrealized_pnl(current_price, self.contract_multiplier)
        return self.cash + unrealized

    def open_position(
        self,
        direction: TradeDirection,
        price: float,
        quantity: int,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Open a new position.

        Args:
            direction: LONG or SHORT
            price: Entry price
            quantity: Number of contracts
            timestamp: Entry timestamp
            stop_loss: Stop-loss price
            take_profit: Take-profit price
        """
        if self.position is not None:
            raise ValueError("Cannot open position while another is open. Close first.")

        # Deduct commission
        commission = self.commission_per_contract * quantity
        self.cash -= commission

        self.position = Position(
            direction=direction,
            entry_time=timestamp,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def close_position(self, price: float, timestamp: datetime) -> Trade | None:
        """Close current position (full close).

        Args:
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            Completed Trade object, or None if no position
        """
        if self.position is None:
            return None

        # Calculate P&L for remaining quantity
        if self.position.direction == TradeDirection.LONG:
            gross_pnl = (price - self.position.entry_price) * self.position.quantity * self.contract_multiplier
        else:
            gross_pnl = (self.position.entry_price - price) * self.position.quantity * self.contract_multiplier

        # Commission for exit
        exit_commission = self.commission_per_contract * self.position.quantity
        net_pnl = gross_pnl - exit_commission

        # Include any partial exit P&L
        total_pnl = net_pnl + self.position.realized_pnl
        total_commission = exit_commission + self.position.total_commission + (
            self.commission_per_contract * self.position.initial_quantity  # Entry commission
        )

        # Create trade record
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            exit_price=price,
            quantity=self.position.initial_quantity,  # Use initial quantity for full trade
            commission=total_commission,
            pnl=total_pnl,
        )

        # Update cash
        self.cash += gross_pnl - exit_commission

        # Record trade and clear position
        self.trades.append(trade)
        self.position = None

        return trade

    def close_partial(
        self,
        price: float,
        timestamp: datetime,
        quantity: int | None = None,
        percentage: float | None = None,
        exit_type: str = "partial",
    ) -> PartialExit | None:
        """Close a portion of the current position.

        Either quantity OR percentage must be specified (not both).

        Args:
            price: Exit price
            timestamp: Exit timestamp
            quantity: Number of contracts to close
            percentage: Percentage of current position to close (0-100)
            exit_type: Type of exit ("target_1", "target_2", "stop", etc.)

        Returns:
            PartialExit record, or None if no position
        """
        if self.position is None:
            return None

        if quantity is None and percentage is None:
            raise ValueError("Either quantity or percentage must be specified")

        if quantity is not None and percentage is not None:
            raise ValueError("Cannot specify both quantity and percentage")

        # Calculate quantity to close
        if percentage is not None:
            quantity = max(1, int(self.position.quantity * percentage / 100))

        # Can't close more than we have
        quantity = min(quantity, self.position.quantity)

        if quantity <= 0:
            return None

        # Calculate P&L for this partial exit
        if self.position.direction == TradeDirection.LONG:
            gross_pnl = (price - self.position.entry_price) * quantity * self.contract_multiplier
        else:
            gross_pnl = (self.position.entry_price - price) * quantity * self.contract_multiplier

        # Commission for this exit
        exit_commission = self.commission_per_contract * quantity
        net_pnl = gross_pnl - exit_commission

        # Create partial exit record
        partial_exit = PartialExit(
            timestamp=timestamp,
            exit_price=price,
            quantity_closed=quantity,
            pnl=net_pnl,
            commission=exit_commission,
            exit_type=exit_type,
        )

        # Update position
        self.position.quantity -= quantity
        self.position.partial_exits.append(partial_exit)

        # Update cash
        self.cash += gross_pnl - exit_commission

        # If position fully closed, finalize trade
        if self.position.quantity <= 0:
            trade = Trade(
                entry_time=self.position.entry_time,
                exit_time=timestamp,
                direction=self.position.direction,
                entry_price=self.position.entry_price,
                exit_price=price,
                quantity=self.position.initial_quantity,
                commission=self.position.total_commission + (
                    self.commission_per_contract * self.position.initial_quantity  # Entry commission
                ),
                pnl=self.position.realized_pnl,
            )
            self.trades.append(trade)
            self.position = None

        return partial_exit

    def update_stop_loss(self, new_stop: float) -> None:
        """Update stop loss for current position.

        Args:
            new_stop: New stop loss price
        """
        if self.position is not None:
            self.position.stop_loss = new_stop

    def update_position_state(self, new_state: str) -> None:
        """Update position state for multi-level exit tracking.

        Args:
            new_state: New state ("BREAKEVEN", "TARGET_1", "TARGET_2", etc.)
        """
        if self.position is not None:
            self.position.state = new_state

    def record_equity(self, timestamp: datetime, current_price: float) -> None:
        """Record equity at a point in time.

        Args:
            timestamp: Current timestamp
            current_price: Current market price
        """
        equity = self.get_equity(current_price)
        self.equity_curve.append((timestamp, equity))

    def get_equity_series(self) -> pd.Series:
        """Get equity curve as pandas Series.

        Returns:
            Series with datetime index and equity values
        """
        if not self.equity_curve:
            return pd.Series(dtype=float)

        times, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "commission": t.commission,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "duration": t.duration,
                    "is_winner": t.is_winner,
                }
            )

        return pd.DataFrame(records)

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
