"""Base strategy interface for trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class Signal(IntEnum):
    """Trading signal values."""

    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass
class TradeSignal:
    """Represents a trading signal at a specific point in time."""

    timestamp: pd.Timestamp
    signal: Signal
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = 1.0  # Signal strength 0-1


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement:
    - generate_signals(): Produce buy/sell signals from price data
    - get_parameters(): Return current strategy parameters
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume
                  Index should be datetime

        Returns:
            Series of Signal values (+1 long, -1 short, 0 flat)
            Same index as input data
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current strategy parameters.

        Returns:
            Dictionary of parameter names and values
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns.

        Args:
            data: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {"open", "high", "low", "close"}
        missing = required_columns - set(data.columns.str.lower())
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_stop_loss(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate stop-loss price for a position.

        Override in subclass to implement custom stop-loss logic.

        Args:
            data: OHLCV DataFrame
            position: Current position direction
            entry_price: Entry price for the position

        Returns:
            Stop-loss price or None if not applicable
        """
        return None

    def get_take_profit(
        self, data: pd.DataFrame, position: Signal, entry_price: float
    ) -> float | None:
        """Calculate take-profit price for a position.

        Override in subclass to implement custom take-profit logic.

        Args:
            data: OHLCV DataFrame
            position: Current position direction
            entry_price: Entry price for the position

        Returns:
            Take-profit price or None if not applicable
        """
        return None

    @property
    def name(self) -> str:
        """Strategy name (class name by default)."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """String representation with parameters."""
        params = self.get_parameters()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.name}({params_str})"
