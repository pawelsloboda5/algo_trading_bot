"""Data schemas for market data."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    """Trade side."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class OHLCVBar:
    """OHLCV bar data structure.

    Represents a single candlestick/bar with price and volume information.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        """Validate bar data."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) cannot be less than low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be <= open and close")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")

    @property
    def range(self) -> float:
        """Price range of the bar."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Body size (absolute difference between open and close)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if bar closed higher than open."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if bar closed lower than open."""
        return self.close < self.open

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3


@dataclass(frozen=True, slots=True)
class TradeData:
    """Individual trade/tick data structure."""

    timestamp: datetime
    symbol: str
    price: float
    size: int
    side: Side | None = None

    def __post_init__(self) -> None:
        """Validate trade data."""
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Futures contract specification."""

    symbol: str
    exchange: str
    multiplier: float
    tick_size: float
    currency: str = "USD"
    trading_hours: str = ""

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick movement."""
        return self.tick_size * self.multiplier


@dataclass(frozen=True, slots=True)
class MBPLevel:
    """Market by Price level (order book level)."""

    price: float
    size: int
    count: int  # Number of orders at this level


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """L2 Order book snapshot (MBP-10 schema).

    Contains up to 10 levels of market depth on each side.
    """

    timestamp: datetime
    symbol: str
    bids: tuple[MBPLevel, ...]  # Best bid first
    asks: tuple[MBPLevel, ...]  # Best ask first

    @property
    def best_bid(self) -> float | None:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> float | None:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass(frozen=True, slots=True)
class BBO:
    """Best Bid/Offer (MBP-1 schema, top of book)."""

    timestamp: datetime
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.bid_price + self.ask_price) / 2


# Pre-defined contract specs for common futures
MCL_CONTRACT = ContractSpec(
    symbol="MCL",
    exchange="NYMEX",
    multiplier=100.0,  # 100 barrels (1/10th of CL)
    tick_size=0.01,  # $0.01 = $1 per contract
    currency="USD",
    trading_hours="18:00-17:00 ET (Sun-Fri)",
)

CL_CONTRACT = ContractSpec(
    symbol="CL",
    exchange="NYMEX",
    multiplier=1000.0,  # 1000 barrels
    tick_size=0.01,  # $0.01 = $10 per contract
    currency="USD",
    trading_hours="18:00-17:00 ET (Sun-Fri)",
)
