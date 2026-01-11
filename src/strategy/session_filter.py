"""Session and time-based filters for trading strategies.

Filters trading to profitable hours based on historical analysis.
MCL (Micro WTI) is most liquid during US trading hours.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class TradingSession:
    """Defines a trading session with start and end hours (UTC)."""

    name: str
    start_hour: int  # 0-23
    start_minute: int  # 0-59
    end_hour: int  # 0-23
    end_minute: int  # 0-59

    def contains_time(self, hour: int, minute: int) -> bool:
        """Check if a time falls within this session."""
        time_decimal = hour + minute / 60
        start_decimal = self.start_hour + self.start_minute / 60
        end_decimal = self.end_hour + self.end_minute / 60

        if start_decimal < end_decimal:
            # Same day range (e.g., 14:00-20:00)
            return start_decimal <= time_decimal < end_decimal
        else:
            # Overnight range (e.g., 22:00-06:00)
            return time_decimal >= start_decimal or time_decimal < end_decimal


# Predefined trading sessions for MCL (all times UTC)
US_MAIN_SESSION = TradingSession(
    name="US Main",
    start_hour=14,
    start_minute=0,
    end_hour=20,
    end_minute=0,
)

US_EXTENDED_SESSION = TradingSession(
    name="US Extended",
    start_hour=13,
    start_minute=0,
    end_hour=21,
    end_minute=0,
)

EUROPEAN_OVERLAP_SESSION = TradingSession(
    name="European Overlap",
    start_hour=12,
    start_minute=0,
    end_hour=16,
    end_minute=0,
)

# Default: US main session (most profitable for MCL)
DEFAULT_SESSIONS = [US_MAIN_SESSION]


class SessionFilter:
    """Filter trading signals based on time of day and day of week.

    Based on historical analysis showing that:
    - Hours 0-5 UTC are unprofitable (75% of losses)
    - US session (14:00-20:00 UTC) is most profitable
    - Mondays have poor performance
    - Sundays and late Friday should be avoided
    """

    def __init__(
        self,
        sessions: list[TradingSession] | None = None,
        exclude_days: list[int] | None = None,
        friday_cutoff_hour: int = 19,
    ):
        """Initialize session filter.

        Args:
            sessions: List of allowed trading sessions (default: US main)
            exclude_days: Days of week to exclude (0=Monday, 6=Sunday)
                         Default: [6] (Sunday)
            friday_cutoff_hour: Hour UTC to stop trading on Friday (default: 19)
        """
        self.sessions = sessions or DEFAULT_SESSIONS
        self.exclude_days = exclude_days if exclude_days is not None else [6]
        self.friday_cutoff_hour = friday_cutoff_hour

    def is_trading_allowed(
        self, timestamp: pd.Timestamp
    ) -> bool:
        """Check if trading is allowed at given timestamp.

        Args:
            timestamp: Datetime to check

        Returns:
            True if trading is allowed
        """
        # Check day of week exclusions
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday
        if day_of_week in self.exclude_days:
            return False

        # Check Friday cutoff
        if day_of_week == 4:  # Friday
            if timestamp.hour >= self.friday_cutoff_hour:
                return False

        # Check if within any allowed session
        hour = timestamp.hour
        minute = timestamp.minute

        for session in self.sessions:
            if session.contains_time(hour, minute):
                return True

        return False

    def filter_signals(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Create boolean mask for allowed trading times.

        Args:
            timestamps: Index of timestamps to filter

        Returns:
            Boolean array, True where trading is allowed
        """
        n = len(timestamps)
        result = np.zeros(n, dtype=bool)

        for i in range(n):
            result[i] = self.is_trading_allowed(timestamps[i])

        return result

    def filter_signals_vectorized(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Vectorized filtering for better performance on large datasets.

        Args:
            timestamps: Index of timestamps to filter

        Returns:
            Boolean array, True where trading is allowed
        """
        # Extract time components
        hours = timestamps.hour
        minutes = timestamps.minute
        day_of_week = timestamps.dayofweek

        # Start with all True
        allowed = np.ones(len(timestamps), dtype=bool)

        # Apply day exclusions
        for day in self.exclude_days:
            allowed &= (day_of_week != day)

        # Apply Friday cutoff
        friday_mask = (day_of_week == 4) & (hours >= self.friday_cutoff_hour)
        allowed &= ~friday_mask

        # Apply session filters
        time_decimal = hours + minutes / 60
        session_allowed = np.zeros(len(timestamps), dtype=bool)

        for session in self.sessions:
            start = session.start_hour + session.start_minute / 60
            end = session.end_hour + session.end_minute / 60

            if start < end:
                # Same day range
                mask = (time_decimal >= start) & (time_decimal < end)
            else:
                # Overnight range
                mask = (time_decimal >= start) | (time_decimal < end)

            session_allowed |= mask

        allowed &= session_allowed

        return allowed

    def get_session_info(self) -> dict:
        """Get human-readable session information."""
        return {
            "sessions": [
                {
                    "name": s.name,
                    "start": f"{s.start_hour:02d}:{s.start_minute:02d} UTC",
                    "end": f"{s.end_hour:02d}:{s.end_minute:02d} UTC",
                }
                for s in self.sessions
            ],
            "excluded_days": [
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d]
                for d in self.exclude_days
            ],
            "friday_cutoff": f"{self.friday_cutoff_hour:02d}:00 UTC",
        }


def create_conservative_filter() -> SessionFilter:
    """Create a conservative session filter for scalping.

    Only trades during peak US session, avoids problematic days/times.
    """
    return SessionFilter(
        sessions=[US_MAIN_SESSION],
        exclude_days=[0, 6],  # Exclude Monday and Sunday
        friday_cutoff_hour=18,  # Earlier Friday cutoff
    )


def create_standard_filter() -> SessionFilter:
    """Create standard session filter.

    Trades US session, avoids Sunday.
    """
    return SessionFilter(
        sessions=[US_MAIN_SESSION],
        exclude_days=[6],  # Exclude Sunday only
        friday_cutoff_hour=19,
    )


def create_extended_filter() -> SessionFilter:
    """Create extended session filter.

    Trades extended US hours including European overlap.
    """
    return SessionFilter(
        sessions=[EUROPEAN_OVERLAP_SESSION, US_EXTENDED_SESSION],
        exclude_days=[6],  # Exclude Sunday only
        friday_cutoff_hour=20,
    )


def analyze_session_performance(
    trades_df: pd.DataFrame,
    pnl_column: str = "pnl",
    time_column: str = "entry_time",
) -> pd.DataFrame:
    """Analyze trading performance by hour and day of week.

    Useful for optimizing session filters based on actual results.

    Args:
        trades_df: DataFrame with trade results
        pnl_column: Column name for P&L
        time_column: Column name for entry time

    Returns:
        DataFrame with performance by hour and day
    """
    if trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df["hour"] = pd.to_datetime(df[time_column]).dt.hour
    df["day_of_week"] = pd.to_datetime(df[time_column]).dt.dayofweek
    df["day_name"] = pd.to_datetime(df[time_column]).dt.day_name()

    # Aggregate by hour
    hourly = df.groupby("hour").agg(
        total_pnl=(pnl_column, "sum"),
        trade_count=(pnl_column, "count"),
        win_rate=(pnl_column, lambda x: (x > 0).mean() * 100),
        avg_pnl=(pnl_column, "mean"),
    ).round(2)

    # Aggregate by day
    daily = df.groupby(["day_of_week", "day_name"]).agg(
        total_pnl=(pnl_column, "sum"),
        trade_count=(pnl_column, "count"),
        win_rate=(pnl_column, lambda x: (x > 0).mean() * 100),
        avg_pnl=(pnl_column, "mean"),
    ).round(2)

    return {"hourly": hourly, "daily": daily}
