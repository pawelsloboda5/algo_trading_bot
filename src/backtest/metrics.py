"""Performance metrics for backtesting."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""

    # Returns
    total_return: float
    total_return_pct: float
    cagr: float  # Compound Annual Growth Rate

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    max_drawdown_duration: pd.Timedelta | None

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_trade_duration: pd.Timedelta | None

    # Exposure
    time_in_market: float  # Percentage of time with open position

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "Total Return ($)": f"${self.total_return:,.2f}",
            "Total Return (%)": f"{self.total_return_pct:.2f}%",
            "CAGR": f"{self.cagr:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown ($)": f"${self.max_drawdown:,.2f}",
            "Max Drawdown (%)": f"{self.max_drawdown_pct:.2f}%",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.2f}%",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Avg Trade P&L": f"${self.avg_trade_pnl:,.2f}",
            "Avg Winner": f"${self.avg_winner:,.2f}",
            "Avg Loser": f"${self.avg_loser:,.2f}",
            "Largest Winner": f"${self.largest_winner:,.2f}",
            "Largest Loser": f"${self.largest_loser:,.2f}",
            "Time in Market": f"{self.time_in_market:.1f}%",
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        lines = ["=" * 50, "BACKTEST PERFORMANCE REPORT", "=" * 50, ""]

        for key, value in self.to_dict().items():
            lines.append(f"{key:.<30} {value}")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)


def calculate_returns(equity_series: pd.Series) -> pd.Series:
    """Calculate period returns from equity curve.

    Args:
        equity_series: Equity values over time

    Returns:
        Series of period returns
    """
    return equity_series.pct_change().dropna()


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Trading periods per year (default: 252 for daily)

    Returns:
        Sharpe Ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_period
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino Ratio (uses downside deviation).

    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino Ratio
    """
    if len(returns) == 0:
        return 0.0

    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_period

    # Downside deviation (only negative returns)
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside**2))
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(equity_series: pd.Series) -> tuple[float, float, pd.Timedelta | None]:
    """Calculate maximum drawdown.

    Args:
        equity_series: Equity values over time

    Returns:
        Tuple of (max drawdown $, max drawdown %, max drawdown duration)
    """
    if len(equity_series) == 0:
        return 0.0, 0.0, None

    # Calculate running maximum
    running_max = equity_series.expanding().max()

    # Drawdown in dollars
    drawdown = running_max - equity_series

    # Drawdown in percentage
    drawdown_pct = drawdown / running_max * 100

    max_dd = drawdown.max()
    max_dd_pct = drawdown_pct.max()

    # Calculate drawdown duration
    max_dd_duration = None
    if max_dd > 0:
        # Find periods where we're in drawdown
        in_drawdown = drawdown > 0

        # Find the longest continuous drawdown period
        if in_drawdown.any():
            # Create groups of consecutive drawdown periods
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            drawdown_groups = drawdown_starts.cumsum()
            drawdown_groups[~in_drawdown] = 0

            if drawdown_groups.max() > 0:
                # Find duration of each drawdown period
                for group in range(1, drawdown_groups.max() + 1):
                    mask = drawdown_groups == group
                    if mask.any():
                        start = equity_series.index[mask].min()
                        end = equity_series.index[mask].max()
                        duration = end - start
                        if max_dd_duration is None or duration > max_dd_duration:
                            max_dd_duration = duration

    return max_dd, max_dd_pct, max_dd_duration


def calculate_cagr(
    initial_value: float, final_value: float, years: float
) -> float:
    """Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting value
        final_value: Ending value
        years: Number of years

    Returns:
        CAGR as percentage
    """
    if initial_value <= 0 or years <= 0:
        return 0.0

    return ((final_value / initial_value) ** (1 / years) - 1) * 100


def calculate_calmar_ratio(cagr: float, max_drawdown_pct: float) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown).

    Args:
        cagr: Compound Annual Growth Rate (%)
        max_drawdown_pct: Maximum drawdown (%)

    Returns:
        Calmar Ratio
    """
    if max_drawdown_pct == 0:
        return np.inf if cagr > 0 else 0.0

    return cagr / abs(max_drawdown_pct)


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate Profit Factor (gross profit / gross loss).

    Args:
        trades_df: DataFrame with 'pnl' column

    Returns:
        Profit Factor
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0

    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_metrics(
    equity_series: pd.Series,
    trades_df: pd.DataFrame,
    initial_capital: float,
    signals: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        equity_series: Equity curve over time
        trades_df: DataFrame of completed trades
        initial_capital: Starting capital
        signals: Trading signals series (for time in market calculation)
        risk_free_rate: Annual risk-free rate

    Returns:
        PerformanceMetrics object
    """
    # Basic returns
    final_equity = equity_series.iloc[-1] if len(equity_series) > 0 else initial_capital
    total_return = final_equity - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    # Time period
    if len(equity_series) > 1:
        time_period = equity_series.index[-1] - equity_series.index[0]
        years = time_period.days / 365.25
    else:
        years = 1.0

    # CAGR
    cagr = calculate_cagr(initial_capital, final_equity, years)

    # Returns series
    returns = calculate_returns(equity_series)

    # Determine periods per year based on data frequency
    if len(equity_series) > 1:
        avg_period = (equity_series.index[-1] - equity_series.index[0]) / len(equity_series)
        if avg_period < pd.Timedelta(minutes=5):
            periods_per_year = 252 * 6.5 * 60  # 1-minute bars
        elif avg_period < pd.Timedelta(hours=1):
            periods_per_year = 252 * 6.5 * 12  # 5-minute bars
        elif avg_period < pd.Timedelta(days=1):
            periods_per_year = 252 * 6.5  # Hourly bars
        else:
            periods_per_year = 252  # Daily
    else:
        periods_per_year = 252

    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown
    max_dd, max_dd_pct, max_dd_duration = calculate_max_drawdown(equity_series)
    avg_dd = (equity_series.expanding().max() - equity_series).mean() if len(equity_series) > 0 else 0.0

    # Calmar
    calmar = calculate_calmar_ratio(cagr, max_dd_pct)

    # Trade statistics
    total_trades = len(trades_df)
    if total_trades > 0:
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] < 0]

        winning_trades = len(winners)
        losing_trades = len(losers)
        win_rate = (winning_trades / total_trades) * 100

        profit_factor = calculate_profit_factor(trades_df)

        avg_trade_pnl = trades_df["pnl"].mean()
        avg_winner = winners["pnl"].mean() if len(winners) > 0 else 0.0
        avg_loser = losers["pnl"].mean() if len(losers) > 0 else 0.0
        largest_winner = trades_df["pnl"].max()
        largest_loser = trades_df["pnl"].min()

        # Average trade duration
        if "duration" in trades_df.columns:
            avg_trade_duration = trades_df["duration"].mean()
        else:
            avg_trade_duration = None
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_winner = 0.0
        avg_loser = 0.0
        largest_winner = 0.0
        largest_loser = 0.0
        avg_trade_duration = None

    # Time in market
    if signals is not None and len(signals) > 0:
        time_in_market = (signals != 0).mean() * 100
    else:
        time_in_market = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_dd_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        largest_winner=largest_winner,
        largest_loser=largest_loser,
        avg_trade_duration=avg_trade_duration,
        time_in_market=time_in_market,
    )
