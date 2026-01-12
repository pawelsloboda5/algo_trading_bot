"""Plotly chart generators for backtest visualization."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.logging_config import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """Generator for interactive backtest charts."""

    # Color scheme
    COLORS = {
        "equity": "#2E86AB",
        "drawdown": "#E94F37",
        "win": "#2ECC71",
        "loss": "#E74C3C",
        "neutral": "#95A5A6",
        "background": "#FFFFFF",
        "grid": "#E5E5E5",
        "text": "#2C3E50",
        "long_entry": "#27AE60",
        "long_exit": "#1ABC9C",
        "short_entry": "#E74C3C",
        "short_exit": "#C0392B",
    }

    # Chart template
    TEMPLATE = "plotly_white"

    def __init__(self, theme: str = "light"):
        """Initialize chart generator.

        Args:
            theme: Color theme ("light" or "dark")
        """
        self.theme = theme
        if theme == "dark":
            self.TEMPLATE = "plotly_dark"
            self.COLORS["background"] = "#1E1E1E"
            self.COLORS["text"] = "#FFFFFF"
            self.COLORS["grid"] = "#444444"

    def create_equity_curve(
        self,
        equity_series: pd.Series,
        trades_df: pd.DataFrame | None = None,
        show_trades: bool = True,
    ) -> go.Figure:
        """Create interactive equity curve with trade markers.

        Args:
            equity_series: Equity values over time
            trades_df: DataFrame of trades with entry_time, exit_time, direction, pnl
            show_trades: Whether to show trade entry/exit markers

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode="lines",
                name="Equity",
                line=dict(color=self.COLORS["equity"], width=2),
                hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
            )
        )

        # Add trade markers
        if show_trades and trades_df is not None and not trades_df.empty:
            # Entry markers
            long_entries = trades_df[trades_df["direction"] == "LONG"]
            short_entries = trades_df[trades_df["direction"] == "SHORT"]

            if not long_entries.empty:
                # Get equity values at entry times
                entry_times = pd.to_datetime(long_entries["entry_time"])
                entry_equity = self._get_equity_at_times(equity_series, entry_times)

                fig.add_trace(
                    go.Scatter(
                        x=entry_times,
                        y=entry_equity,
                        mode="markers",
                        name="Long Entry",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color=self.COLORS["long_entry"],
                        ),
                        hovertemplate="<b>Long Entry</b><br>%{x}<br>Equity: $%{y:,.2f}<extra></extra>",
                    )
                )

            if not short_entries.empty:
                entry_times = pd.to_datetime(short_entries["entry_time"])
                entry_equity = self._get_equity_at_times(equity_series, entry_times)

                fig.add_trace(
                    go.Scatter(
                        x=entry_times,
                        y=entry_equity,
                        mode="markers",
                        name="Short Entry",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color=self.COLORS["short_entry"],
                        ),
                        hovertemplate="<b>Short Entry</b><br>%{x}<br>Equity: $%{y:,.2f}<extra></extra>",
                    )
                )

            # Exit markers (color by P&L)
            winners = trades_df[trades_df["pnl"] > 0]
            losers = trades_df[trades_df["pnl"] <= 0]

            if not winners.empty:
                exit_times = pd.to_datetime(winners["exit_time"])
                exit_equity = self._get_equity_at_times(equity_series, exit_times)

                fig.add_trace(
                    go.Scatter(
                        x=exit_times,
                        y=exit_equity,
                        mode="markers",
                        name="Winning Exit",
                        marker=dict(
                            symbol="circle",
                            size=8,
                            color=self.COLORS["win"],
                        ),
                        hovertemplate="<b>Winner</b><br>%{x}<br>P&L: $%{customdata:,.2f}<extra></extra>",
                        customdata=winners["pnl"].values,
                    )
                )

            if not losers.empty:
                exit_times = pd.to_datetime(losers["exit_time"])
                exit_equity = self._get_equity_at_times(equity_series, exit_times)

                fig.add_trace(
                    go.Scatter(
                        x=exit_times,
                        y=exit_equity,
                        mode="markers",
                        name="Losing Exit",
                        marker=dict(
                            symbol="circle",
                            size=8,
                            color=self.COLORS["loss"],
                        ),
                        hovertemplate="<b>Loser</b><br>%{x}<br>P&L: $%{customdata:,.2f}<extra></extra>",
                        customdata=losers["pnl"].values,
                    )
                )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template=self.TEMPLATE,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def create_drawdown_chart(self, equity_series: pd.Series) -> go.Figure:
        """Create underwater drawdown chart.

        Args:
            equity_series: Equity values over time

        Returns:
            Plotly Figure object
        """
        # Calculate drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color=self.COLORS["drawdown"], width=1),
                fillcolor=f"rgba(233, 79, 55, 0.3)",
                hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title="Drawdown (Underwater Chart)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.TEMPLATE,
            yaxis=dict(ticksuffix="%"),
        )

        return fig

    def create_trade_distribution(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create histogram of trade P&L distribution.

        Args:
            trades_df: DataFrame of trades with 'pnl' column

        Returns:
            Plotly Figure object
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return self._empty_chart("No trades to display")

        pnl = trades_df["pnl"]

        # Create histogram with colors based on P&L
        fig = go.Figure()

        # Winners
        winners = pnl[pnl > 0]
        if not winners.empty:
            fig.add_trace(
                go.Histogram(
                    x=winners,
                    name="Winners",
                    marker_color=self.COLORS["win"],
                    opacity=0.7,
                    hovertemplate="P&L: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
                )
            )

        # Losers
        losers = pnl[pnl <= 0]
        if not losers.empty:
            fig.add_trace(
                go.Histogram(
                    x=losers,
                    name="Losers",
                    marker_color=self.COLORS["loss"],
                    opacity=0.7,
                    hovertemplate="P&L: $%{x:,.2f}<br>Count: %{y}<extra></extra>",
                )
            )

        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color=self.COLORS["neutral"])

        # Add mean line
        mean_pnl = pnl.mean()
        fig.add_vline(
            x=mean_pnl,
            line_dash="dot",
            line_color=self.COLORS["equity"],
            annotation_text=f"Mean: ${mean_pnl:.2f}",
            annotation_position="top",
        )

        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Number of Trades",
            template=self.TEMPLATE,
            barmode="overlay",
            showlegend=True,
        )

        return fig

    def create_monthly_returns_heatmap(self, equity_series: pd.Series) -> go.Figure:
        """Create calendar heatmap of monthly returns.

        Args:
            equity_series: Equity values over time

        Returns:
            Plotly Figure object
        """
        if len(equity_series) < 2:
            return self._empty_chart("Not enough data for monthly returns")

        # Calculate monthly returns
        monthly = equity_series.resample("ME").last()
        monthly_returns = monthly.pct_change() * 100
        monthly_returns = monthly_returns.dropna()

        if monthly_returns.empty:
            return self._empty_chart("Not enough data for monthly returns")

        # Create pivot table for heatmap
        df = pd.DataFrame(
            {
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month,
                "return": monthly_returns.values,
            }
        )

        pivot = df.pivot(index="year", columns="month", values="return")

        # Month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=month_names[: len(pivot.columns)],
                y=pivot.index.astype(str),
                colorscale=[
                    [0, self.COLORS["loss"]],
                    [0.5, "white"],
                    [1, self.COLORS["win"]],
                ],
                zmid=0,
                text=np.round(pivot.values, 2),
                texttemplate="%{text:.1f}%",
                textfont={"size": 10},
                hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
                colorbar=dict(title="Return (%)", ticksuffix="%"),
            )
        )

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.TEMPLATE,
        )

        return fig

    def create_rolling_sharpe(
        self, equity_series: pd.Series, window: int = 30, annualize: bool = True
    ) -> go.Figure:
        """Create rolling Sharpe ratio chart.

        Args:
            equity_series: Equity values over time
            window: Rolling window in periods
            annualize: Whether to annualize the Sharpe ratio

        Returns:
            Plotly Figure object
        """
        if len(equity_series) < window:
            return self._empty_chart(f"Not enough data for {window}-period rolling Sharpe")

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Rolling mean and std
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        # Calculate rolling Sharpe
        rolling_sharpe = rolling_mean / rolling_std

        if annualize:
            # Estimate periods per year
            total_days = (equity_series.index[-1] - equity_series.index[0]).days
            periods_per_year = len(equity_series) / (total_days / 365.25) if total_days > 0 else 252
            rolling_sharpe = rolling_sharpe * np.sqrt(periods_per_year)

        rolling_sharpe = rolling_sharpe.dropna()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"{window}-Period Rolling Sharpe",
                line=dict(color=self.COLORS["equity"], width=2),
                hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
            )
        )

        # Add horizontal lines for reference
        fig.add_hline(y=0, line_dash="solid", line_color=self.COLORS["neutral"])
        fig.add_hline(y=1, line_dash="dot", line_color=self.COLORS["win"])
        fig.add_hline(y=-1, line_dash="dot", line_color=self.COLORS["loss"])

        fig.update_layout(
            title=f"Rolling Sharpe Ratio ({window} periods)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template=self.TEMPLATE,
        )

        return fig

    def create_trade_analysis(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create trade analysis charts (by hour, day of week, hold time).

        Args:
            trades_df: DataFrame of trades

        Returns:
            Plotly Figure with subplots
        """
        if trades_df.empty:
            return self._empty_chart("No trades to analyze")

        # Prepare data
        trades = trades_df.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
        trades["hour"] = trades["entry_time"].dt.hour
        trades["day_of_week"] = trades["entry_time"].dt.day_name()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "P&L by Hour of Day",
                "P&L by Day of Week",
                "Win Rate by Hour",
                "Trade Duration Distribution",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. P&L by Hour
        hourly_pnl = trades.groupby("hour")["pnl"].sum()
        colors = [self.COLORS["win"] if v > 0 else self.COLORS["loss"] for v in hourly_pnl.values]

        fig.add_trace(
            go.Bar(
                x=hourly_pnl.index,
                y=hourly_pnl.values,
                marker_color=colors,
                name="Hourly P&L",
                hovertemplate="Hour: %{x}<br>P&L: $%{y:,.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 2. P&L by Day of Week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_pnl = trades.groupby("day_of_week")["pnl"].sum().reindex(day_order).dropna()
        colors = [self.COLORS["win"] if v > 0 else self.COLORS["loss"] for v in daily_pnl.values]

        fig.add_trace(
            go.Bar(
                x=daily_pnl.index,
                y=daily_pnl.values,
                marker_color=colors,
                name="Daily P&L",
                hovertemplate="Day: %{x}<br>P&L: $%{y:,.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Win Rate by Hour
        hourly_wins = trades.groupby("hour", group_keys=False).apply(
            lambda x: (x["pnl"] > 0).mean() * 100 if len(x) > 0 else 0,
            include_groups=False,
        )

        fig.add_trace(
            go.Bar(
                x=hourly_wins.index,
                y=hourly_wins.values,
                marker_color=self.COLORS["equity"],
                name="Win Rate",
                hovertemplate="Hour: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add 50% reference line
        fig.add_hline(
            y=50, line_dash="dash", line_color=self.COLORS["neutral"], row=2, col=1
        )

        # 4. Trade Duration Distribution
        if "duration" in trades.columns:
            # Convert duration to minutes if it's a timedelta
            durations = trades["duration"]
            if pd.api.types.is_timedelta64_dtype(durations):
                durations = durations.dt.total_seconds() / 60
            else:
                # Try to convert if stored as string or other format
                try:
                    durations = pd.to_timedelta(durations).dt.total_seconds() / 60
                except (ValueError, TypeError):
                    durations = pd.Series([0])

            fig.add_trace(
                go.Histogram(
                    x=durations,
                    marker_color=self.COLORS["equity"],
                    name="Duration",
                    hovertemplate="Duration: %{x:.0f} min<br>Count: %{y}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Trade Analysis",
            template=self.TEMPLATE,
            height=600,
            showlegend=False,
        )

        # Update axis labels
        fig.update_xaxes(title_text="Hour", row=1, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="Day", row=1, col=2)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=2)
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Duration (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        return fig

    def create_cumulative_pnl(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create cumulative P&L chart from trades.

        Args:
            trades_df: DataFrame of trades with 'pnl' and 'exit_time' columns

        Returns:
            Plotly Figure object
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return self._empty_chart("No trades to display")

        trades = trades_df.copy()
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
        trades = trades.sort_values("exit_time")
        trades["cumulative_pnl"] = trades["pnl"].cumsum()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trades["exit_time"],
                y=trades["cumulative_pnl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color=self.COLORS["equity"], width=2),
                marker=dict(
                    size=6,
                    color=[
                        self.COLORS["win"] if p > 0 else self.COLORS["loss"]
                        for p in trades["pnl"]
                    ],
                ),
                hovertemplate="<b>%{x}</b><br>Cumulative: $%{y:,.2f}<br>Trade: $%{customdata:,.2f}<extra></extra>",
                customdata=trades["pnl"],
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color=self.COLORS["neutral"])

        fig.update_layout(
            title="Cumulative Trade P&L",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
            template=self.TEMPLATE,
        )

        return fig

    def create_streak_analysis(self, trades_df: pd.DataFrame) -> go.Figure:
        """Create win/loss streak analysis chart.

        Args:
            trades_df: DataFrame of trades with 'pnl' column

        Returns:
            Plotly Figure object
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return self._empty_chart("No trades to analyze")

        # Calculate streaks
        wins = (trades_df["pnl"] > 0).astype(int)
        streak_groups = (wins != wins.shift()).cumsum()
        streaks = wins.groupby(streak_groups).agg(["sum", "count"])

        win_streaks = []
        loss_streaks = []

        for _, row in streaks.iterrows():
            if row["sum"] > 0:  # Winning streak
                win_streaks.append(row["count"])
            else:  # Losing streak
                loss_streaks.append(row["count"])

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Winning Streaks", "Losing Streaks"),
        )

        if win_streaks:
            fig.add_trace(
                go.Histogram(
                    x=win_streaks,
                    marker_color=self.COLORS["win"],
                    name="Win Streaks",
                    hovertemplate="Streak Length: %{x}<br>Count: %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if loss_streaks:
            fig.add_trace(
                go.Histogram(
                    x=loss_streaks,
                    marker_color=self.COLORS["loss"],
                    name="Loss Streaks",
                    hovertemplate="Streak Length: %{x}<br>Count: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            title="Win/Loss Streak Distribution",
            template=self.TEMPLATE,
            showlegend=False,
        )

        fig.update_xaxes(title_text="Streak Length", row=1, col=1)
        fig.update_xaxes(title_text="Streak Length", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        return fig

    def _get_equity_at_times(
        self, equity_series: pd.Series, times: pd.DatetimeIndex
    ) -> list[float]:
        """Get equity values at specific times using nearest match.

        Args:
            equity_series: Equity values over time
            times: Times to look up

        Returns:
            List of equity values
        """
        # Handle duplicate timestamps by keeping the last value for each timestamp
        if equity_series.index.duplicated().any():
            equity_series = equity_series[~equity_series.index.duplicated(keep='last')]

        values = []
        for t in times:
            # Find nearest time in equity series
            try:
                idx = equity_series.index.get_indexer([t], method="nearest")[0]
                if idx >= 0 and idx < len(equity_series):
                    values.append(equity_series.iloc[idx])
                else:
                    values.append(equity_series.iloc[0] if len(equity_series) > 0 else 0)
            except Exception:
                # Fallback: find closest manually
                if len(equity_series) > 0:
                    time_diffs = abs(equity_series.index - t)
                    closest_idx = time_diffs.argmin()
                    values.append(equity_series.iloc[closest_idx])
                else:
                    values.append(0)
        return values

    def _empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message.

        Args:
            message: Message to display

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.COLORS["neutral"]),
        )
        fig.update_layout(
            template=self.TEMPLATE,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        return fig

    def create_overview_figure(
        self,
        equity_series: pd.Series,
        trades_df: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Create combined overview figure with equity curve and drawdown.

        Args:
            equity_series: Equity values over time
            trades_df: DataFrame of trades

        Returns:
            Plotly Figure with subplots
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Equity Curve", "Drawdown"),
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode="lines",
                name="Equity",
                line=dict(color=self.COLORS["equity"], width=2),
                hovertemplate="<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Add trade markers if available
        if trades_df is not None and not trades_df.empty:
            winners = trades_df[trades_df["pnl"] > 0]
            losers = trades_df[trades_df["pnl"] <= 0]

            if not winners.empty:
                exit_times = pd.to_datetime(winners["exit_time"])
                exit_equity = self._get_equity_at_times(equity_series, exit_times)
                fig.add_trace(
                    go.Scatter(
                        x=exit_times,
                        y=exit_equity,
                        mode="markers",
                        name="Winners",
                        marker=dict(symbol="circle", size=6, color=self.COLORS["win"]),
                        hovertemplate="<b>Winner</b><br>P&L: $%{customdata:,.2f}<extra></extra>",
                        customdata=winners["pnl"].values,
                    ),
                    row=1,
                    col=1,
                )

            if not losers.empty:
                exit_times = pd.to_datetime(losers["exit_time"])
                exit_equity = self._get_equity_at_times(equity_series, exit_times)
                fig.add_trace(
                    go.Scatter(
                        x=exit_times,
                        y=exit_equity,
                        mode="markers",
                        name="Losers",
                        marker=dict(symbol="circle", size=6, color=self.COLORS["loss"]),
                        hovertemplate="<b>Loser</b><br>P&L: $%{customdata:,.2f}<extra></extra>",
                        customdata=losers["pnl"].values,
                    ),
                    row=1,
                    col=1,
                )

        # Drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color=self.COLORS["drawdown"], width=1),
                fillcolor="rgba(233, 79, 55, 0.3)",
                hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Backtest Overview",
            template=self.TEMPLATE,
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig
