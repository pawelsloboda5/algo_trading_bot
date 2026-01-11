"""Report generation and result saving for backtests."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.logging_config import get_logger
from src.backtest.engine import BacktestResult
from src.visualization.charts import ChartGenerator

logger = get_logger(__name__)


class ResultSaver:
    """Saves backtest results to structured files for analysis."""

    def __init__(self, results_dir: str | Path = "results/backtests"):
        """Initialize result saver.

        Args:
            results_dir: Base directory for saving results
        """
        self.results_dir = Path(results_dir)
        self.chart_generator = ChartGenerator()

    def save(
        self,
        result: BacktestResult,
        symbol: str,
        run_id: str | None = None,
        save_charts: bool = True,
    ) -> Path:
        """Save all backtest results to files.

        Args:
            result: BacktestResult object
            symbol: Symbol being backtested
            run_id: Optional run identifier (auto-generated if not provided)
            save_charts: Whether to save individual chart HTML files

        Returns:
            Path to the results directory
        """
        # Generate run_id if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            strategy_name = result.strategy.name.lower().replace(" ", "_")
            run_id = f"{timestamp}_{strategy_name}_{symbol}"

        # Create results directory
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("saving_backtest_results", run_id=run_id, directory=str(run_dir))

        # Save all components
        self._save_summary_json(result, symbol, run_id, run_dir)
        self._save_trades_csv(result, run_dir)
        self._save_equity_curve_csv(result, run_dir)
        self._save_daily_pnl_csv(result, run_dir)
        self._save_parameters_json(result, run_dir)

        if save_charts:
            self._save_charts(result, run_dir)

        self._save_report_html(result, symbol, run_id, run_dir)

        logger.info("backtest_results_saved", run_id=run_id, files_created=7)

        return run_dir

    def _save_summary_json(
        self,
        result: BacktestResult,
        symbol: str,
        run_id: str,
        run_dir: Path,
    ) -> None:
        """Save AI-friendly summary JSON.

        This file contains everything an AI needs to optimize the strategy.
        """
        metrics = result.metrics
        trades_df = result.trades
        equity_series = result.equity_curve

        # Data range
        if len(equity_series) > 0:
            start_date = equity_series.index[0]
            end_date = equity_series.index[-1]
            trading_days = (end_date - start_date).days
        else:
            start_date = end_date = None
            trading_days = 0

        # Time analysis
        time_analysis = self._calculate_time_analysis(trades_df)

        # Risk metrics from config
        risk_metrics = {
            "risk_per_trade": result.config.risk_per_trade,
            "daily_loss_limit": result.config.daily_loss_limit,
            "max_position_contracts": result.config.max_position_contracts,
            "use_risk_manager": result.config.use_risk_manager,
        }

        # Calculate streak analysis
        streak_analysis = self._calculate_streak_analysis(trades_df)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, trades_df, time_analysis, streak_analysis
        )

        summary = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "strategy": {
                "name": result.strategy.name,
                "class": result.strategy.__class__.__name__,
                "parameters": result.strategy.get_parameters(),
            },
            "data_range": {
                "start": str(start_date.date()) if start_date else None,
                "end": str(end_date.date()) if end_date else None,
                "trading_days": trading_days,
                "total_bars": len(equity_series),
            },
            "performance": {
                "total_return_pct": round(metrics.total_return_pct, 2),
                "total_return_usd": round(metrics.total_return, 2),
                "cagr": round(metrics.cagr, 2),
                "sharpe_ratio": round(metrics.sharpe_ratio, 2),
                "sortino_ratio": round(metrics.sortino_ratio, 2),
                "calmar_ratio": round(metrics.calmar_ratio, 2),
                "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
                "max_drawdown_usd": round(metrics.max_drawdown, 2),
                "win_rate": round(metrics.win_rate, 2),
                "profit_factor": round(metrics.profit_factor, 2),
                "avg_trade_pnl": round(metrics.avg_trade_pnl, 2),
                "avg_winner": round(metrics.avg_winner, 2),
                "avg_loser": round(metrics.avg_loser, 2),
                "largest_winner": round(metrics.largest_winner, 2),
                "largest_loser": round(metrics.largest_loser, 2),
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "avg_hold_time_minutes": self._get_avg_hold_time_minutes(trades_df),
                "max_consecutive_wins": streak_analysis.get("max_win_streak", 0),
                "max_consecutive_losses": streak_analysis.get("max_loss_streak", 0),
                "time_in_market": round(metrics.time_in_market, 2),
            },
            "risk_metrics": risk_metrics,
            "time_analysis": time_analysis,
            "streak_analysis": streak_analysis,
            "recommendations": recommendations,
        }

        # Save to file
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.debug("saved_summary_json", path=str(summary_path))

    def _save_trades_csv(self, result: BacktestResult, run_dir: Path) -> None:
        """Save trades to CSV."""
        if result.trades.empty:
            # Create empty file with headers
            pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "direction",
                    "quantity",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "commission",
                    "duration",
                ]
            ).to_csv(run_dir / "trades.csv", index=False)
            return

        result.trades.to_csv(run_dir / "trades.csv", index=False)

    def _save_equity_curve_csv(self, result: BacktestResult, run_dir: Path) -> None:
        """Save equity curve to CSV."""
        equity_df = pd.DataFrame(
            {
                "timestamp": result.equity_curve.index,
                "equity": result.equity_curve.values,
            }
        )
        equity_df.to_csv(run_dir / "equity_curve.csv", index=False)

    def _save_daily_pnl_csv(self, result: BacktestResult, run_dir: Path) -> None:
        """Save daily P&L breakdown to CSV."""
        if len(result.equity_curve) < 2:
            pd.DataFrame(columns=["date", "pnl", "cumulative_pnl"]).to_csv(
                run_dir / "daily_pnl.csv", index=False
            )
            return

        # Calculate daily P&L from equity curve
        daily_equity = result.equity_curve.resample("D").last().dropna()
        daily_pnl = daily_equity.diff().dropna()

        daily_df = pd.DataFrame(
            {
                "date": daily_pnl.index.date,
                "pnl": daily_pnl.values,
                "cumulative_pnl": daily_pnl.cumsum().values,
                "equity": daily_equity.iloc[1:].values,
            }
        )

        daily_df.to_csv(run_dir / "daily_pnl.csv", index=False)

    def _save_parameters_json(self, result: BacktestResult, run_dir: Path) -> None:
        """Save strategy and backtest parameters to JSON."""
        params = {
            "strategy": {
                "name": result.strategy.name,
                "class": result.strategy.__class__.__name__,
                "parameters": result.strategy.get_parameters(),
            },
            "backtest_config": {
                "initial_capital": result.config.initial_capital,
                "contract_multiplier": result.config.contract_multiplier,
                "commission_per_contract": result.config.commission_per_contract,
                "slippage_ticks": result.config.slippage_ticks,
                "tick_size": result.config.tick_size,
                "position_size": result.config.position_size,
                "use_risk_manager": result.config.use_risk_manager,
                "max_position_contracts": result.config.max_position_contracts,
                "risk_per_trade": result.config.risk_per_trade,
                "daily_loss_limit": result.config.daily_loss_limit,
            },
        }

        with open(run_dir / "parameters.json", "w") as f:
            json.dump(params, f, indent=2)

    def _save_charts(self, result: BacktestResult, run_dir: Path) -> None:
        """Save individual chart HTML files."""
        charts_dir = run_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Generate and save each chart
        charts = [
            ("equity_curve", self.chart_generator.create_equity_curve(
                result.equity_curve, result.trades
            )),
            ("drawdown", self.chart_generator.create_drawdown_chart(result.equity_curve)),
            ("trade_distribution", self.chart_generator.create_trade_distribution(result.trades)),
            ("monthly_returns", self.chart_generator.create_monthly_returns_heatmap(
                result.equity_curve
            )),
            ("rolling_sharpe", self.chart_generator.create_rolling_sharpe(result.equity_curve)),
            ("trade_analysis", self.chart_generator.create_trade_analysis(result.trades)),
            ("cumulative_pnl", self.chart_generator.create_cumulative_pnl(result.trades)),
            ("streak_analysis", self.chart_generator.create_streak_analysis(result.trades)),
        ]

        for name, fig in charts:
            fig.write_html(charts_dir / f"{name}.html", include_plotlyjs="cdn")

        logger.debug("saved_charts", count=len(charts))

    def _save_report_html(
        self,
        result: BacktestResult,
        symbol: str,
        run_id: str,
        run_dir: Path,
    ) -> None:
        """Save human-readable HTML report with embedded charts."""
        metrics = result.metrics

        # Generate overview chart
        overview_fig = self.chart_generator.create_overview_figure(
            result.equity_curve, result.trades
        )
        overview_html = overview_fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Generate trade distribution chart
        dist_fig = self.chart_generator.create_trade_distribution(result.trades)
        dist_html = dist_fig.to_html(full_html=False, include_plotlyjs=False)

        # Generate monthly returns
        monthly_fig = self.chart_generator.create_monthly_returns_heatmap(result.equity_curve)
        monthly_html = monthly_fig.to_html(full_html=False, include_plotlyjs=False)

        # Time analysis
        time_analysis = self._calculate_time_analysis(result.trades)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .metric-card .value.positive {{ color: #27ae60; }}
        .metric-card .value.negative {{ color: #e74c3c; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .info-box {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .strategy-params {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .param {{
            background: white;
            padding: 10px;
            border-radius: 4px;
        }}
        .param .name {{ font-size: 12px; color: #7f8c8d; }}
        .param .value {{ font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        footer {{
            margin-top: 40px;
            padding: 20px;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>

        <div class="info-box">
            <strong>Run ID:</strong> {run_id}<br>
            <strong>Symbol:</strong> {symbol}<br>
            <strong>Strategy:</strong> {result.strategy.name} ({result.strategy.__class__.__name__})<br>
            <strong>Date Range:</strong> {result.equity_curve.index[0].date() if len(result.equity_curve) > 0 else 'N/A'} to {result.equity_curve.index[-1].date() if len(result.equity_curve) > 0 else 'N/A'}
        </div>

        <h2>Strategy Parameters</h2>
        <div class="strategy-params">
            {self._generate_param_html(result.strategy.get_parameters())}
        </div>

        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Total Return</div>
                <div class="value {'positive' if metrics.total_return >= 0 else 'negative'}">
                    ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value {'positive' if metrics.sharpe_ratio >= 1 else 'negative' if metrics.sharpe_ratio < 0 else ''}">{metrics.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{metrics.max_drawdown_pct:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Win Rate</div>
                <div class="value {'positive' if metrics.win_rate >= 50 else 'negative'}">{metrics.win_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="label">Profit Factor</div>
                <div class="value {'positive' if metrics.profit_factor >= 1 else 'negative'}">{metrics.profit_factor:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{metrics.total_trades}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Winner</div>
                <div class="value positive">${metrics.avg_winner:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg Loser</div>
                <div class="value negative">${metrics.avg_loser:.2f}</div>
            </div>
        </div>

        <h2>Equity Curve & Drawdown</h2>
        <div class="chart-container">
            {overview_html}
        </div>

        <h2>Trade P&L Distribution</h2>
        <div class="chart-container">
            {dist_html}
        </div>

        <h2>Monthly Returns</h2>
        <div class="chart-container">
            {monthly_html}
        </div>

        <h2>Time Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="label">Best Hour</div>
                <div class="value">{time_analysis.get('best_hour', 'N/A')}:00</div>
            </div>
            <div class="metric-card">
                <div class="label">Worst Hour</div>
                <div class="value">{time_analysis.get('worst_hour', 'N/A')}:00</div>
            </div>
            <div class="metric-card">
                <div class="label">Best Day</div>
                <div class="value">{time_analysis.get('best_day_of_week', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Worst Day</div>
                <div class="value">{time_analysis.get('worst_day_of_week', 'N/A')}</div>
            </div>
        </div>

        {self._generate_recommendations_html(result, time_analysis)}

        <footer>
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Algo Trading Bot - Phase 5 Visualization
        </footer>
    </div>
</body>
</html>
"""

        with open(run_dir / "report.html", "w") as f:
            f.write(html)

    def _generate_param_html(self, params: dict) -> str:
        """Generate HTML for strategy parameters."""
        html_parts = []
        for name, value in params.items():
            html_parts.append(f"""
            <div class="param">
                <div class="name">{name}</div>
                <div class="value">{value}</div>
            </div>
            """)
        return "".join(html_parts)

    def _generate_recommendations_html(
        self,
        result: BacktestResult,
        time_analysis: dict,
    ) -> str:
        """Generate recommendations HTML section."""
        streak_analysis = self._calculate_streak_analysis(result.trades)
        recommendations = self._generate_recommendations(
            result.metrics, result.trades, time_analysis, streak_analysis
        )

        if not recommendations:
            return ""

        items = "\n".join([f"<li>{rec}</li>" for rec in recommendations])
        return f"""
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
                {items}
            </ul>
        </div>
        """

    def _calculate_time_analysis(self, trades_df: pd.DataFrame) -> dict[str, Any]:
        """Calculate time-based performance analysis.

        Args:
            trades_df: DataFrame of trades

        Returns:
            Dictionary with time analysis results
        """
        if trades_df.empty:
            return {
                "best_hour": None,
                "worst_hour": None,
                "best_day_of_week": None,
                "worst_day_of_week": None,
                "profitable_hours": [],
                "unprofitable_hours": [],
                "hourly_pnl": {},
                "daily_pnl": {},
            }

        trades = trades_df.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["hour"] = trades["entry_time"].dt.hour
        trades["day_of_week"] = trades["entry_time"].dt.day_name()

        # Hourly analysis
        hourly_pnl = trades.groupby("hour")["pnl"].sum()
        best_hour = int(hourly_pnl.idxmax()) if not hourly_pnl.empty else None
        worst_hour = int(hourly_pnl.idxmin()) if not hourly_pnl.empty else None
        profitable_hours = [int(h) for h in hourly_pnl[hourly_pnl > 0].index.tolist()]
        unprofitable_hours = [int(h) for h in hourly_pnl[hourly_pnl < 0].index.tolist()]

        # Daily analysis
        daily_pnl = trades.groupby("day_of_week")["pnl"].sum()
        best_day = daily_pnl.idxmax() if not daily_pnl.empty else None
        worst_day = daily_pnl.idxmin() if not daily_pnl.empty else None

        return {
            "best_hour": best_hour,
            "worst_hour": worst_hour,
            "best_day_of_week": best_day,
            "worst_day_of_week": worst_day,
            "profitable_hours": profitable_hours,
            "unprofitable_hours": unprofitable_hours,
            "hourly_pnl": {int(k): round(v, 2) for k, v in hourly_pnl.items()},
            "daily_pnl": {k: round(v, 2) for k, v in daily_pnl.items()},
        }

    def _calculate_streak_analysis(self, trades_df: pd.DataFrame) -> dict[str, Any]:
        """Calculate win/loss streak analysis.

        Args:
            trades_df: DataFrame of trades

        Returns:
            Dictionary with streak analysis
        """
        if trades_df.empty or "pnl" not in trades_df.columns:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "avg_win_streak": 0,
                "avg_loss_streak": 0,
                "current_streak": 0,
                "current_streak_type": None,
            }

        # Calculate streaks
        wins = (trades_df["pnl"] > 0).astype(int)
        streak_groups = (wins != wins.shift()).cumsum()

        win_streaks = []
        loss_streaks = []

        for group_id in streak_groups.unique():
            group = wins[streak_groups == group_id]
            if group.iloc[0] == 1:  # Win streak
                win_streaks.append(len(group))
            else:  # Loss streak
                loss_streaks.append(len(group))

        # Current streak
        if len(wins) > 0:
            last_result = wins.iloc[-1]
            last_group = streak_groups.iloc[-1]
            current_streak = len(wins[streak_groups == last_group])
            current_streak_type = "win" if last_result == 1 else "loss"
        else:
            current_streak = 0
            current_streak_type = None

        return {
            "max_win_streak": max(win_streaks) if win_streaks else 0,
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": round(sum(win_streaks) / len(win_streaks), 2) if win_streaks else 0,
            "avg_loss_streak": round(sum(loss_streaks) / len(loss_streaks), 2) if loss_streaks else 0,
            "current_streak": current_streak,
            "current_streak_type": current_streak_type,
        }

    def _generate_recommendations(
        self,
        metrics,
        trades_df: pd.DataFrame,
        time_analysis: dict,
        streak_analysis: dict,
    ) -> list[str]:
        """Generate actionable recommendations based on backtest results.

        Args:
            metrics: PerformanceMetrics object
            trades_df: DataFrame of trades
            time_analysis: Time analysis results
            streak_analysis: Streak analysis results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Win rate recommendations
        if metrics.win_rate < 40:
            recommendations.append(
                f"Win rate is low ({metrics.win_rate:.1f}%). Consider adjusting entry signals "
                "or using stronger trend confirmation."
            )
        elif metrics.win_rate > 70:
            recommendations.append(
                f"High win rate ({metrics.win_rate:.1f}%) but verify profit factor. "
                "May be cutting winners too early."
            )

        # Risk/reward recommendations
        if metrics.avg_winner != 0 and metrics.avg_loser != 0:
            rr_ratio = abs(metrics.avg_winner / metrics.avg_loser)
            if rr_ratio < 1:
                recommendations.append(
                    f"Risk/reward ratio is poor ({rr_ratio:.2f}:1). "
                    "Consider widening profit targets or tightening stops."
                )

        # Drawdown recommendations
        if metrics.max_drawdown_pct > 20:
            recommendations.append(
                f"Max drawdown is high ({metrics.max_drawdown_pct:.1f}%). "
                "Consider reducing position size or adding additional filters."
            )

        # Time-based recommendations
        if time_analysis.get("worst_hour") is not None:
            worst_hour = time_analysis["worst_hour"]
            hourly_pnl = time_analysis.get("hourly_pnl", {})
            if worst_hour in hourly_pnl and hourly_pnl[worst_hour] < -100:
                recommendations.append(
                    f"Significant losses at {worst_hour}:00 (${hourly_pnl[worst_hour]:.2f}). "
                    f"Consider adding time filter to avoid trading during this hour."
                )

        if time_analysis.get("worst_day_of_week") is not None:
            worst_day = time_analysis["worst_day_of_week"]
            daily_pnl = time_analysis.get("daily_pnl", {})
            if worst_day in daily_pnl and daily_pnl[worst_day] < -100:
                recommendations.append(
                    f"Performance degrades on {worst_day}s (${daily_pnl[worst_day]:.2f}). "
                    f"Consider reducing position size or skipping trades on this day."
                )

        # Streak recommendations
        if streak_analysis.get("max_loss_streak", 0) >= 5:
            recommendations.append(
                f"Max losing streak of {streak_analysis['max_loss_streak']} trades. "
                "Consider implementing a cool-down period after consecutive losses."
            )

        # Profit factor recommendations
        if 1 < metrics.profit_factor < 1.2:
            recommendations.append(
                f"Profit factor is marginal ({metrics.profit_factor:.2f}). "
                "Transaction costs could erode profits. Consider increasing win rate or R:R."
            )

        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append(
                f"Low Sharpe ratio ({metrics.sharpe_ratio:.2f}) indicates poor risk-adjusted returns. "
                "Consider adding volatility-based position sizing."
            )

        # Time in market
        if metrics.time_in_market > 80:
            recommendations.append(
                f"High time in market ({metrics.time_in_market:.1f}%). "
                "Strategy may be overtrading. Consider stricter entry criteria."
            )
        elif metrics.time_in_market < 5:
            recommendations.append(
                f"Low time in market ({metrics.time_in_market:.1f}%). "
                "Strategy may be too conservative. Consider relaxing entry criteria."
            )

        return recommendations

    def _get_avg_hold_time_minutes(self, trades_df: pd.DataFrame) -> float:
        """Get average hold time in minutes.

        Args:
            trades_df: DataFrame of trades

        Returns:
            Average hold time in minutes
        """
        if trades_df.empty or "duration" not in trades_df.columns:
            return 0.0

        durations = trades_df["duration"]
        if pd.api.types.is_timedelta64_dtype(durations):
            return round(durations.mean().total_seconds() / 60, 2)

        # Try to convert
        try:
            durations = pd.to_timedelta(durations)
            return round(durations.mean().total_seconds() / 60, 2)
        except (ValueError, TypeError):
            return 0.0
