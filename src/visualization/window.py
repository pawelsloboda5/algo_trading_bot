"""Dash-based interactive visualization window for backtest results."""

import threading
import webbrowser
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output

from config.logging_config import get_logger
from src.backtest.engine import BacktestResult
from src.visualization.charts import ChartGenerator
from src.visualization.report import ResultSaver

logger = get_logger(__name__)


class BacktestViewer:
    """Interactive Dash-based visualization window for backtest results."""

    def __init__(self, result: BacktestResult, symbol: str = "MCL_FUT"):
        """Initialize the backtest viewer.

        Args:
            result: BacktestResult object to visualize
            symbol: Symbol being backtested
        """
        self.result = result
        self.symbol = symbol
        self.chart_generator = ChartGenerator()
        self.result_saver = ResultSaver()

        # Pre-calculate time and streak analysis
        self.time_analysis = self.result_saver._calculate_time_analysis(result.trades)
        self.streak_analysis = self.result_saver._calculate_streak_analysis(result.trades)

        self.app = self._create_app()

    def _create_app(self) -> dash.Dash:
        """Create the Dash application.

        Returns:
            Configured Dash app
        """
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title=f"Backtest Results - {self.symbol}",
        )

        app.layout = self._create_layout()
        self._register_callbacks(app)

        return app

    def _create_layout(self) -> dbc.Container:
        """Create the dashboard layout.

        Returns:
            Dash Bootstrap Container with all components
        """
        metrics = self.result.metrics

        return dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(
                                    f"Backtest Results: {self.symbol}",
                                    className="text-primary mb-0",
                                ),
                                html.P(
                                    f"{self.result.strategy.name} | "
                                    f"{self.result.equity_curve.index[0].date() if len(self.result.equity_curve) > 0 else 'N/A'} to "
                                    f"{self.result.equity_curve.index[-1].date() if len(self.result.equity_curve) > 0 else 'N/A'}",
                                    className="text-muted",
                                ),
                            ]
                        )
                    ],
                    className="my-4",
                ),
                # Tabs
                dbc.Tabs(
                    [
                        dbc.Tab(self._create_overview_tab(), label="Overview", tab_id="overview"),
                        dbc.Tab(self._create_trades_tab(), label="Trades", tab_id="trades"),
                        dbc.Tab(self._create_analysis_tab(), label="Analysis", tab_id="analysis"),
                        dbc.Tab(
                            self._create_parameters_tab(), label="Parameters", tab_id="parameters"
                        ),
                    ],
                    id="tabs",
                    active_tab="overview",
                ),
            ],
            fluid=True,
            className="p-4",
        )

    def _create_overview_tab(self) -> dbc.Container:
        """Create the Overview tab content."""
        metrics = self.result.metrics

        # Metric cards
        metric_cards = dbc.Row(
            [
                dbc.Col(
                    self._metric_card(
                        "Total Return",
                        f"${metrics.total_return:,.2f}",
                        f"{metrics.total_return_pct:+.2f}%",
                        "success" if metrics.total_return >= 0 else "danger",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Sharpe Ratio",
                        f"{metrics.sharpe_ratio:.2f}",
                        "Risk-adjusted return",
                        "success" if metrics.sharpe_ratio >= 1 else "warning" if metrics.sharpe_ratio > 0 else "danger",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Max Drawdown",
                        f"{metrics.max_drawdown_pct:.2f}%",
                        f"${metrics.max_drawdown:,.2f}",
                        "danger" if metrics.max_drawdown_pct > 15 else "warning",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Win Rate",
                        f"{metrics.win_rate:.1f}%",
                        f"{metrics.winning_trades}/{metrics.total_trades} trades",
                        "success" if metrics.win_rate >= 50 else "danger",
                    ),
                    md=3,
                ),
            ],
            className="mb-4",
        )

        # Secondary metrics
        secondary_metrics = dbc.Row(
            [
                dbc.Col(
                    self._metric_card(
                        "Profit Factor",
                        f"{metrics.profit_factor:.2f}",
                        "Gross profit / Gross loss",
                        "success" if metrics.profit_factor >= 1.5 else "warning" if metrics.profit_factor >= 1 else "danger",
                    ),
                    md=2,
                ),
                dbc.Col(
                    self._metric_card(
                        "Sortino Ratio",
                        f"{metrics.sortino_ratio:.2f}",
                        "Downside risk",
                        "info",
                    ),
                    md=2,
                ),
                dbc.Col(
                    self._metric_card(
                        "CAGR",
                        f"{metrics.cagr:.2f}%",
                        "Compound annual",
                        "success" if metrics.cagr > 0 else "danger",
                    ),
                    md=2,
                ),
                dbc.Col(
                    self._metric_card(
                        "Avg Winner",
                        f"${metrics.avg_winner:.2f}",
                        f"Best: ${metrics.largest_winner:.2f}",
                        "success",
                    ),
                    md=2,
                ),
                dbc.Col(
                    self._metric_card(
                        "Avg Loser",
                        f"${metrics.avg_loser:.2f}",
                        f"Worst: ${metrics.largest_loser:.2f}",
                        "danger",
                    ),
                    md=2,
                ),
                dbc.Col(
                    self._metric_card(
                        "Time in Market",
                        f"{metrics.time_in_market:.1f}%",
                        "Exposure",
                        "info",
                    ),
                    md=2,
                ),
            ],
            className="mb-4",
        )

        # Charts
        overview_chart = self.chart_generator.create_overview_figure(
            self.result.equity_curve, self.result.trades
        )

        return dbc.Container(
            [
                metric_cards,
                secondary_metrics,
                dbc.Card(
                    [
                        dbc.CardHeader("Equity Curve & Drawdown"),
                        dbc.CardBody(dcc.Graph(figure=overview_chart, id="overview-chart")),
                    ],
                    className="mb-4",
                ),
            ],
            fluid=True,
        )

    def _create_trades_tab(self) -> dbc.Container:
        """Create the Trades tab content."""
        trades_df = self.result.trades

        # Trade distribution chart
        dist_chart = self.chart_generator.create_trade_distribution(trades_df)

        # Cumulative P&L chart
        cum_pnl_chart = self.chart_generator.create_cumulative_pnl(trades_df)

        # Trade table
        if not trades_df.empty:
            # Prepare table data
            display_df = trades_df.copy()
            if "entry_time" in display_df.columns:
                display_df["entry_time"] = pd.to_datetime(display_df["entry_time"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
            if "exit_time" in display_df.columns:
                display_df["exit_time"] = pd.to_datetime(display_df["exit_time"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
            if "pnl" in display_df.columns:
                display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:,.2f}")
            if "entry_price" in display_df.columns:
                display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
            if "exit_price" in display_df.columns:
                display_df["exit_price"] = display_df["exit_price"].apply(lambda x: f"${x:.2f}")

            # Select columns to display
            display_cols = [
                col
                for col in ["entry_time", "exit_time", "direction", "quantity", "entry_price", "exit_price", "pnl"]
                if col in display_df.columns
            ]

            table = dbc.Table.from_dataframe(
                display_df[display_cols].tail(50),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm",
            )
        else:
            table = html.P("No trades to display", className="text-muted")

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("P&L Distribution"),
                                    dbc.CardBody(dcc.Graph(figure=dist_chart)),
                                ]
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Cumulative P&L"),
                                    dbc.CardBody(dcc.Graph(figure=cum_pnl_chart)),
                                ]
                            ),
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(f"Trade History (Last 50 of {len(trades_df)})"),
                        dbc.CardBody(table, style={"maxHeight": "400px", "overflowY": "auto"}),
                    ]
                ),
            ],
            fluid=True,
        )

    def _create_analysis_tab(self) -> dbc.Container:
        """Create the Analysis tab content."""
        # Monthly returns heatmap
        monthly_chart = self.chart_generator.create_monthly_returns_heatmap(self.result.equity_curve)

        # Trade analysis
        trade_analysis_chart = self.chart_generator.create_trade_analysis(self.result.trades)

        # Rolling Sharpe
        rolling_sharpe_chart = self.chart_generator.create_rolling_sharpe(self.result.equity_curve)

        # Streak analysis
        streak_chart = self.chart_generator.create_streak_analysis(self.result.trades)

        # Time analysis cards
        time_cards = dbc.Row(
            [
                dbc.Col(
                    self._metric_card(
                        "Best Hour",
                        f"{self.time_analysis.get('best_hour', 'N/A')}:00"
                        if self.time_analysis.get("best_hour") is not None
                        else "N/A",
                        "Most profitable",
                        "success",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Worst Hour",
                        f"{self.time_analysis.get('worst_hour', 'N/A')}:00"
                        if self.time_analysis.get("worst_hour") is not None
                        else "N/A",
                        "Least profitable",
                        "danger",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Best Day",
                        self.time_analysis.get("best_day_of_week", "N/A") or "N/A",
                        "Most profitable",
                        "success",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Worst Day",
                        self.time_analysis.get("worst_day_of_week", "N/A") or "N/A",
                        "Least profitable",
                        "danger",
                    ),
                    md=3,
                ),
            ],
            className="mb-4",
        )

        # Streak cards
        streak_cards = dbc.Row(
            [
                dbc.Col(
                    self._metric_card(
                        "Max Win Streak",
                        str(self.streak_analysis.get("max_win_streak", 0)),
                        "Consecutive wins",
                        "success",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Max Loss Streak",
                        str(self.streak_analysis.get("max_loss_streak", 0)),
                        "Consecutive losses",
                        "danger",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Avg Win Streak",
                        f"{self.streak_analysis.get('avg_win_streak', 0):.1f}",
                        "Average length",
                        "info",
                    ),
                    md=3,
                ),
                dbc.Col(
                    self._metric_card(
                        "Avg Loss Streak",
                        f"{self.streak_analysis.get('avg_loss_streak', 0):.1f}",
                        "Average length",
                        "info",
                    ),
                    md=3,
                ),
            ],
            className="mb-4",
        )

        return dbc.Container(
            [
                html.H4("Time Analysis", className="mb-3"),
                time_cards,
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Monthly Returns"),
                                    dbc.CardBody(dcc.Graph(figure=monthly_chart)),
                                ]
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Rolling Sharpe Ratio (30-period)"),
                                    dbc.CardBody(dcc.Graph(figure=rolling_sharpe_chart)),
                                ]
                            ),
                            md=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Trade Analysis by Time"),
                        dbc.CardBody(dcc.Graph(figure=trade_analysis_chart)),
                    ],
                    className="mb-4",
                ),
                html.H4("Streak Analysis", className="mb-3"),
                streak_cards,
                dbc.Card(
                    [
                        dbc.CardHeader("Win/Loss Streak Distribution"),
                        dbc.CardBody(dcc.Graph(figure=streak_chart)),
                    ]
                ),
            ],
            fluid=True,
        )

    def _create_parameters_tab(self) -> dbc.Container:
        """Create the Parameters tab content."""
        strategy_params = self.result.strategy.get_parameters()
        config = self.result.config

        # Generate recommendations
        recommendations = self.result_saver._generate_recommendations(
            self.result.metrics,
            self.result.trades,
            self.time_analysis,
            self.streak_analysis,
        )

        # Strategy parameters card
        strategy_items = [
            dbc.ListGroupItem([html.Strong(k.replace("_", " ").title() + ": "), str(v)])
            for k, v in strategy_params.items()
        ]

        # Backtest config card
        config_items = [
            dbc.ListGroupItem([html.Strong("Initial Capital: "), f"${config.initial_capital:,.2f}"]),
            dbc.ListGroupItem([html.Strong("Contract Multiplier: "), str(config.contract_multiplier)]),
            dbc.ListGroupItem([html.Strong("Commission: "), f"${config.commission_per_contract}"]),
            dbc.ListGroupItem([html.Strong("Slippage: "), f"{config.slippage_ticks} ticks"]),
            dbc.ListGroupItem([html.Strong("Tick Size: "), f"${config.tick_size}"]),
        ]

        # Risk management card
        risk_items = [
            dbc.ListGroupItem([html.Strong("Risk Manager: "), "Enabled" if config.use_risk_manager else "Disabled"]),
            dbc.ListGroupItem([html.Strong("Risk Per Trade: "), f"{config.risk_per_trade * 100:.1f}%"]),
            dbc.ListGroupItem([html.Strong("Daily Loss Limit: "), f"{config.daily_loss_limit * 100:.1f}%"]),
            dbc.ListGroupItem([html.Strong("Max Position: "), f"{config.max_position_contracts} contracts"]),
        ]

        # Recommendations
        if recommendations:
            rec_items = [
                dbc.ListGroupItem(rec, color="warning") for rec in recommendations
            ]
        else:
            rec_items = [dbc.ListGroupItem("No recommendations at this time.", color="success")]

        return dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [
                                                html.I(className="fas fa-cog me-2"),
                                                "Strategy Parameters",
                                            ]
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                [
                                                    html.Strong("Strategy: "),
                                                    f"{self.result.strategy.name} ({self.result.strategy.__class__.__name__})",
                                                ],
                                                className="mb-3",
                                            ),
                                            dbc.ListGroup(strategy_items, flush=True),
                                        ]
                                    ),
                                ]
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Backtest Configuration")),
                                    dbc.CardBody(dbc.ListGroup(config_items, flush=True)),
                                ]
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Risk Management")),
                                    dbc.CardBody(dbc.ListGroup(risk_items, flush=True)),
                                ]
                            ),
                            md=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5(
                                [
                                    html.I(className="fas fa-lightbulb me-2"),
                                    "Recommendations for Optimization",
                                ]
                            )
                        ),
                        dbc.CardBody(dbc.ListGroup(rec_items, flush=True)),
                    ]
                ),
            ],
            fluid=True,
        )

    def _metric_card(
        self,
        title: str,
        value: str,
        subtitle: str,
        color: str = "primary",
    ) -> dbc.Card:
        """Create a metric display card.

        Args:
            title: Card title
            value: Main metric value
            subtitle: Additional context
            color: Bootstrap color name

        Returns:
            Dash Bootstrap Card component
        """
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H6(title, className="card-subtitle mb-2 text-muted"),
                    html.H3(value, className=f"card-title text-{color}"),
                    html.P(subtitle, className="card-text small text-muted mb-0"),
                ]
            ),
            className="h-100",
        )

    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register any interactive callbacks.

        Args:
            app: Dash application
        """
        # Currently no dynamic callbacks needed, but this is where they would go
        pass

    def show(self, port: int = 8050, debug: bool = False, open_browser: bool = True) -> None:
        """Open the visualization window in a browser.

        Args:
            port: Port number for the Dash server
            debug: Enable Dash debug mode
            open_browser: Automatically open browser
        """
        url = f"http://127.0.0.1:{port}"
        logger.info("starting_visualization_server", port=port, url=url)
        print(f"\n  Dashboard URL: {url}\n")

        if open_browser:
            # Open browser in a separate thread after short delay
            def open_browser_delayed():
                import os
                import subprocess
                import sys
                import time

                time.sleep(2.0)

                # Try multiple methods to open browser on Windows
                try:
                    if sys.platform == "win32":
                        # Use os.startfile on Windows (most reliable)
                        os.startfile(url)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", url], check=False)
                    else:
                        subprocess.run(["xdg-open", url], check=False)
                except Exception:
                    # Fallback to webbrowser module
                    webbrowser.open(url)

            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()

        # Run the server (blocking)
        self.app.run(debug=debug, port=port, host="127.0.0.1")

    def run_in_background(self, port: int = 8050) -> threading.Thread:
        """Run the visualization server in a background thread.

        Args:
            port: Port number for the Dash server

        Returns:
            Thread object running the server
        """
        logger.info("starting_visualization_server_background", port=port)

        def run_server():
            self.app.run(debug=False, port=port, host="127.0.0.1", use_reloader=False)

        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

        return server_thread


def show_backtest_results(
    result: BacktestResult,
    symbol: str = "MCL_FUT",
    port: int = 8050,
    open_browser: bool = True,
) -> None:
    """Convenience function to show backtest results.

    Args:
        result: BacktestResult object
        symbol: Symbol being backtested
        port: Port for the visualization server
        open_browser: Whether to open browser automatically
    """
    viewer = BacktestViewer(result, symbol)
    viewer.show(port=port, open_browser=open_browser)
