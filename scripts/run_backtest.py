"""Run backtest on historical data."""

import sys
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.backtest.engine import BacktestConfig, BacktestEngine
from src.data.storage import DataStorage
from src.strategy.momentum_strategy import MACDMomentumStrategy, MomentumStrategy
from src.strategy.scalping_strategy import ScalpingConfig, ScalpingStrategy

logger = get_logger(__name__)


# Default EMA periods by timeframe (optimized for each granularity)
# 1s: Very fast scalping (seconds to minutes hold time)
# 1m: Scalping/intraday (minutes to hours hold time)
# 1h: Swing trading (hours to days hold time)
# 1d: Position trading (days to weeks hold time)
DEFAULT_PERIODS = {
    "1s": {"fast": 60, "slow": 180, "trend": 600},  # 1-3-10 minute EMAs
    "1m": {"fast": 5, "slow": 15, "trend": 60},  # 5-15-60 minute EMAs
    "1h": {"fast": 8, "slow": 21, "trend": 100},  # 8-21-100 hour EMAs
    "1d": {"fast": 10, "slow": 30, "trend": 200},  # 10-30-200 day EMAs
}

# Map timeframe to schema name
TIMEFRAME_TO_SCHEMA = {
    "1s": "ohlcv-1s",
    "1m": "ohlcv-1m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}


@click.command()
@click.option(
    "--symbol",
    default="MCL_FUT",
    help="Symbol to backtest (default: MCL_FUT)",
)
@click.option(
    "--timeframe",
    type=click.Choice(["1s", "1m", "1h", "1d"]),
    default="1m",
    help="Data timeframe: 1s (second), 1m (minute), 1h (hour), 1d (day). Default: 1m",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--strategy",
    type=click.Choice(["ema", "macd", "scalp"]),
    default="ema",
    help="Strategy type: ema, macd, or scalp (default: ema)",
)
@click.option(
    "--fast-period",
    type=int,
    default=None,
    help="Fast EMA period (default: auto based on timeframe)",
)
@click.option(
    "--slow-period",
    type=int,
    default=None,
    help="Slow EMA period (default: auto based on timeframe)",
)
@click.option(
    "--capital",
    type=float,
    default=None,
    help="Initial capital (default: from settings)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for results (CSV)",
)
@click.option(
    "--show",
    is_flag=True,
    help="Open interactive visualization window after backtest",
)
@click.option(
    "--save-results/--no-save-results",
    default=True,
    help="Save detailed results to files (default: True)",
)
@click.option(
    "--results-dir",
    type=click.Path(),
    default="results/backtests",
    help="Directory for saving results (default: results/backtests)",
)
def main(
    symbol: str,
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
    strategy: str,
    fast_period: int | None,
    slow_period: int | None,
    capital: float | None,
    output: str | None,
    show: bool,
    save_results: bool,
    results_dir: str,
) -> None:
    """Run backtest on historical MCL data.

    Examples:
        # Basic backtest with 1-minute data (default)
        python scripts/run_backtest.py

        # High-frequency backtest with 1-second data
        python scripts/run_backtest.py --timeframe 1s

        # Swing trading with hourly data
        python scripts/run_backtest.py --timeframe 1h

        # Custom date range and strategy parameters
        python scripts/run_backtest.py --start 2025-01-01 --end 2025-06-01 --fast-period 10 --slow-period 30

        # MACD strategy with visualization
        python scripts/run_backtest.py --strategy macd --show

        # Run without saving detailed results
        python scripts/run_backtest.py --no-save-results
    """
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)

    initial_capital = capital or settings.initial_capital

    # Get default periods for timeframe if not specified
    defaults = DEFAULT_PERIODS[timeframe]
    fast = fast_period or defaults["fast"]
    slow = slow_period or defaults["slow"]
    trend = defaults["trend"]

    # Get schema for timeframe
    schema = TIMEFRAME_TO_SCHEMA[timeframe]

    click.echo("=" * 60)
    click.echo("ALGO TRADING BOT - BACKTEST")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"Symbol: {symbol}")
    click.echo(f"Timeframe: {timeframe}")
    click.echo(f"Strategy: {strategy.upper()}")
    click.echo(f"Initial Capital: ${initial_capital:,.2f}")
    click.echo(f"Fast Period: {fast}")
    click.echo(f"Slow Period: {slow}")
    click.echo()

    # Load data
    storage = DataStorage(settings.raw_data_dir)

    click.echo("Loading historical data...")

    # Try to load data from storage
    available_dates = storage.list_available_dates(symbol, schema)

    if not available_dates:
        click.echo(
            f"No {schema} data found for {symbol}. Run download_historical.py first.", err=True
        )
        click.echo()
        click.echo("Example:")
        click.echo(f"  python scripts/download_historical.py --symbol {symbol} --schema {schema}")
        click.echo()
        click.echo("Available schemas:")
        for tf, sch in TIMEFRAME_TO_SCHEMA.items():
            dates = storage.list_available_dates(symbol, sch)
            if dates:
                click.echo(f"  {tf}: {sch} ({len(dates)} days: {dates[0].date()} to {dates[-1].date()})")
            else:
                click.echo(f"  {tf}: {sch} (no data)")
        sys.exit(1)

    # Filter dates
    if start:
        available_dates = [d for d in available_dates if d >= start]
    if end:
        available_dates = [d for d in available_dates if d <= end]

    if not available_dates:
        click.echo("No data in specified date range.", err=True)
        sys.exit(1)

    click.echo(f"Date range: {available_dates[0].date()} to {available_dates[-1].date()}")
    click.echo(f"Trading days: {len(available_dates)}")

    # Load all data
    dfs = []
    with click.progressbar(available_dates, label="Loading data") as dates:
        for date in dates:
            df = storage.load_dataframe(symbol, date, schema)
            if df is not None:
                dfs.append(df)

    if not dfs:
        click.echo("Failed to load any data.", err=True)
        sys.exit(1)

    data = pd.concat(dfs, ignore_index=True)

    # Prepare data for backtest
    # Databento uses 'ts_event' as timestamp
    if "ts_event" in data.columns:
        data["ts_event"] = pd.to_datetime(data["ts_event"])
        data = data.set_index("ts_event")

    # Normalize column names
    data.columns = data.columns.str.lower()

    # Filter for primary contract only (exclude spreads and back-month contracts)
    # MCL data contains multiple symbols (MCLF6, MCLG6, spreads like MCLF6-MCLG6)
    if "symbol" in data.columns:
        # Get unique symbols and filter for front-month only
        unique_symbols = data["symbol"].unique()
        # Front-month contracts are simple symbols like MCLF6, MCLG6 (not spreads with "-")
        front_month_symbols = [s for s in unique_symbols if "-" not in str(s)]
        if front_month_symbols:
            # Use the first (most common) front-month contract
            primary_symbol = data[data["symbol"].isin(front_month_symbols)]["symbol"].mode().iloc[0]
            original_len = len(data)
            data = data[data["symbol"] == primary_symbol]
            click.echo(f"Filtered to primary contract: {primary_symbol} ({len(data):,} of {original_len:,} bars)")

    # Ensure OHLCV columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(data.columns)
    if missing:
        click.echo(f"Missing columns: {missing}", err=True)
        sys.exit(1)

    # Sort and remove duplicate timestamps
    data = data.sort_index()
    if data.index.duplicated().any():
        data = data[~data.index.duplicated(keep="first")]
        click.echo(f"Removed duplicate timestamps, final bars: {len(data):,}")
    click.echo(f"Total bars: {len(data):,}")
    click.echo()

    # Create strategy
    if strategy == "ema":
        strat = MomentumStrategy(
            fast_period=fast,
            slow_period=slow,
            atr_period=settings.atr_period,
            trend_filter_period=trend if timeframe in ["1h", "1d"] else None,  # Disable trend filter for HFT
        )
    elif strategy == "macd":
        strat = MACDMomentumStrategy(
            fast_period=fast,
            slow_period=slow,
        )
    else:  # scalp
        # Configure scalping strategy for the timeframe
        # Use balanced defaults - not too strict, not too loose
        scalp_config = ScalpingConfig(
            momentum_period=8 if timeframe == "1s" else 5,
            momentum_threshold=0.02 if timeframe == "1s" else 0.015,
            volume_ma_period=20 if timeframe == "1s" else 10,
            volume_accel_threshold=1.3,  # 30% above average
            volatility_lookback=100 if timeframe == "1s" else 50,
            volatility_min_pct=15.0,
            volatility_max_pct=85.0,
            trend_period=15 if timeframe == "1s" else 8,
            atr_period=14 if timeframe == "1s" else 14,
            atr_stop_multiplier=2.0,  # Wider stop to avoid whipsaws
            target_1_atr=0.75,  # Take profits earlier
            target_2_atr=1.5,
            target_3_atr=2.5,
            breakeven_trigger_atr=0.4,
            max_hold_seconds=120 if timeframe == "1s" else 600,
            use_session_filter=True,
        )
        strat = ScalpingStrategy(config=scalp_config)

    click.echo(f"Strategy: {strat}")
    click.echo()

    # Configure backtest with IBKR commission structure
    # MCL (Micro WTI): $0.85 IBKR + $0.50 exchange + $0.02 regulatory = $1.37/contract/side
    # Round trip = $2.74/contract
    config = BacktestConfig(
        initial_capital=initial_capital,
        contract_multiplier=settings.contract_multiplier,
        tick_size=settings.tick_size,
        position_size=1,
        slippage_ticks=1,
        # IBKR Commission Structure for MCL (NYMEX Tier 1)
        ibkr_commission=0.85,  # Tiered rate (≤1000 contracts/month)
        exchange_fee=0.50,  # NYMEX fee recovery for MCL
        regulatory_fee=0.02,  # NYMEX regulatory fee
        give_up_fee=0.00,  # Waived when IBKR is prime+executing
    )

    # Display commission info
    click.echo(f"Commission: ${config.commission_per_contract:.2f}/contract/side (IBKR + NYMEX fees)")
    click.echo(f"Round trip: ${config.commission_per_contract * 2:.2f}/contract")
    click.echo()

    # Run backtest
    click.echo("Running backtest...")
    engine = BacktestEngine(config)

    try:
        result = engine.run(data, strat)
    except Exception as e:
        click.echo(f"Backtest failed: {e}", err=True)
        logger.exception("backtest_failed")
        sys.exit(1)

    # Display results
    click.echo()
    click.echo(str(result.metrics))

    # Trade summary
    if not result.trades.empty:
        click.echo()
        click.echo("RECENT TRADES (last 10):")
        click.echo("-" * 60)
        recent = result.trades.tail(10)[
            ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl"]
        ]
        click.echo(recent.to_string())

    # Save legacy CSV output if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save trades
        result.trades.to_csv(output_path, index=False)
        click.echo()
        click.echo(f"Trades saved to: {output_path}")

        # Save equity curve
        equity_path = output_path.with_suffix(".equity.csv")
        result.equity_curve.to_csv(equity_path)
        click.echo(f"Equity curve saved to: {equity_path}")

    # Save detailed results
    run_dir = None
    if save_results:
        click.echo()
        click.echo("Saving detailed results...")

        try:
            from src.visualization.report import ResultSaver

            saver = ResultSaver(results_dir)
            run_dir = saver.save(result, symbol=symbol)
            click.echo(f"Results saved to: {run_dir}")
            click.echo()
            click.echo("Files created:")
            click.echo(f"  - {run_dir / 'summary.json'} (AI-friendly metrics)")
            click.echo(f"  - {run_dir / 'trades.csv'}")
            click.echo(f"  - {run_dir / 'equity_curve.csv'}")
            click.echo(f"  - {run_dir / 'daily_pnl.csv'}")
            click.echo(f"  - {run_dir / 'parameters.json'}")
            click.echo(f"  - {run_dir / 'report.html'}")
            click.echo(f"  - {run_dir / 'charts/'} (interactive charts)")
        except ImportError as e:
            click.echo(f"Warning: Could not save results - missing dependencies: {e}", err=True)
            click.echo("Install with: pip install dash dash-bootstrap-components", err=True)
        except Exception as e:
            click.echo(f"Warning: Could not save results: {e}", err=True)
            logger.exception("save_results_failed")

    # Open visualization window
    if show:
        click.echo()
        click.echo("Opening visualization window...")
        click.echo("Press Ctrl+C to stop the server and exit.")
        click.echo()

        try:
            from src.visualization.window import show_backtest_results

            show_backtest_results(result, symbol=symbol)
        except ImportError as e:
            click.echo(f"Error: Could not open visualization - missing dependencies: {e}", err=True)
            click.echo("Install with: pip install dash dash-bootstrap-components", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error opening visualization: {e}", err=True)
            logger.exception("visualization_failed")
            sys.exit(1)
    else:
        click.echo()
        click.echo("Backtest complete!")
        if save_results and run_dir:
            click.echo()
            click.echo("To view interactive visualization, run:")
            click.echo(f"  python scripts/run_backtest.py --symbol {symbol} --timeframe {timeframe} --show")
            click.echo()
            click.echo(f"Or open the HTML report: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
