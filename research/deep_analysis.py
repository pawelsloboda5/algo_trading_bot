"""
Deep Analysis: Mean Reversion Strategy with Full Visualization
==============================================================
Runs the optimized mean reversion strategy and generates comprehensive reports.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.storage import DataStorage
from src.strategy.mean_reversion_strategy import (
    MeanReversionStrategy,
    MeanReversionConfig,
    create_optimized_config,
)
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.visualization.report import ResultSaver
from src.visualization.window import show_backtest_results

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
storage = DataStorage(DATA_DIR)


def get_front_month_symbol(date: datetime) -> str:
    """Get the front-month MCL symbol for a given date.

    MCL contract months: F(Jan), G(Feb), H(Mar), J(Apr), K(May), M(Jun),
                        N(Jul), Q(Aug), U(Sep), V(Oct), X(Nov), Z(Dec)
    """
    month_codes = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']

    # Front month is typically the next month (roll ~2 weeks before expiry)
    # Simplified: use current month + 1
    front_month = (date.month % 12) + 1
    year = date.year if front_month > date.month else date.year + 1
    year_suffix = year % 100  # e.g., 2025 -> 25

    return f"MCL{month_codes[front_month - 1]}{year_suffix}"


def load_clean_data(start_date: datetime, end_date: datetime, schema: str = "ohlcv-1m") -> pd.DataFrame:
    """Load and clean data, filtering for front-month outright futures only."""
    dfs = []
    current = start_date
    while current <= end_date:
        df = storage.load_dataframe("MCL_FUT", current, schema)
        if df is not None and not df.empty:
            # CRITICAL: Filter for single front-month contract only
            # Exclude calendar spreads (contain '-') and keep only outrights
            if 'symbol' in df.columns:
                # Get front month symbol for this date
                front_sym = get_front_month_symbol(current)

                # Filter: outright futures only (no spreads with '-')
                # AND the most liquid contract (highest volume)
                df_outrights = df[~df['symbol'].str.contains('-', na=False)]

                if not df_outrights.empty:
                    # Get the symbol with highest total volume for the day
                    vol_by_sym = df_outrights.groupby('symbol')['volume'].sum()
                    if len(vol_by_sym) > 0:
                        most_liquid = vol_by_sym.idxmax()
                        df = df_outrights[df_outrights['symbol'] == most_liquid].copy()

            dfs.append(df)
        current += timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_event").reset_index(drop=True)

    # Clean: Keep only reasonable MCL prices ($50-$90)
    df = df[(df['close'] > 50) & (df['close'] < 90)]

    # CRITICAL: Remove duplicate timestamps (keep last, which has most complete data)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.drop_duplicates(subset=['ts_event'], keep='last')

    # Remove extreme returns (> 2% in 1 minute is unrealistic)
    df['returns'] = df['close'].pct_change()
    df = df[(df['returns'].abs() < 0.02) | (df['returns'].isna())]

    # Set index to timestamp
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
    df = df.set_index('ts_event')

    return df


def run_deep_backtest(
    months: int = 6,
    initial_capital: float = 10_000.0,
    show_dashboard: bool = True,
    save_results: bool = True,
) -> None:
    """Run deep backtest with full visualization.

    Args:
        months: Number of months of historical data to test
        initial_capital: Starting capital
        show_dashboard: Whether to launch interactive dashboard
        save_results: Whether to save results to disk
    """
    print("=" * 80)
    print("DEEP ANALYSIS: Mean Reversion Strategy")
    print("=" * 80)

    # Load data
    end_date = datetime(2026, 1, 8)
    start_date = end_date - timedelta(days=months * 30)

    print(f"\nLoading {months} months of data ({start_date.date()} to {end_date.date()})...")
    data = load_clean_data(start_date, end_date, "ohlcv-1m")

    if data.empty:
        print("ERROR: No data loaded!")
        return

    print(f"Loaded {len(data):,} bars")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Create strategy with optimized config
    config = create_optimized_config()
    strategy = MeanReversionStrategy(config)

    print(f"\nStrategy Configuration:")
    print(f"  - Stop Loss: {config.stop_loss_cents} cents")
    print(f"  - Targets: {config.target_1_cents} / {config.target_2_cents} / {config.target_3_cents} cents")
    print(f"  - Scale-out: {config.target_1_pct}% / {config.target_2_pct}% / {config.target_3_pct}%")
    print(f"  - Breakeven trigger: {config.breakeven_trigger_cents} cents")
    print(f"  - Trailing offset: {config.trail_offset_cents} cents")
    print(f"  - Volatility filter: {config.min_volatility_percentile}-{config.max_volatility_percentile} percentile")

    # Create backtest engine
    engine_config = BacktestConfig(
        initial_capital=initial_capital,
        use_risk_manager=False,  # Raw strategy performance
        position_size=1,
    )
    engine = BacktestEngine(engine_config)

    print(f"\nRunning backtest with ${initial_capital:,.0f} capital...")

    # Run backtest
    result = engine.run(data, strategy)

    # Print results summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(result.metrics)

    # Save results if requested
    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mean_reversion_{timestamp}"

        print(f"\nSaving results to: {RESULTS_DIR / run_name}")

        saver = ResultSaver(RESULTS_DIR)
        output_dir = saver.save(
            result=result,
            symbol="MCL_FUT",
            run_id=run_name,
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"  - summary.json (metrics)")
        print(f"  - trades.csv (all trades)")
        print(f"  - equity_curve.csv")
        print(f"  - charts/ (HTML charts)")
        print(f"  - report.html (full HTML report)")

    # Launch interactive dashboard if requested
    if show_dashboard:
        print("\n" + "=" * 80)
        print("LAUNCHING INTERACTIVE DASHBOARD")
        print("=" * 80)
        print("Opening browser... (Press Ctrl+C to stop)")

        show_backtest_results(
            result=result,
            symbol="MCL_FUT",
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run deep analysis with visualization")
    parser.add_argument("--months", type=int, default=6, help="Months of data to test (default: 6)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital (default: 10000)")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip interactive dashboard")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results")

    args = parser.parse_args()

    run_deep_backtest(
        months=args.months,
        initial_capital=args.capital,
        show_dashboard=not args.no_dashboard,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()
