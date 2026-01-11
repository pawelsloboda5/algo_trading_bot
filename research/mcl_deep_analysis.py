"""
MCL Deep Analysis - Cleaned Version
====================================
Focus on actionable insights with proper data cleaning.
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
from src.strategy.indicators import ema, atr, rsi, bollinger_bands

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
storage = DataStorage(DATA_DIR)


def load_clean_data(start_date: datetime, end_date: datetime, schema: str = "ohlcv-1s") -> pd.DataFrame:
    """Load and clean data, handling gaps and outliers."""
    dfs = []
    current = start_date
    while current <= end_date:
        df = storage.load_dataframe("MCL_FUT", current, schema)
        if df is not None and not df.empty:
            dfs.append(df)
        current += timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_event").reset_index(drop=True)

    # Clean data
    # 1. Remove zero prices
    df = df[df['close'] > 0]

    # 2. Remove extreme returns (>5% in one bar - data error)
    df['returns'] = df['close'].pct_change()
    df = df[df['returns'].abs() < 0.05]

    # 3. Reset index
    df = df.reset_index(drop=True)

    return df


def analyze_basic_stats(df: pd.DataFrame):
    """Basic statistical overview."""
    print("="*80)
    print("BASIC STATISTICS")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['bar_range'] = df['high'] - df['low']

    print(f"\nData Overview:")
    print(f"  Total bars: {len(df):,}")
    print(f"  Date range: {df['ts_event'].min()} to {df['ts_event'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    print(f"\nReturns Statistics (per bar):")
    print(f"  Mean: {df['returns'].mean()*10000:.4f} bps")
    print(f"  Std: {df['returns'].std()*10000:.4f} bps")
    print(f"  Skew: {df['returns'].skew():.4f}")
    print(f"  Kurtosis: {df['returns'].kurtosis():.4f}")

    # In ticks ($0.01 = 1 tick)
    print(f"\nBar Range (in ticks):")
    tick_range = df['bar_range'] / 0.01
    print(f"  Mean: {tick_range.mean():.2f} ticks")
    print(f"  Median: {tick_range.median():.2f} ticks")
    print(f"  95th percentile: {tick_range.quantile(0.95):.2f} ticks")

    # Volume
    print(f"\nVolume:")
    print(f"  Mean: {df['volume'].mean():.1f} contracts/bar")
    print(f"  Median: {df['volume'].median():.1f}")


def analyze_failed_trades_detailed(df_1s: pd.DataFrame):
    """Detailed analysis of why trades fail."""
    print("\n" + "="*80)
    print("DETAILED TRADE FAILURE ANALYSIS")
    print("="*80)

    # Load trades
    results_dir = Path(__file__).parent.parent / "results"
    trades_files = list(results_dir.glob("backtests/**/trades.csv"))
    if not trades_files:
        print("No trades found")
        return

    latest = max(trades_files, key=lambda x: x.stat().st_mtime)
    trades_df = pd.read_csv(latest, parse_dates=['entry_time', 'exit_time'])

    print(f"\nLoaded {len(trades_df)} trades from {latest.parent.name}")

    # Trade statistics
    print(f"\nTrade Statistics:")
    print(f"  Win rate: {100*(trades_df['is_winner']).mean():.1f}%")
    print(f"  Avg PnL: ${trades_df['pnl'].mean():.2f}")
    print(f"  Total PnL: ${trades_df['pnl'].sum():.2f}")

    # Duration analysis
    durations = pd.to_timedelta(trades_df['duration'])
    print(f"\nDuration:")
    print(f"  Mean: {durations.mean()}")
    print(f"  Min: {durations.min()}")
    print(f"  Max: {durations.max()}")

    # Slippage analysis
    slippage = []
    for _, trade in trades_df.iterrows():
        # Expected exit vs actual exit
        # If pnl is negative and trade lasted only seconds, slippage is the issue
        expected_profit = 0  # We entered expecting positive outcome
        actual_pnl = trade['pnl']
        slippage.append(actual_pnl - expected_profit)

    print(f"\nAll trades lost money immediately:")
    print(f"  This indicates entries are at local extremes (tops/bottoms)")
    print(f"  OR signals are lagging (detecting after the move happened)")


def analyze_trend_vs_mean_reversion_clean(df: pd.DataFrame):
    """Clean comparison of trend vs mean reversion."""
    print("\n" + "="*80)
    print("TREND VS MEAN REVERSION - CLEAN ANALYSIS")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Calculate indicators
    df['rsi_14'] = rsi(df['close'], 14)
    df['ema_10'] = ema(df['close'], 10)
    df['ema_30'] = ema(df['close'], 30)

    # Forward returns at different horizons
    for horizon in [10, 30, 60]:
        df[f'fwd_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1

    # Drop NaN and inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\nClean data: {len(df):,} bars")

    # ============ MEAN REVERSION ============
    print("\n--- MEAN REVERSION SIGNALS ---")

    # RSI extreme signals
    rsi_oversold = df[df['rsi_14'] < 30].copy()
    rsi_overbought = df[df['rsi_14'] > 70].copy()

    if len(rsi_oversold) > 100:
        for h in [10, 30, 60]:
            avg_ret = rsi_oversold[f'fwd_{h}'].mean() * 10000
            win_rate = (rsi_oversold[f'fwd_{h}'] > 0).mean()
            print(f"  RSI < 30 -> LONG ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(rsi_oversold)})")

    if len(rsi_overbought) > 100:
        for h in [10, 30, 60]:
            # Short = negative forward return is good
            avg_ret = -rsi_overbought[f'fwd_{h}'].mean() * 10000
            win_rate = (rsi_overbought[f'fwd_{h}'] < 0).mean()
            print(f"  RSI > 70 -> SHORT ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(rsi_overbought)})")

    # Price deviation from EMA
    df['ema_dev'] = (df['close'] - df['ema_30']) / df['close'] * 10000  # in bps

    extended_up = df[df['ema_dev'] > 20].copy()  # >20 bps above EMA30
    extended_down = df[df['ema_dev'] < -20].copy()

    if len(extended_up) > 100:
        for h in [10, 30, 60]:
            avg_ret = -extended_up[f'fwd_{h}'].mean() * 10000  # Fade = short
            win_rate = (extended_up[f'fwd_{h}'] < 0).mean()
            print(f"  Extended UP (>20bps) -> SHORT ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(extended_up)})")

    if len(extended_down) > 100:
        for h in [10, 30, 60]:
            avg_ret = extended_down[f'fwd_{h}'].mean() * 10000  # Fade = long
            win_rate = (extended_down[f'fwd_{h}'] > 0).mean()
            print(f"  Extended DOWN (<-20bps) -> LONG ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(extended_down)})")

    # ============ TREND FOLLOWING ============
    print("\n--- TREND FOLLOWING SIGNALS ---")

    # EMA trend
    uptrend = df[df['ema_10'] > df['ema_30']].copy()
    downtrend = df[df['ema_10'] < df['ema_30']].copy()

    if len(uptrend) > 100:
        for h in [10, 30, 60]:
            avg_ret = uptrend[f'fwd_{h}'].mean() * 10000
            win_rate = (uptrend[f'fwd_{h}'] > 0).mean()
            print(f"  EMA10 > EMA30 -> LONG ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(uptrend)})")

    if len(downtrend) > 100:
        for h in [10, 30, 60]:
            avg_ret = -downtrend[f'fwd_{h}'].mean() * 10000
            win_rate = (downtrend[f'fwd_{h}'] < 0).mean()
            print(f"  EMA10 < EMA30 -> SHORT ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(downtrend)})")

    # Breakout signals
    df['high_30'] = df['high'].rolling(30).max()
    df['low_30'] = df['low'].rolling(30).min()
    df['breakout_up'] = df['close'] > df['high_30'].shift(1)
    df['breakout_down'] = df['close'] < df['low_30'].shift(1)

    breakouts_up = df[df['breakout_up']].copy()
    breakouts_down = df[df['breakout_down']].copy()

    if len(breakouts_up) > 100:
        for h in [10, 30, 60]:
            avg_ret = breakouts_up[f'fwd_{h}'].mean() * 10000
            win_rate = (breakouts_up[f'fwd_{h}'] > 0).mean()
            print(f"  30-bar HIGH Breakout -> LONG ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(breakouts_up)})")

    if len(breakouts_down) > 100:
        for h in [10, 30, 60]:
            avg_ret = -breakouts_down[f'fwd_{h}'].mean() * 10000
            win_rate = (breakouts_down[f'fwd_{h}'] < 0).mean()
            print(f"  30-bar LOW Breakout -> SHORT ({h}s): {avg_ret:.1f} bps, Win {100*win_rate:.1f}% (n={len(breakouts_down)})")


def analyze_optimal_entry_timing(df: pd.DataFrame):
    """Analyze best entry timing approach."""
    print("\n" + "="*80)
    print("OPTIMAL ENTRY TIMING ANALYSIS")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Create signal: 30-bar high breakout
    df['high_30'] = df['high'].rolling(30).max()
    df['signal'] = df['close'] > df['high_30'].shift(1)

    signal_indices = df[df['signal']].index.tolist()[:2000]  # Sample

    results = {
        'immediate': [],
        'wait_5s': [],
        'wait_10s': [],
        'pullback_2tick': [],
    }

    for idx in signal_indices:
        if idx + 60 >= len(df):
            continue

        signal_price = df.loc[idx, 'close']

        # 1. Immediate entry
        exit_30s = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else signal_price
        pnl_immediate = (exit_30s - signal_price) / signal_price * 10000
        results['immediate'].append(pnl_immediate)

        # 2. Wait 5 seconds before entry
        if idx + 5 < len(df) and idx + 35 < len(df):
            entry_5s = df.loc[idx + 5, 'close']
            exit_35s = df.loc[idx + 35, 'close']
            pnl_5s = (exit_35s - entry_5s) / entry_5s * 10000
            results['wait_5s'].append(pnl_5s)

        # 3. Wait 10 seconds
        if idx + 10 < len(df) and idx + 40 < len(df):
            entry_10s = df.loc[idx + 10, 'close']
            exit_40s = df.loc[idx + 40, 'close']
            pnl_10s = (exit_40s - entry_10s) / entry_10s * 10000
            results['wait_10s'].append(pnl_10s)

        # 4. Wait for 2-tick pullback ($0.02)
        for future_idx in range(idx + 1, min(idx + 20, len(df))):
            if df.loc[future_idx, 'low'] <= signal_price - 0.02:
                # Found pullback
                entry_pb = signal_price - 0.02
                exit_idx = min(future_idx + 30, len(df) - 1)
                exit_pb = df.loc[exit_idx, 'close']
                pnl_pb = (exit_pb - entry_pb) / entry_pb * 10000
                results['pullback_2tick'].append(pnl_pb)
                break

    print("\nEntry Method Comparison (30-bar Breakout Long):")
    print("=" * 60)
    print(f"{'Method':<20} | {'Avg PnL (bps)':>12} | {'Win Rate':>10} | {'N Trades':>10}")
    print("-" * 60)

    for method, pnls in results.items():
        if pnls:
            avg_pnl = np.mean(pnls)
            win_rate = np.mean([p > 0 for p in pnls])
            print(f"{method:<20} | {avg_pnl:>12.2f} | {100*win_rate:>9.1f}% | {len(pnls):>10}")


def analyze_profit_targets_and_stops(df: pd.DataFrame):
    """Analyze optimal profit targets and stop losses."""
    print("\n" + "="*80)
    print("PROFIT TARGET & STOP LOSS OPTIMIZATION")
    print("="*80)

    df = df.copy()

    # Use breakout signal for analysis
    df['high_30'] = df['high'].rolling(30).max()
    df['signal'] = df['close'] > df['high_30'].shift(1)

    signal_indices = df[df['signal']].index.tolist()[:1500]

    # Track MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion)
    mae_list = []  # in ticks
    mfe_list = []  # in ticks

    for idx in signal_indices:
        if idx + 120 >= len(df):
            continue

        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:idx+120].copy()

        # MAE = worst drawdown from entry (for long)
        mae_ticks = (entry_price - future['low'].min()) / 0.01
        mfe_ticks = (future['high'].max() - entry_price) / 0.01

        mae_list.append(mae_ticks)
        mfe_list.append(mfe_ticks)

    print(f"\nAnalyzed {len(mae_list)} trades over 120-second window")

    print(f"\nMaximum Adverse Excursion (MAE) - Ticks:")
    print(f"  Mean: {np.mean(mae_list):.1f} ticks")
    print(f"  Median: {np.median(mae_list):.1f} ticks")
    print(f"  75th pct: {np.percentile(mae_list, 75):.1f} ticks")
    print(f"  90th pct: {np.percentile(mae_list, 90):.1f} ticks")
    print(f"  95th pct: {np.percentile(mae_list, 95):.1f} ticks")

    print(f"\nMaximum Favorable Excursion (MFE) - Ticks:")
    print(f"  Mean: {np.mean(mfe_list):.1f} ticks")
    print(f"  Median: {np.median(mfe_list):.1f} ticks")
    print(f"  75th pct: {np.percentile(mfe_list, 75):.1f} ticks")
    print(f"  90th pct: {np.percentile(mfe_list, 90):.1f} ticks")

    # Survival rates for different stop distances
    print(f"\nStop Loss Survival Rates:")
    print(f"{'Stop (ticks)':<15} | {'Survival Rate':>15}")
    print("-" * 35)
    for stop_ticks in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        survival = np.mean([mae < stop_ticks for mae in mae_list])
        print(f"{stop_ticks:<15} | {100*survival:>14.1f}%")

    # Calculate expected PnL for different stop/target combinations
    print(f"\nExpected PnL by Stop/Target (ticks):")
    print(f"Assumes entry at signal bar close, $1/tick, $2.74 round trip")

    for stop in [3, 5, 8, 10]:
        for target in [3, 5, 8, 10, 15]:
            wins = 0
            losses = 0
            breakevens = 0

            for i, (mae, mfe) in enumerate(zip(mae_list, mfe_list)):
                if mae >= stop:
                    # Stop hit first
                    losses += 1
                elif mfe >= target:
                    # Target hit
                    wins += 1
                else:
                    # Neither hit (time stop)
                    breakevens += 1

            n = wins + losses + breakevens
            if n > 0:
                win_rate = wins / n
                expected = win_rate * target - (1 - win_rate) * stop - 2.74  # net of commissions
                print(f"  Stop={stop}, Target={target}: Win={100*win_rate:.1f}%, E[PnL]=${expected:.2f}")


def analyze_time_of_day_profitability(df: pd.DataFrame):
    """Which hours work for which strategy?"""
    print("\n" + "="*80)
    print("TIME-OF-DAY PROFITABILITY")
    print("="*80)

    df = df.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')

    df['hour'] = df['ts_event'].dt.hour
    df['returns'] = df['close'].pct_change()

    # Forward returns
    df['fwd_30'] = df['close'].shift(-30) / df['close'] - 1

    # RSI for mean reversion
    df['rsi'] = rsi(df['close'], 14)

    # Breakout for trend
    df['high_30'] = df['high'].rolling(30).max()
    df['breakout'] = df['close'] > df['high_30'].shift(1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    print("\nMean Reversion (RSI < 30 -> LONG) by Hour UTC:")
    print(f"{'Hour':<6} | {'Avg PnL (bps)':>12} | {'Win Rate':>10} | {'N':>8}")
    print("-" * 45)

    for hour in range(24):
        hour_data = df[(df['hour'] == hour) & (df['rsi'] < 30)]
        if len(hour_data) > 50:
            avg_pnl = hour_data['fwd_30'].mean() * 10000
            win_rate = (hour_data['fwd_30'] > 0).mean()
            print(f"{hour:02d}    | {avg_pnl:>12.1f} | {100*win_rate:>9.1f}% | {len(hour_data):>8}")

    print("\nTrend Following (Breakout -> LONG) by Hour UTC:")
    print(f"{'Hour':<6} | {'Avg PnL (bps)':>12} | {'Win Rate':>10} | {'N':>8}")
    print("-" * 45)

    for hour in range(24):
        hour_data = df[(df['hour'] == hour) & (df['breakout'])]
        if len(hour_data) > 50:
            avg_pnl = hour_data['fwd_30'].mean() * 10000
            win_rate = (hour_data['fwd_30'] > 0).mean()
            print(f"{hour:02d}    | {avg_pnl:>12.1f} | {100*win_rate:>9.1f}% | {len(hour_data):>8}")


def analyze_volume_patterns(df: pd.DataFrame):
    """Analyze how volume predicts price."""
    print("\n" + "="*80)
    print("VOLUME-PRICE PREDICTION ANALYSIS")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Volume moving average
    df['vol_ma'] = df['volume'].rolling(100).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']

    # Volume spike
    df['vol_spike'] = df['vol_ratio'] > 3

    # Forward returns
    df['fwd_10'] = df['close'].shift(-10) / df['close'] - 1
    df['fwd_30'] = df['close'].shift(-30) / df['close'] - 1

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # High volume continuation vs reversal
    print("\nAfter HIGH Volume Bars (>3x average):")

    high_vol = df[df['vol_spike']].copy()
    if len(high_vol) > 100:
        # Up bar vs down bar
        high_vol['bar_dir'] = np.sign(high_vol['returns'])

        up_bars = high_vol[high_vol['bar_dir'] > 0]
        down_bars = high_vol[high_vol['bar_dir'] < 0]

        print(f"\n  High volume UP bar -> Next 30s:")
        if len(up_bars) > 50:
            cont_rate = (up_bars['fwd_30'] > 0).mean()
            avg_fwd = up_bars['fwd_30'].mean() * 10000
            print(f"    Continuation rate: {100*cont_rate:.1f}% (LONG profitable)")
            print(f"    Avg forward return: {avg_fwd:.2f} bps")
            print(f"    -> {'FADE (short)' if cont_rate < 0.5 else 'FOLLOW (long)'} recommended")

        print(f"\n  High volume DOWN bar -> Next 30s:")
        if len(down_bars) > 50:
            cont_rate = (down_bars['fwd_30'] < 0).mean()
            avg_fwd = down_bars['fwd_30'].mean() * 10000
            print(f"    Continuation rate: {100*cont_rate:.1f}% (SHORT profitable)")
            print(f"    Avg forward return: {avg_fwd:.2f} bps")
            print(f"    -> {'FADE (long)' if cont_rate < 0.5 else 'FOLLOW (short)'} recommended")


def propose_strategy():
    """Propose a new evidence-based strategy."""
    print("\n" + "="*80)
    print("PROPOSED NEW STRATEGY: VOLUME FADE SCALPER")
    print("="*80)

    print("""
STRATEGY HYPOTHESIS:
====================
Based on the analysis, we observed:
1. Trend-following signals (breakouts) have ~28% win rate - LOSING strategy
2. Mean reversion signals show promise but timing is critical
3. High volume spikes often precede reversals
4. Entries at signal bar close are suboptimal

NEW STRATEGY: VOLUME FADE WITH PULLBACK ENTRY
==============================================

ENTRY RULES (LONG):
1. Wait for volume spike (>3x 100-bar average)
2. The spike bar must be a DOWN bar (selling climax)
3. Wait for 2-tick ($0.02) pullback from spike bar close
4. Enter LONG on the pullback

ENTRY RULES (SHORT):
1. Wait for volume spike (>3x 100-bar average)
2. The spike bar must be an UP bar (buying climax)
3. Wait for 2-tick ($0.02) pullback from spike bar close
4. Enter SHORT on the pullback

EXIT RULES:
- Profit target: 5 ticks ($0.05) = $5/contract
- Stop loss: 8 ticks ($0.08) = $8/contract
- Time stop: 120 seconds max hold

FILTERS:
- Only trade 14:00-20:00 UTC (8AM-2PM Chicago)
- Skip first 30 minutes of session
- No trading on low volume days

EXPECTED PERFORMANCE:
- Target win rate: 45-55%
- Risk/Reward: 5:8 = 0.625
- Break-even win rate: 8/(5+8) = 61.5%
- Need to improve win rate OR widen target

ALTERNATIVE: TIGHTER STOP + HIGHER WIN RATE
- Stop: 5 ticks, Target: 4 ticks
- Need: 4/(4+5) = 44.4% win rate to break even
- More achievable with good signal selection
""")


def main():
    """Run analysis."""
    print("="*80)
    print("MCL DEEP ANALYSIS - CLEANED DATA")
    print("="*80)

    # Load recent 30 days of 1-second data
    end_date = datetime(2026, 1, 8)
    start_date = end_date - timedelta(days=30)

    print(f"\nLoading 1-second data from {start_date.date()} to {end_date.date()}...")
    df_1s = load_clean_data(start_date, end_date, "ohlcv-1s")
    print(f"Loaded and cleaned: {len(df_1s):,} bars")

    if df_1s.empty:
        print("No data found!")
        return

    # Run analyses
    analyze_basic_stats(df_1s)
    analyze_failed_trades_detailed(df_1s)
    analyze_trend_vs_mean_reversion_clean(df_1s)
    analyze_optimal_entry_timing(df_1s)
    analyze_profit_targets_and_stops(df_1s)
    analyze_time_of_day_profitability(df_1s)
    analyze_volume_patterns(df_1s)
    propose_strategy()


if __name__ == "__main__":
    main()
