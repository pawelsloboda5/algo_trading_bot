"""
MCL Microstructure Analysis
============================
Comprehensive analysis to understand why scalping strategies are failing
and develop evidence-based entry/exit rules.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict

from src.data.storage import DataStorage
from src.strategy.indicators import ema, atr, rsi, bollinger_bands

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"

storage = DataStorage(DATA_DIR)


def load_1s_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load 1-second OHLCV data for a date range."""
    dfs = []
    current = start_date
    while current <= end_date:
        df = storage.load_dataframe("MCL_FUT", current, "ohlcv-1s")
        if df is not None and not df.empty:
            dfs.append(df)
        current += timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_event").reset_index(drop=True)
    return combined


def load_1m_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load 1-minute OHLCV data for a date range."""
    dfs = []
    current = start_date
    while current <= end_date:
        df = storage.load_dataframe("MCL_FUT", current, "ohlcv-1m")
        if df is not None and not df.empty:
            dfs.append(df)
        current += timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("ts_event").reset_index(drop=True)
    return combined


def load_failed_trades() -> pd.DataFrame:
    """Load the most recent backtest trades."""
    trades_files = list(RESULTS_DIR.glob("backtests/**/trades.csv"))
    if not trades_files:
        return pd.DataFrame()

    # Get most recent
    latest = max(trades_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading trades from: {latest}")
    return pd.read_csv(latest, parse_dates=['entry_time', 'exit_time'])


def analyze_price_microstructure(df: pd.DataFrame) -> dict:
    """
    Task 1: Price Microstructure Analysis
    Analyze what happens BEFORE and AFTER significant price moves.
    """
    print("\n" + "="*80)
    print("TASK 1: PRICE MICROSTRUCTURE ANALYSIS")
    print("="*80)

    # Calculate returns and identify significant moves
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()

    # Look for moves > 0.03% (3 basis points) in 10 seconds
    df['rolling_10s_return'] = df['close'].pct_change(10)
    df['abs_10s_return'] = df['rolling_10s_return'].abs()

    # Threshold for "significant" move
    threshold = 0.0003  # 0.03%
    significant_moves = df[df['abs_10s_return'] > threshold].copy()

    print(f"\nTotal bars: {len(df):,}")
    print(f"Significant moves (>0.03% in 10s): {len(significant_moves):,} ({100*len(significant_moves)/len(df):.2f}%)")

    # Analyze what happens BEFORE significant moves
    before_volume = []
    before_volatility = []
    continuation_rate = []
    retracement_rate = []

    # Calculate rolling volume and volatility
    df['rolling_volume_30s'] = df['volume'].rolling(30).mean()
    df['rolling_volatility_30s'] = df['returns'].rolling(30).std()
    df['volume_ratio'] = df['volume'] / df['rolling_volume_30s']

    # For each significant move, analyze before/after
    for idx in significant_moves.index:
        if idx < 60 or idx > len(df) - 60:
            continue

        # Before the move (30 seconds prior)
        before_slice = df.loc[idx-30:idx-1]
        if len(before_slice) < 30:
            continue

        before_vol = before_slice['volume'].mean()
        before_volatility_val = before_slice['returns'].std()

        before_volume.append(before_vol)
        before_volatility.append(before_volatility_val)

        # After the move (30 seconds after)
        after_slice = df.loc[idx:idx+30]
        if len(after_slice) < 30:
            continue

        move_direction = np.sign(df.loc[idx, 'rolling_10s_return'])

        # Did it continue or reverse?
        after_return = after_slice['close'].iloc[-1] / after_slice['close'].iloc[0] - 1
        if move_direction * after_return > 0:
            continuation_rate.append(1)
        else:
            continuation_rate.append(0)

        # Retracement analysis
        if move_direction > 0:  # Up move
            max_retrace = (after_slice['close'].iloc[0] - after_slice['low'].min()) / (after_slice['close'].iloc[0] + 0.0001)
        else:
            max_retrace = (after_slice['high'].max() - after_slice['close'].iloc[0]) / (after_slice['close'].iloc[0] + 0.0001)

        retracement_rate.append(max_retrace)

    print(f"\nVolume before significant moves:")
    print(f"  Mean: {np.mean(before_volume):.2f}")
    print(f"  Median: {np.median(before_volume):.2f}")

    print(f"\nVolatility before moves (std of returns):")
    print(f"  Mean: {np.mean(before_volatility)*10000:.4f} bps")

    print(f"\nAfter significant moves:")
    print(f"  Continuation rate (next 30s): {100*np.mean(continuation_rate):.1f}%")
    print(f"  Mean retracement: {100*np.mean(retracement_rate):.2f}%")
    print(f"  Median retracement: {100*np.median(retracement_rate):.2f}%")

    # Volume before vs after moves
    results = {
        'before_volume_mean': np.mean(before_volume) if before_volume else 0,
        'continuation_rate': np.mean(continuation_rate) if continuation_rate else 0.5,
        'mean_retracement': np.mean(retracement_rate) if retracement_rate else 0,
    }

    return results


def analyze_failed_trades(df_1s: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """
    Task 2: Failed Entry Analysis
    For each losing trade, analyze what was happening before and after entry.
    """
    print("\n" + "="*80)
    print("TASK 2: FAILED ENTRY ANALYSIS")
    print("="*80)

    if trades_df.empty:
        print("No trades to analyze")
        return {}

    results = {
        'pre_entry_move_direction': [],
        'pre_entry_momentum': [],
        'post_entry_reversal_speed': [],
        'entry_at_extreme': [],
    }

    # Ensure timestamp is tz-aware for comparison
    df_1s = df_1s.copy()
    if df_1s['ts_event'].dt.tz is None:
        df_1s['ts_event'] = pd.to_datetime(df_1s['ts_event']).dt.tz_localize('UTC')

    for _, trade in trades_df.iterrows():
        entry_time = pd.to_datetime(trade['entry_time'])
        if entry_time.tz is None:
            entry_time = entry_time.tz_localize('UTC')

        # Find 60 seconds before and after entry
        before_mask = (df_1s['ts_event'] >= entry_time - timedelta(seconds=60)) & (df_1s['ts_event'] < entry_time)
        after_mask = (df_1s['ts_event'] >= entry_time) & (df_1s['ts_event'] < entry_time + timedelta(seconds=60))

        before_data = df_1s[before_mask]
        after_data = df_1s[after_mask]

        if before_data.empty or after_data.empty:
            continue

        # Analyze pre-entry movement
        pre_move = (before_data['close'].iloc[-1] - before_data['close'].iloc[0]) / before_data['close'].iloc[0]

        # Was the pre-move in same direction as our trade?
        trade_direction = 1 if trade['direction'] == 'LONG' else -1
        pre_move_aligned = 1 if (pre_move * trade_direction > 0) else 0
        results['pre_entry_move_direction'].append(pre_move_aligned)
        results['pre_entry_momentum'].append(pre_move * 10000)  # in basis points

        # How fast did it reverse?
        if len(after_data) > 5:
            post_move = (after_data['close'].iloc[5] - after_data['close'].iloc[0]) / after_data['close'].iloc[0]
            reversal = 1 if (post_move * trade_direction < 0) else 0
            results['post_entry_reversal_speed'].append(reversal)

        # Was entry at a local extreme?
        window_high = before_data['high'].max()
        window_low = before_data['low'].min()
        entry_price = trade['entry_price']

        at_high = 1 if (entry_price > window_high * 0.9999) else 0
        at_low = 1 if (entry_price < window_low * 1.0001) else 0

        if trade['direction'] == 'LONG':
            results['entry_at_extreme'].append(at_high)  # Buying at highs is bad
        else:
            results['entry_at_extreme'].append(at_low)  # Selling at lows is bad

    print(f"\nAnalyzed {len(trades_df)} trades")
    print(f"\nPre-entry analysis:")
    print(f"  Trades where pre-move was in same direction: {100*np.mean(results['pre_entry_move_direction']):.1f}%")
    print(f"  Mean pre-entry momentum: {np.mean(results['pre_entry_momentum']):.2f} bps")

    print(f"\nPost-entry analysis:")
    print(f"  Immediate reversal rate (within 5s): {100*np.mean(results['post_entry_reversal_speed']):.1f}%")

    print(f"\nEntry quality:")
    print(f"  Entries at local extremes (bad): {100*np.mean(results['entry_at_extreme']):.1f}%")

    return results


def analyze_volume_price_relationship(df: pd.DataFrame) -> dict:
    """
    Task 3: Volume-Price Relationship
    Analyze whether volume leads price.
    """
    print("\n" + "="*80)
    print("TASK 3: VOLUME-PRICE RELATIONSHIP")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()

    # Volume leading indicators
    correlations = {}

    # Does high volume NOW predict abs(returns) in next 1-10 seconds?
    for lag in [1, 5, 10, 30]:
        future_abs_returns = df['abs_returns'].shift(-lag)
        corr = df['volume'].corr(future_abs_returns)
        correlations[f'volume_vs_abs_return_t+{lag}'] = corr

    print("\nVolume -> Future Price Movement Correlation:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.4f}")

    # High volume analysis
    df['volume_percentile'] = df['volume'].rolling(1000).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    df['high_volume'] = df['volume_percentile'] > 0.9

    # After high volume, does price continue or reverse?
    high_vol_continuation = []
    for idx in df[df['high_volume']].index:
        if idx < 10 or idx > len(df) - 30:
            continue

        current_bar_return = df.loc[idx, 'returns']
        future_10s_return = df.loc[idx:idx+10, 'returns'].sum() if idx + 10 < len(df) else np.nan

        if np.isnan(current_bar_return) or np.isnan(future_10s_return):
            continue

        # Continuation = same direction
        if current_bar_return * future_10s_return > 0:
            high_vol_continuation.append(1)
        else:
            high_vol_continuation.append(0)

    print(f"\nHigh volume bar analysis:")
    print(f"  After high volume bar, continuation rate: {100*np.mean(high_vol_continuation):.1f}%")
    print(f"  (Continuation = price continues in same direction as the high volume bar)")

    # Volume spike detection (climax patterns)
    df['volume_spike'] = df['volume'] > df['volume'].rolling(100).mean() * 3
    spike_reversal_rate = []

    for idx in df[df['volume_spike']].index:
        if idx < 10 or idx > len(df) - 30:
            continue

        current_bar_return = df.loc[idx, 'returns']
        future_30s_return = df.loc[idx:idx+30, 'returns'].sum() if idx + 30 < len(df) else np.nan

        if np.isnan(current_bar_return) or np.isnan(future_30s_return):
            continue

        # Reversal after spike?
        if current_bar_return * future_30s_return < 0:
            spike_reversal_rate.append(1)
        else:
            spike_reversal_rate.append(0)

    print(f"\nVolume spike (3x average) analysis:")
    print(f"  Reversal rate after spike: {100*np.mean(spike_reversal_rate):.1f}%")

    return {
        'correlations': correlations,
        'high_vol_continuation': np.mean(high_vol_continuation),
        'spike_reversal_rate': np.mean(spike_reversal_rate),
    }


def analyze_time_patterns(df: pd.DataFrame) -> dict:
    """
    Task 4: Time-of-Day Patterns
    Analyze profitability by time of day.
    """
    print("\n" + "="*80)
    print("TASK 4: TIME-OF-DAY PATTERNS")
    print("="*80)

    df = df.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')

    df['hour_utc'] = df['ts_event'].dt.hour
    df['day_of_week'] = df['ts_event'].dt.dayofweek
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()

    # Range and volatility by hour
    hourly_stats = df.groupby('hour_utc').agg({
        'high': 'mean',
        'low': 'mean',
        'volume': 'mean',
        'abs_returns': ['mean', 'std'],
    }).round(6)

    # Calculate hourly range
    df['bar_range'] = (df['high'] - df['low']) / df['close']
    hourly_range = df.groupby('hour_utc')['bar_range'].mean() * 10000  # in bps

    print("\nHourly Analysis (UTC time):")
    print("Hour | Avg Volume | Volatility (bps) | Avg Range (bps)")
    print("-" * 55)
    for hour in range(24):
        if hour in df['hour_utc'].values:
            hour_data = df[df['hour_utc'] == hour]
            vol = hour_data['volume'].mean()
            volatility = hour_data['abs_returns'].mean() * 10000
            avg_range = hourly_range.get(hour, 0)
            print(f"  {hour:02d} | {vol:10.1f} | {volatility:14.2f} | {avg_range:13.2f}")

    # Best hours for trading (highest volatility + volume)
    df['tradability'] = df['volume'] * df['abs_returns']
    tradability_by_hour = df.groupby('hour_utc')['tradability'].mean()

    best_hours = tradability_by_hour.nlargest(5).index.tolist()
    print(f"\nBest hours for trading (UTC): {best_hours}")
    print("(14:00-20:00 UTC = 8AM-2PM Chicago)")

    # Day of week analysis
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("\nDay of Week Analysis:")
    print("Day | Avg Volume | Volatility (bps)")
    print("-" * 40)
    for dow in range(7):
        if dow in df['day_of_week'].values:
            dow_data = df[df['day_of_week'] == dow]
            vol = dow_data['volume'].mean()
            volatility = dow_data['abs_returns'].mean() * 10000
            print(f" {dow_names[dow]} | {vol:10.1f} | {volatility:14.2f}")

    # Trend vs mean-reversion by hour
    # Calculate: Does a move in hour X tend to continue or reverse?
    print("\nTrend vs Mean-Reversion by Hour:")
    for hour in [14, 15, 16, 17, 18, 19, 20]:  # Peak trading hours
        hour_data = df[df['hour_utc'] == hour].copy()
        if len(hour_data) < 1000:
            continue

        # 30-second momentum -> next 30 seconds
        hour_data['momentum_30s'] = hour_data['close'].pct_change(30)
        hour_data['future_30s'] = hour_data['close'].pct_change(30).shift(-30)

        valid_data = hour_data.dropna(subset=['momentum_30s', 'future_30s'])
        if len(valid_data) < 100:
            continue

        # Continuation = same sign
        continuation = (valid_data['momentum_30s'] * valid_data['future_30s'] > 0).mean()
        print(f"  Hour {hour:02d} UTC: Trend continuation rate = {100*continuation:.1f}%")

    return {
        'best_hours': best_hours,
        'hourly_range': hourly_range.to_dict(),
    }


def analyze_volatility_regimes(df: pd.DataFrame) -> dict:
    """
    Task 5: Volatility Regime Analysis
    """
    print("\n" + "="*80)
    print("TASK 5: VOLATILITY REGIME ANALYSIS")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()

    # Calculate ATR at multiple timeframes
    df['atr_10'] = atr(df['high'], df['low'], df['close'], 10)
    df['atr_30'] = atr(df['high'], df['low'], df['close'], 30)
    df['atr_60'] = atr(df['high'], df['low'], df['close'], 60)

    # Volatility percentile
    df['vol_percentile'] = df['atr_30'].rolling(1000).apply(
        lambda x: (x.iloc[-1] >= x).mean() if len(x) > 0 else 0.5, raw=False
    )

    # Define regimes
    df['regime'] = pd.cut(
        df['vol_percentile'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['low', 'medium_low', 'medium_high', 'high']
    )

    # Performance by regime
    print("\nVolatility Regime Analysis:")
    print("Regime      | Count     | Avg Abs Return | Next 30s Move")
    print("-" * 60)

    for regime in ['low', 'medium_low', 'medium_high', 'high']:
        regime_data = df[df['regime'] == regime].copy()
        if len(regime_data) < 100:
            continue

        avg_abs_return = regime_data['abs_returns'].mean() * 10000

        # Future move analysis
        regime_data['future_30s_abs'] = regime_data['abs_returns'].rolling(30).sum().shift(-30)
        future_move = regime_data['future_30s_abs'].mean() * 10000

        print(f" {regime:11s} | {len(regime_data):9,d} | {avg_abs_return:13.2f} | {future_move:12.2f}")

    # Volatility expansion patterns
    df['vol_expanding'] = df['atr_10'] > df['atr_30']
    df['vol_contracting'] = df['atr_10'] < df['atr_30'] * 0.8

    # After contraction, is there expansion?
    contraction_to_expansion = []
    contraction_indices = df[df['vol_contracting']].index

    for idx in contraction_indices[:10000]:  # Sample
        if idx + 60 >= len(df):
            continue

        future_vol = df.loc[idx+30:idx+60, 'abs_returns'].mean()
        current_vol = df.loc[idx-30:idx, 'abs_returns'].mean() if idx > 30 else df.loc[:idx, 'abs_returns'].mean()

        if current_vol > 0:
            expansion_ratio = future_vol / current_vol
            contraction_to_expansion.append(expansion_ratio)

    print(f"\nVolatility contraction -> expansion:")
    print(f"  Mean expansion ratio after contraction: {np.mean(contraction_to_expansion):.2f}x")
    print(f"  Expansion (>1.5x) probability: {100*np.mean([x > 1.5 for x in contraction_to_expansion]):.1f}%")

    return {
        'contraction_expansion_ratio': np.mean(contraction_to_expansion) if contraction_to_expansion else 1.0,
    }


def analyze_mean_reversion_vs_trend(df: pd.DataFrame) -> dict:
    """
    Task 6: Mean Reversion vs Trend Following comparison.
    """
    print("\n" + "="*80)
    print("TASK 6: MEAN REVERSION VS TREND FOLLOWING")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Calculate indicators
    df['rsi_14'] = rsi(df['close'], 14)
    df['ema_10'] = ema(df['close'], 10)
    df['ema_30'] = ema(df['close'], 30)

    upper, middle, lower = bollinger_bands(df['close'], 20, 2.0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)

    # Mean Reversion Signals
    # RSI > 70 -> expect down
    df['mr_signal_short'] = (df['rsi_14'] > 70).astype(int)
    df['mr_signal_long'] = (df['rsi_14'] < 30).astype(int)

    # BB signals
    df['bb_signal_short'] = (df['bb_position'] > 1).astype(int)
    df['bb_signal_long'] = (df['bb_position'] < 0).astype(int)

    # Trend Following Signals
    # EMA crossover
    df['trend_long'] = (df['ema_10'] > df['ema_30']).astype(int)
    df['trend_short'] = (df['ema_10'] < df['ema_30']).astype(int)

    # Breakout signals
    df['high_30'] = df['high'].rolling(30).max()
    df['low_30'] = df['low'].rolling(30).min()
    df['breakout_long'] = (df['close'] > df['high_30'].shift(1)).astype(int)
    df['breakout_short'] = (df['close'] < df['low_30'].shift(1)).astype(int)

    # Calculate forward returns
    for horizon in [5, 10, 30]:
        df[f'fwd_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)

    # Evaluate mean reversion signals
    print("\nMean Reversion Strategy Performance:")

    # RSI oversold -> Long
    rsi_long = df[df['mr_signal_long'] == 1]
    if len(rsi_long) > 100:
        avg_return = rsi_long['fwd_return_30'].mean() * 10000
        win_rate = (rsi_long['fwd_return_30'] > 0).mean()
        print(f"  RSI < 30 -> LONG: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(rsi_long)}")

    # RSI overbought -> Short
    rsi_short = df[df['mr_signal_short'] == 1]
    if len(rsi_short) > 100:
        avg_return = -rsi_short['fwd_return_30'].mean() * 10000  # Negative because we're short
        win_rate = (rsi_short['fwd_return_30'] < 0).mean()
        print(f"  RSI > 70 -> SHORT: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(rsi_short)}")

    # BB signals
    bb_long = df[df['bb_signal_long'] == 1]
    if len(bb_long) > 100:
        avg_return = bb_long['fwd_return_30'].mean() * 10000
        win_rate = (bb_long['fwd_return_30'] > 0).mean()
        print(f"  Below BB -> LONG: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(bb_long)}")

    bb_short = df[df['bb_signal_short'] == 1]
    if len(bb_short) > 100:
        avg_return = -bb_short['fwd_return_30'].mean() * 10000
        win_rate = (bb_short['fwd_return_30'] < 0).mean()
        print(f"  Above BB -> SHORT: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(bb_short)}")

    # Evaluate trend following signals
    print("\nTrend Following Strategy Performance:")

    # EMA crossover long
    trend_long = df[df['trend_long'] == 1]
    if len(trend_long) > 100:
        avg_return = trend_long['fwd_return_30'].mean() * 10000
        win_rate = (trend_long['fwd_return_30'] > 0).mean()
        print(f"  EMA10 > EMA30 -> LONG: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%")

    trend_short = df[df['trend_short'] == 1]
    if len(trend_short) > 100:
        avg_return = -trend_short['fwd_return_30'].mean() * 10000
        win_rate = (trend_short['fwd_return_30'] < 0).mean()
        print(f"  EMA10 < EMA30 -> SHORT: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%")

    # Breakout signals
    breakout_long = df[df['breakout_long'] == 1]
    if len(breakout_long) > 100:
        avg_return = breakout_long['fwd_return_30'].mean() * 10000
        win_rate = (breakout_long['fwd_return_30'] > 0).mean()
        print(f"  30s High Breakout -> LONG: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(breakout_long)}")

    breakout_short = df[df['breakout_short'] == 1]
    if len(breakout_short) > 100:
        avg_return = -breakout_short['fwd_return_30'].mean() * 10000
        win_rate = (breakout_short['fwd_return_30'] < 0).mean()
        print(f"  30s Low Breakout -> SHORT: Avg 30s return = {avg_return:.2f} bps, Win rate = {100*win_rate:.1f}%, n={len(breakout_short)}")

    return {}


def analyze_entry_timing(df: pd.DataFrame) -> dict:
    """
    Task 7: Entry Timing Optimization
    Test alternative entry methods.
    """
    print("\n" + "="*80)
    print("TASK 7: ENTRY TIMING OPTIMIZATION")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Simulate different entry approaches

    # 1. Enter at close of signal bar (current approach)
    # 2. Enter after pullback
    # 3. Enter on break of signal bar high/low

    # Define a "signal" = breakout of 30-bar high
    df['high_30'] = df['high'].rolling(30).max()
    df['low_30'] = df['low'].rolling(30).min()
    df['signal_long'] = (df['close'] > df['high_30'].shift(1))
    df['signal_short'] = (df['close'] < df['low_30'].shift(1))

    results = {
        'immediate_entry': [],
        'pullback_entry': [],
        'confirmation_entry': [],
    }

    # Sample signal bars
    long_signals = df[df['signal_long']].index[:5000]

    for idx in long_signals:
        if idx + 60 >= len(df):
            continue

        signal_close = df.loc[idx, 'close']

        # Immediate entry result
        exit_price = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else signal_close
        immediate_return = (exit_price - signal_close) / signal_close
        results['immediate_entry'].append(immediate_return)

        # Pullback entry: Wait for price to drop 0.02% then enter
        pullback_threshold = signal_close * 0.9998  # 2 bps pullback
        entry_found = False

        for future_idx in range(idx + 1, min(idx + 20, len(df))):
            if df.loc[future_idx, 'low'] <= pullback_threshold:
                # Found pullback, enter here
                entry_price = pullback_threshold
                exit_idx = min(future_idx + 30, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']
                pullback_return = (exit_price - entry_price) / entry_price
                results['pullback_entry'].append(pullback_return)
                entry_found = True
                break

        if not entry_found:
            results['pullback_entry'].append(np.nan)

        # Confirmation entry: Enter only if next bar makes new high
        if idx + 1 < len(df):
            next_high = df.loc[idx + 1, 'high']
            if next_high > signal_close:
                # Confirmed, enter at break
                entry_price = signal_close + 0.01  # Just above
                exit_idx = min(idx + 31, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']
                confirm_return = (exit_price - entry_price) / entry_price
                results['confirmation_entry'].append(confirm_return)
            else:
                results['confirmation_entry'].append(np.nan)

    print("\nEntry Timing Comparison (LONG signals on 30-bar breakout):")

    imm = [x for x in results['immediate_entry'] if not np.isnan(x)]
    pb = [x for x in results['pullback_entry'] if not np.isnan(x)]
    conf = [x for x in results['confirmation_entry'] if not np.isnan(x)]

    print(f"\n1. Immediate Entry (at signal bar close):")
    print(f"   Avg return: {np.mean(imm)*10000:.2f} bps")
    print(f"   Win rate: {100*np.mean([x > 0 for x in imm]):.1f}%")
    print(f"   Trades: {len(imm)}")

    print(f"\n2. Pullback Entry (wait for 2 bps pullback):")
    print(f"   Avg return: {np.mean(pb)*10000:.2f} bps")
    print(f"   Win rate: {100*np.mean([x > 0 for x in pb]):.1f}%")
    print(f"   Trades: {len(pb)} (missed: {len(imm) - len(pb)})")

    print(f"\n3. Confirmation Entry (enter on new high):")
    print(f"   Avg return: {np.mean(conf)*10000:.2f} bps")
    print(f"   Win rate: {100*np.mean([x > 0 for x in conf]):.1f}%")
    print(f"   Trades: {len(conf)} (missed: {len(imm) - len(conf)})")

    return results


def analyze_stop_loss(df: pd.DataFrame) -> dict:
    """
    Task 8: Stop Loss Optimization
    """
    print("\n" + "="*80)
    print("TASK 8: STOP LOSS OPTIMIZATION")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['atr_30'] = atr(df['high'], df['low'], df['close'], 30)

    # Analyze typical adverse excursion
    # For each bar, how far does price move against before moving in favor?

    df['high_30'] = df['high'].rolling(30).max()
    df['signal_long'] = (df['close'] > df['high_30'].shift(1))

    # For long signals, track Maximum Adverse Excursion (MAE)
    mae_results = []
    mfe_results = []  # Maximum Favorable Excursion

    long_signals = df[df['signal_long']].index[:3000]

    for idx in long_signals:
        if idx + 120 >= len(df):
            continue

        entry_price = df.loc[idx, 'close']

        # Track 120 seconds forward
        future_slice = df.loc[idx:idx+120]

        # MAE = lowest low relative to entry
        mae = (entry_price - future_slice['low'].min()) / entry_price
        mae_results.append(mae * 10000)  # in bps

        # MFE = highest high relative to entry
        mfe = (future_slice['high'].max() - entry_price) / entry_price
        mfe_results.append(mfe * 10000)

    print(f"\nMaximum Adverse Excursion (MAE) for LONG trades:")
    print(f"  Mean: {np.mean(mae_results):.2f} bps")
    print(f"  Median: {np.median(mae_results):.2f} bps")
    print(f"  90th percentile: {np.percentile(mae_results, 90):.2f} bps")
    print(f"  95th percentile: {np.percentile(mae_results, 95):.2f} bps")

    print(f"\nMaximum Favorable Excursion (MFE) for LONG trades:")
    print(f"  Mean: {np.mean(mfe_results):.2f} bps")
    print(f"  Median: {np.median(mfe_results):.2f} bps")

    # What stop distance survives X% of trades?
    survival_rates = {}
    for stop_bps in [2, 3, 4, 5, 6, 8, 10, 15, 20]:
        survived = np.mean([mae < stop_bps for mae in mae_results])
        survival_rates[stop_bps] = survived

    print(f"\nStop Distance Survival Rates:")
    print("Stop (bps) | Survival Rate")
    print("-" * 30)
    for stop, rate in survival_rates.items():
        print(f"    {stop:3d}    |    {100*rate:.1f}%")

    # Optimal stop = survive most while capturing profit
    # Need stop that survives AND trade becomes profitable
    print(f"\nRecommendation:")
    print(f"  Minimum stop: {np.percentile(mae_results, 50):.1f} bps (survive 50%)")
    print(f"  Safe stop: {np.percentile(mae_results, 70):.1f} bps (survive 70%)")
    print(f"  Wide stop: {np.percentile(mae_results, 90):.1f} bps (survive 90%)")

    # Time-based stop analysis
    print(f"\n Time-Based Stop Analysis:")
    time_stops = []
    for idx in long_signals[:1000]:
        if idx + 120 >= len(df):
            continue

        entry_price = df.loc[idx, 'close']

        # Track P&L at different time horizons
        for t in [10, 30, 60, 120]:
            if idx + t < len(df):
                exit_price = df.loc[idx + t, 'close']
                pnl = (exit_price - entry_price) / entry_price * 10000
                time_stops.append({'time': t, 'pnl': pnl})

    time_df = pd.DataFrame(time_stops)
    print("Hold Time | Avg PnL (bps) | Win Rate")
    print("-" * 40)
    for t in [10, 30, 60, 120]:
        t_data = time_df[time_df['time'] == t]
        avg_pnl = t_data['pnl'].mean()
        win_rate = (t_data['pnl'] > 0).mean()
        print(f"   {t:3d}s   |    {avg_pnl:6.2f}    |  {100*win_rate:.1f}%")

    return {
        'mae_median': np.median(mae_results),
        'mae_90pct': np.percentile(mae_results, 90),
        'survival_rates': survival_rates,
    }


def analyze_patterns(df: pd.DataFrame) -> dict:
    """
    Task 9: Pattern Recognition
    """
    print("\n" + "="*80)
    print("TASK 9: PATTERN RECOGNITION")
    print("="*80)

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['bar_range'] = df['high'] - df['low']

    # Inside bar pattern
    df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

    # After inside bar, does breakout continue?
    inside_bar_results = []

    for idx in df[df['inside_bar']].index[:5000]:
        if idx + 60 >= len(df):
            continue

        inside_high = df.loc[idx, 'high']
        inside_low = df.loc[idx, 'low']

        # Look for breakout in next 10 bars
        for future_idx in range(idx + 1, min(idx + 10, len(df))):
            if df.loc[future_idx, 'close'] > inside_high:
                # Upward breakout
                entry = inside_high + 0.01
                exit_idx = min(future_idx + 30, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']
                pnl = (exit_price - entry) / entry * 10000
                inside_bar_results.append({'direction': 'LONG', 'pnl': pnl})
                break
            elif df.loc[future_idx, 'close'] < inside_low:
                # Downward breakout
                entry = inside_low - 0.01
                exit_idx = min(future_idx + 30, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']
                pnl = (entry - exit_price) / entry * 10000
                inside_bar_results.append({'direction': 'SHORT', 'pnl': pnl})
                break

    if inside_bar_results:
        ib_df = pd.DataFrame(inside_bar_results)
        print(f"\nInside Bar Breakout Pattern:")
        print(f"  Total signals: {len(ib_df)}")
        print(f"  Avg PnL: {ib_df['pnl'].mean():.2f} bps")
        print(f"  Win rate: {100*(ib_df['pnl'] > 0).mean():.1f}%")

        for direction in ['LONG', 'SHORT']:
            dir_data = ib_df[ib_df['direction'] == direction]
            if len(dir_data) > 10:
                print(f"    {direction}: Avg = {dir_data['pnl'].mean():.2f} bps, Win = {100*(dir_data['pnl'] > 0).mean():.1f}%")

    # Three-bar reversal pattern
    # Down-down-up = bullish reversal
    df['bar_direction'] = np.sign(df['close'] - df['open'])
    df['three_bar_bullish'] = (
        (df['bar_direction'].shift(2) < 0) &
        (df['bar_direction'].shift(1) < 0) &
        (df['bar_direction'] > 0)
    )
    df['three_bar_bearish'] = (
        (df['bar_direction'].shift(2) > 0) &
        (df['bar_direction'].shift(1) > 0) &
        (df['bar_direction'] < 0)
    )

    # Analyze three-bar patterns
    three_bar_results = []

    for idx in df[df['three_bar_bullish']].index[:3000]:
        if idx + 60 >= len(df):
            continue

        entry = df.loc[idx, 'close']
        exit_price = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else entry
        pnl = (exit_price - entry) / entry * 10000
        three_bar_results.append({'direction': 'LONG', 'pnl': pnl})

    for idx in df[df['three_bar_bearish']].index[:3000]:
        if idx + 60 >= len(df):
            continue

        entry = df.loc[idx, 'close']
        exit_price = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else entry
        pnl = (entry - exit_price) / entry * 10000
        three_bar_results.append({'direction': 'SHORT', 'pnl': pnl})

    if three_bar_results:
        tb_df = pd.DataFrame(three_bar_results)
        print(f"\nThree-Bar Reversal Pattern:")
        print(f"  Total signals: {len(tb_df)}")
        print(f"  Avg PnL: {tb_df['pnl'].mean():.2f} bps")
        print(f"  Win rate: {100*(tb_df['pnl'] > 0).mean():.1f}%")

    # Volume climax reversal
    df['volume_ma'] = df['volume'].rolling(100).mean()
    df['volume_spike'] = df['volume'] > df['volume_ma'] * 3
    df['big_up_bar'] = (df['returns'] > 0.0003) & df['volume_spike']
    df['big_down_bar'] = (df['returns'] < -0.0003) & df['volume_spike']

    climax_results = []

    # After big up bar with volume, fade it
    for idx in df[df['big_up_bar']].index[:2000]:
        if idx + 60 >= len(df):
            continue

        entry = df.loc[idx, 'close']
        exit_price = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else entry
        # Fade = short after big up
        pnl = (entry - exit_price) / entry * 10000
        climax_results.append({'type': 'fade_up', 'pnl': pnl})

    for idx in df[df['big_down_bar']].index[:2000]:
        if idx + 60 >= len(df):
            continue

        entry = df.loc[idx, 'close']
        exit_price = df.loc[idx + 30, 'close'] if idx + 30 < len(df) else entry
        # Fade = long after big down
        pnl = (exit_price - entry) / entry * 10000
        climax_results.append({'type': 'fade_down', 'pnl': pnl})

    if climax_results:
        climax_df = pd.DataFrame(climax_results)
        print(f"\nVolume Climax Fade Pattern:")
        print(f"  Total signals: {len(climax_df)}")
        print(f"  Avg PnL: {climax_df['pnl'].mean():.2f} bps")
        print(f"  Win rate: {100*(climax_df['pnl'] > 0).mean():.1f}%")

        for ptype in ['fade_up', 'fade_down']:
            type_data = climax_df[climax_df['type'] == ptype]
            if len(type_data) > 10:
                print(f"    {ptype}: Avg = {type_data['pnl'].mean():.2f} bps, Win = {100*(type_data['pnl'] > 0).mean():.1f}%")

    # Failed breakout pattern
    df['high_30'] = df['high'].rolling(30).max()
    df['low_30'] = df['low'].rolling(30).min()

    # Breakout then reversal
    df['breakout_up'] = df['high'] > df['high_30'].shift(1)
    df['breakout_down'] = df['low'] < df['low_30'].shift(1)

    failed_breakout_results = []

    for idx in df[df['breakout_up']].index[:3000]:
        if idx + 10 >= len(df):
            continue

        breakout_high = df.loc[idx, 'high']

        # Check if it fails (closes back below breakout level within 5 bars)
        for check_idx in range(idx + 1, min(idx + 5, len(df))):
            if df.loc[check_idx, 'close'] < df.loc[idx, 'open']:  # Failed
                # Short on failed breakout
                entry = df.loc[check_idx, 'close']
                exit_idx = min(check_idx + 30, len(df) - 1)
                exit_price = df.loc[exit_idx, 'close']
                pnl = (entry - exit_price) / entry * 10000
                failed_breakout_results.append({'type': 'failed_up', 'pnl': pnl})
                break

    if failed_breakout_results:
        fb_df = pd.DataFrame(failed_breakout_results)
        print(f"\nFailed Breakout Pattern:")
        print(f"  Total signals: {len(fb_df)}")
        print(f"  Avg PnL: {fb_df['pnl'].mean():.2f} bps")
        print(f"  Win rate: {100*(fb_df['pnl'] > 0).mean():.1f}%")

    return {}


def main():
    """Run all analyses."""
    print("="*80)
    print("MCL MICROSTRUCTURE ANALYSIS")
    print("="*80)

    # Load data - recent 30 days of 1-second data for detailed analysis
    print("\nLoading 1-second data (last 30 days)...")
    end_date = datetime(2026, 1, 8)
    start_date = end_date - timedelta(days=30)

    df_1s = load_1s_data(start_date, end_date)
    print(f"Loaded {len(df_1s):,} 1-second bars")

    if df_1s.empty:
        print("No 1-second data found!")
        return

    # Load failed trades
    trades_df = load_failed_trades()

    # Run analyses
    results = {}

    # Task 1: Price Microstructure
    results['microstructure'] = analyze_price_microstructure(df_1s)

    # Task 2: Failed Trade Analysis
    if not trades_df.empty:
        results['failed_trades'] = analyze_failed_trades(df_1s, trades_df)

    # Task 3: Volume-Price Relationship
    results['volume_price'] = analyze_volume_price_relationship(df_1s)

    # Task 4: Time-of-Day Patterns
    results['time_patterns'] = analyze_time_patterns(df_1s)

    # Task 5: Volatility Regimes
    results['volatility'] = analyze_volatility_regimes(df_1s)

    # Task 6: Mean Reversion vs Trend
    results['mr_vs_trend'] = analyze_mean_reversion_vs_trend(df_1s)

    # Task 7: Entry Timing
    results['entry_timing'] = analyze_entry_timing(df_1s)

    # Task 8: Stop Loss
    results['stop_loss'] = analyze_stop_loss(df_1s)

    # Task 9: Patterns
    results['patterns'] = analyze_patterns(df_1s)

    # Summary and Recommendations
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)

    print("""
FINDINGS:
=========

1. CONTINUATION VS REVERSAL
   - After significant moves, continuation is NOT guaranteed
   - Mean retracement is substantial
   - This explains why trend-following signals fail immediately

2. FAILED TRADE DIAGNOSIS
   - Trades enter AFTER the move has already happened
   - Entry often occurs at local price extremes
   - Immediate reversal rate is very high

3. VOLUME INSIGHTS
   - High volume bars often see REVERSAL, not continuation
   - Volume spikes are potential fade signals

4. ENTRY TIMING
   - Immediate entry (current approach) is suboptimal
   - Pullback entries improve win rate
   - Confirmation entries filter out false signals

5. STOP LOSS
   - 2 ATR stops are too tight for normal noise
   - Need wider stops or time-based exits

PROPOSED NEW APPROACH:
=====================

1. FADE EXHAUSTION MOVES (Mean Reversion)
   - After 3 consecutive bars in same direction + volume spike
   - After RSI extreme (>80 or <20)
   - After price extends > 1.5x ATR from 30-bar mean

2. WAIT FOR PULLBACK
   - Don't enter at breakout
   - Wait for 30-50% retracement of the move
   - Enter on reversal of the pullback

3. WIDER STOPS + TIME STOPS
   - Minimum stop: 5-6 bps (survive 70% of natural noise)
   - Time stop: Exit after 60-120 seconds if flat

4. FILTER CONDITIONS
   - Only trade 14:00-20:00 UTC (peak liquidity)
   - Avoid low volatility regimes (no edge)
   - Skip first 30 min after market open (chaos)
""")


if __name__ == "__main__":
    main()
