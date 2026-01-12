"""
Backtest Comparison: Mean Reversion vs Scalping (Trend Following)
================================================================
Compare the new mean reversion strategy against the failing scalping strategy.
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
    create_conservative_config,
    create_optimized_config,
)
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import Signal
from src.backtest.engine import BacktestEngine, BacktestConfig

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
storage = DataStorage(DATA_DIR)


def load_clean_data(start_date: datetime, end_date: datetime, schema: str = "ohlcv-1m") -> pd.DataFrame:
    """Load and clean data."""
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

    # Clean: Keep only reasonable MCL prices
    df = df[(df['close'] > 50) & (df['close'] < 90)]

    # Remove extreme returns
    df['returns'] = df['close'].pct_change()
    df = df[(df['returns'].abs() < 0.02) | (df['returns'].isna())]

    # Set index to timestamp
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    if df['ts_event'].dt.tz is None:
        df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
    df = df.set_index('ts_event')

    return df.reset_index()


def simple_backtest(
    data: pd.DataFrame,
    strategy,
    commission_per_side: float = 1.37,  # $1.37/side for MCL
    slippage_ticks: float = 1,  # 1 tick slippage ($0.01)
    tick_value: float = 1.0,  # $1 per tick for MCL
) -> dict:
    """Run a simple backtest on the strategy.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy instance
        commission_per_side: Commission per contract per side
        slippage_ticks: Expected slippage in ticks

    Returns:
        Dictionary with backtest results
    """
    # Generate signals
    df = data.copy()
    if 'ts_event' in df.columns:
        df = df.set_index('ts_event')

    # Create a copy with ts_event as a column for the strategy
    signal_data = df.reset_index()
    if 'ts_event' not in signal_data.columns and 'index' in signal_data.columns:
        signal_data = signal_data.rename(columns={'index': 'ts_event'})
    signal_data.index = signal_data['ts_event']

    signals = strategy.generate_signals(signal_data)

    # Track trades
    trades = []
    position = Signal.FLAT
    entry_price = 0.0
    entry_idx = 0
    entry_time = None

    # Get config parameters
    if hasattr(strategy, 'config'):
        if hasattr(strategy.config, 'stop_loss_cents'):
            stop_cents = strategy.config.stop_loss_cents / 100
            target_cents = strategy.config.take_profit_cents / 100
            max_bars = strategy.config.max_hold_bars
        else:
            # Scalping strategy uses ATR
            stop_cents = 0.10  # Default 10 cents
            target_cents = 0.08
            max_bars = 120
    else:
        stop_cents = 0.10
        target_cents = 0.08
        max_bars = 20

    slippage = slippage_ticks * 0.01  # Convert to dollars
    commission_rt = commission_per_side * 2  # Round trip

    for i in range(len(df)):
        current_signal = signals.iloc[i] if i < len(signals) else Signal.FLAT
        current_close = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_time = df.index[i] if hasattr(df.index[i], 'hour') else None

        # Check exit conditions if in position
        if position != Signal.FLAT:
            bars_held = i - entry_idx
            should_exit = False
            exit_price = current_close
            exit_reason = ""

            if position == Signal.LONG:
                # Check stop loss
                if current_low <= entry_price - stop_cents:
                    should_exit = True
                    exit_price = entry_price - stop_cents - slippage
                    exit_reason = "stop_loss"
                # Check take profit
                elif current_high >= entry_price + target_cents:
                    should_exit = True
                    exit_price = entry_price + target_cents - slippage
                    exit_reason = "take_profit"
                # Check time stop
                elif bars_held >= max_bars:
                    should_exit = True
                    exit_price = current_close - slippage
                    exit_reason = "time_stop"

            elif position == Signal.SHORT:
                # Check stop loss
                if current_high >= entry_price + stop_cents:
                    should_exit = True
                    exit_price = entry_price + stop_cents + slippage
                    exit_reason = "stop_loss"
                # Check take profit
                elif current_low <= entry_price - target_cents:
                    should_exit = True
                    exit_price = entry_price - target_cents + slippage
                    exit_reason = "take_profit"
                # Check time stop
                elif bars_held >= max_bars:
                    should_exit = True
                    exit_price = current_close + slippage
                    exit_reason = "time_stop"

            if should_exit:
                # Calculate P&L
                if position == Signal.LONG:
                    pnl = (exit_price - entry_price) * 100 - commission_rt  # 100 = MCL multiplier
                else:
                    pnl = (entry_price - exit_price) * 100 - commission_rt

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'direction': 'LONG' if position == Signal.LONG else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                })

                position = Signal.FLAT

        # Check for new entry (only if flat)
        if position == Signal.FLAT and current_signal != Signal.FLAT:
            position = current_signal
            entry_price = current_close + (slippage if current_signal == Signal.LONG else -slippage)
            entry_idx = i
            entry_time = current_time

    # Calculate statistics
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'trades': [],
        }

    trades_df = pd.DataFrame(trades)

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0

    # Calculate equity curve for drawdown
    equity = [0]
    for pnl in trades_df['pnl']:
        equity.append(equity[-1] + pnl)

    equity_series = pd.Series(equity)
    rolling_max = equity_series.cummax()
    drawdown = equity_series - rolling_max
    max_drawdown = drawdown.min()

    return {
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
        'total_pnl': trades_df['pnl'].sum(),
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
        'max_drawdown': max_drawdown,
        'longs': len(trades_df[trades_df['direction'] == 'LONG']),
        'shorts': len(trades_df[trades_df['direction'] == 'SHORT']),
        'stop_outs': len(trades_df[trades_df['exit_reason'] == 'stop_loss']),
        'targets_hit': len(trades_df[trades_df['exit_reason'] == 'take_profit']),
        'time_stops': len(trades_df[trades_df['exit_reason'] == 'time_stop']),
        'trades': trades,
    }


def engine_backtest(
    data: pd.DataFrame,
    strategy,
    initial_capital: float = 5_000.0,
) -> dict:
    """Run backtest using the BacktestEngine with multi-level exits.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy instance
        initial_capital: Starting capital

    Returns:
        Dictionary with backtest results
    """
    # Prepare data with proper index
    df = data.copy()
    if 'ts_event' in df.columns:
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        if df['ts_event'].dt.tz is None:
            df['ts_event'] = df['ts_event'].dt.tz_localize('UTC')
        df = df.set_index('ts_event')

    # Create backtest engine with risk management disabled for fair comparison
    config = BacktestConfig(
        initial_capital=initial_capital,
        use_risk_manager=False,  # Disable to get raw strategy performance
        position_size=1,  # 1 contract
    )
    engine = BacktestEngine(config)

    try:
        result = engine.run(df, strategy)

        # Extract metrics
        trades_df = result.trades
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'trades': [],
            }

        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0

        return {
            'total_trades': result.metrics.total_trades,
            'win_rate': result.metrics.win_rate,
            'total_pnl': result.metrics.total_return,  # total_return is the $ P&L
            'avg_pnl': result.metrics.avg_trade_pnl,
            'avg_win': result.metrics.avg_winner,
            'avg_loss': result.metrics.avg_loser,
            'profit_factor': result.metrics.profit_factor,
            'max_drawdown': result.metrics.max_drawdown,
            'longs': len(trades_df[trades_df['direction'] == 'LONG']) if 'direction' in trades_df.columns else 0,
            'shorts': len(trades_df[trades_df['direction'] == 'SHORT']) if 'direction' in trades_df.columns else 0,
            'stop_outs': len(trades_df[trades_df['exit_type'] == 'stop_loss']) if 'exit_type' in trades_df.columns else 0,
            'targets_hit': len(trades_df[trades_df['exit_type'].str.contains('target', na=False)]) if 'exit_type' in trades_df.columns else 0,
            'time_stops': len(trades_df[trades_df['exit_type'] == 'time_stop']) if 'exit_type' in trades_df.columns else 0,
            'trades': trades_df.to_dict('records'),
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'trades': [],
            'error': str(e),
        }


def main():
    """Run comparison backtest."""
    print("="*80)
    print("BACKTEST COMPARISON: Mean Reversion vs Trend Following")
    print("="*80)

    # Load data
    end_date = datetime(2026, 1, 8)
    start_date = datetime(2025, 11, 1)  # 2 months

    print(f"\nLoading data from {start_date.date()} to {end_date.date()}...")
    data = load_clean_data(start_date, end_date, "ohlcv-1m")

    if data.empty:
        print("No data!")
        return

    print(f"Loaded {len(data):,} bars")

    # Initialize strategies
    strategies = {
        'Mean Reversion (Optimized)': MeanReversionStrategy(create_optimized_config()),
        'Mean Reversion (Default)': MeanReversionStrategy(MeanReversionConfig()),
        'Mean Reversion (Conservative)': MeanReversionStrategy(create_conservative_config()),
        'Scalping (Trend Following)': ScalpingStrategy(ScalpingConfig()),
    }

    # Run backtests
    results = {}

    for name, strategy in strategies.items():
        print(f"\nRunning backtest: {name}...")
        # Use engine backtest for MeanReversionStrategy to get multi-level exits
        if isinstance(strategy, MeanReversionStrategy):
            result = engine_backtest(data, strategy)
        else:
            result = simple_backtest(data, strategy)
        results[name] = result

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    print(f"\n{'Strategy':<35} | {'Trades':>8} | {'Win Rate':>10} | {'Total PnL':>12} | {'Avg PnL':>10} | {'PF':>6} | {'Max DD':>10}")
    print("-" * 110)

    for name, result in results.items():
        # win_rate from engine_backtest is already percentage, from simple_backtest is 0-1
        win_rate = result['win_rate']
        if win_rate <= 1.0 and result['total_trades'] > 0:  # simple_backtest returns 0-1
            win_rate = win_rate * 100
        print(
            f"{name:<35} | "
            f"{result['total_trades']:>8} | "
            f"{win_rate:>9.1f}% | "
            f"${result['total_pnl']:>10.2f} | "
            f"${result['avg_pnl']:>9.2f} | "
            f"{result['profit_factor']:>5.2f} | "
            f"${result['max_drawdown']:>9.2f}"
        )

    # Detailed breakdown for best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['total_pnl'])
    print(f"\n{'='*80}")
    print(f"DETAILED BREAKDOWN: {best_strategy[0]}")
    print("="*80)

    result = best_strategy[1]
    total_trades = result['total_trades'] or 1  # Avoid division by zero

    # Get values with defaults
    longs = result.get('longs', 0)
    shorts = result.get('shorts', 0)
    targets_hit = result.get('targets_hit', 0)
    stop_outs = result.get('stop_outs', 0)
    time_stops = result.get('time_stops', 0)
    avg_win = result.get('avg_win', 0)
    avg_loss = result.get('avg_loss', 0)

    print(f"""
Total Trades: {result['total_trades']}
  - Long: {longs}
  - Short: {shorts}

Win Rate: {result['win_rate'] if result['win_rate'] > 1 else result['win_rate']*100:.1f}%

Exit Reasons:
  - Take Profit: {targets_hit} ({100*targets_hit/total_trades:.1f}%)
  - Stop Loss: {stop_outs} ({100*stop_outs/total_trades:.1f}%)
  - Time Stop: {time_stops} ({100*time_stops/total_trades:.1f}%)

P&L:
  - Total: ${result['total_pnl']:.2f}
  - Average: ${result['avg_pnl']:.2f}
  - Avg Win: ${avg_win:.2f}
  - Avg Loss: ${avg_loss:.2f}
  - Profit Factor: {result['profit_factor']:.2f}
  - Max Drawdown: ${result['max_drawdown']:.2f}
""")

    # Analysis of why scalping fails
    scalping_result = results.get('Scalping (Trend Following)', {})
    if scalping_result.get('total_trades', 0) > 0:
        print("\n" + "="*80)
        print("WHY SCALPING (TREND FOLLOWING) FAILS")
        print("="*80)

        print(f"""
Scalping Strategy Results:
  - Total Trades: {scalping_result['total_trades']}
  - Win Rate: {scalping_result['win_rate'] if scalping_result['win_rate'] > 1 else scalping_result['win_rate']*100:.1f}%
  - Stop Outs: {scalping_result['stop_outs']} ({100*scalping_result['stop_outs']/scalping_result['total_trades']:.1f}%)

ANALYSIS:
---------
1. The scalping strategy enters on MOMENTUM + TREND confirmation
2. By the time all conditions align, the move has already happened
3. Price then reverts, hitting the stop loss immediately
4. This is why the win rate is extremely low

SOLUTION:
---------
The Mean Reversion strategy FADES these exhaustion moves instead of following them.
When momentum + volume spike = exhaustion, price is MORE likely to reverse.
""")


if __name__ == "__main__":
    main()
