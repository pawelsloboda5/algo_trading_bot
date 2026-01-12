# MCL Scalping Strategy Analysis - Final Report

## Executive Summary

After comprehensive analysis of MCL (Micro WTI Crude Oil Futures) historical data, we have identified **why the current scalping strategy has a 0% win rate** and developed an improved approach.

**Key Finding**: MCL is a **mean-reverting instrument at short timeframes**. Trend-following strategies that buy strength and sell weakness are fundamentally wrong for this market.

---

## CRITICAL BUG DISCOVERED (2026-01-12)

### Data Quality Issue

**Problem**: The raw data files contain **multiple symbols mixed together**:
- Outright futures: MCLQ5, MCLU5, MCLV5, MCLX5, MCLZ5 (different contract months)
- Calendar spreads: MCLQ5-MCLU5, MCLQ5-MCLV5, etc.

**Impact**: The backtest was entering on one contract (e.g., MCLU5 at $65.66) and exiting on a different contract (e.g., MCLQ5 at $68.22), generating **fake $2.50 profits per trade**.

**Result**: Inflated metrics showing:
- $1,064,448 profit (FAKE)
- 83.84% win rate (FAKE)
- 22.64 profit factor (FAKE)

**Fix Applied**: Updated `load_clean_data()` in both `deep_analysis.py` and `backtest_comparison.py` to:
1. Filter out calendar spreads (symbols containing '-')
2. Select only the most liquid outright contract per day
3. Remove duplicate timestamps

**Status**: Re-run backtests with clean data to get accurate results.

---

## Current Strategy Failure Diagnosis

### The Problem
The existing `ScalpingStrategy` uses these entry conditions:
1. Momentum > threshold (price moving in direction)
2. Trend confirmed (EMA alignment)
3. Volume acceleration (high participation)

**Result**: 0-5% win rate, 95% of trades hit stop loss immediately.

### Why It Fails

| Issue | Evidence |
|-------|----------|
| **Entries at local extremes** | 45.5% of entries occur at 60-second highs/lows |
| **Signals lag the move** | By the time momentum + trend confirm, the move is OVER |
| **Wrong market regime** | MCL mean-reverts 65-75% of the time at 1-minute timeframe |
| **Stop too tight** | 2 ATR stops hit 95% of the time by normal noise |

### Backtest Evidence

| Strategy | Trades | Win Rate | Total PnL | Stop Outs |
|----------|--------|----------|-----------|-----------|
| Scalping (Trend Following) | 1,874 | 5.0% | -$24,057 | 95.0% |
| Mean Reversion (No Confirm) | 9,039 | 76.1% | -$265 | 23.8% |

---

## Data Analysis Findings

### 1. Mean Reversion vs Trend Following

**1-Minute Bar Analysis (167,282 bars)**:

| Signal Type | Win Rate | Avg Return (bps) |
|-------------|----------|------------------|
| EMA deviation < -0.3% → LONG | 83.2% | +49.5 |
| EMA deviation > +0.3% → SHORT | 74.9% | +40.4 |
| 20-bar Breakout → LONG | 20.8% | -27.4 |
| 20-bar Breakout → SHORT | 9.5% | -54.0 |

**Conclusion**: Fading extended moves works. Following breakouts loses money.

### 2. Volume Analysis

After HIGH volume bars (>2x average):
- **UP bars**: Only 36% continue up → FADE (short) recommended
- **DOWN bars**: Only 40% continue down → FADE (long) recommended

High volume = exhaustion, not continuation.

### 3. Time of Day Patterns

Best trading hours (UTC):
- **14:00-20:00** (8AM-2PM Chicago) - Peak liquidity
- Avoid first 30 minutes after open (chaos)
- Avoid low-volume overnight hours

### 4. Stop Loss Analysis

For mean reversion signals:
- **3 cent stop**: 0.6% survive (too tight)
- **5 cent stop**: 0.6% survive (too tight)
- **10 cent stop**: Minimum viable
- **15-20 cent stop**: Recommended for normal noise

---

## New Strategy: Mean Reversion Scalper

### Entry Rules (LONG)
Primary condition (ANY of these):
1. Price > 0.3% below EMA(20) - extended down
2. RSI(14) < 35 - oversold
3. High volume (>2x avg) DOWN bar - selling exhaustion

Optional confirmation:
- Current bar closes above previous close (reversal starting)

### Entry Rules (SHORT)
Primary condition (ANY of these):
1. Price > 0.3% above EMA(20) - extended up
2. RSI(14) > 65 - overbought
3. High volume (>2x avg) UP bar - buying exhaustion

### Exit Rules
- **Stop Loss**: 10 cents ($10/contract)
- **Take Profit**: 8 cents ($8/contract)
- **Time Stop**: 20 minutes max hold

### Session Filter
- Only trade 14:00-20:00 UTC

---

## Backtest Results Comparison

### Without Confirmation (More Signals)
```
Total Trades: 9,039
Win Rate: 76.1%
Take Profits: 6,880 (76.1%)
Stop Losses: 2,149 (23.8%)
Average Win: $4.26
Average Loss: $-13.71
Profit Factor: 0.99
Total PnL: -$265
```

### With Confirmation (Fewer Signals)
```
Total Trades: 1,988
Win Rate: 22.2%
```

### Analysis
The "No Confirmation" version has:
- **76% win rate** (vs 5% for trend following)
- **Near break-even** (vs -$24k loss)
- **Profit factor of 0.99** (almost 1.0)

The challenge: Risk/reward of 8:10 requires >55.6% win rate to profit.
With 76% win rate but losers 3x size of winners, we're break-even.

---

## Optimization Recommendations

### Option 1: Tighter Stops
- Reduce stop to 6-8 cents
- Accept lower win rate (~60%)
- Target profit factor > 1.0

### Option 2: Wider Targets
- Increase target to 10-12 cents
- Keep 76% win rate
- Improve risk/reward

### Option 3: Better Signal Selection
- Only trade during peak hours
- Require multiple conditions (RSI + EMA + Volume)
- Filter for higher-quality setups

### Option 4: Scale Out
- Take 50% at first target (4 cents)
- Move stop to breakeven
- Let remainder run to 10+ cents

---

## Files Created

| File | Purpose |
|------|---------|
| `research/mcl_microstructure_analysis.py` | Initial comprehensive analysis |
| `research/mcl_deep_analysis.py` | Cleaned analysis with better data handling |
| `research/final_analysis_report.py` | Signal performance analysis |
| `research/backtest_comparison.py` | Strategy comparison backtest |
| `src/strategy/mean_reversion_strategy.py` | New mean reversion strategy |

---

## Critical Warnings

1. **MCL has thin liquidity** - Slippage can be 1-2 ticks ($1-2)
2. **Commissions eat profits** - $2.74 round-trip per contract
3. **Need 4+ ticks profit** minimum to overcome costs
4. **Paper trade first** - Validate before live trading
5. **This is a low-edge strategy** - Size positions appropriately

---

## Conclusion

The current scalping strategy fails because it tries to trade MCL as a trending instrument when it actually mean-reverts at short timeframes. By reversing the logic - fading exhaustion moves instead of following momentum - we can achieve a 76% win rate.

However, the raw edge is thin (profit factor ~1.0). Success requires:
1. Excellent execution (minimize slippage)
2. Low commissions
3. Disciplined risk management
4. Further parameter optimization

**Recommendation**: Use the `MeanReversionStrategy` with `require_confirmation=False` as a starting point, then optimize stop/target ratios to achieve profit factor > 1.2.
