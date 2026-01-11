# Implementation Plan - Remaining Phases

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Data pipeline (Databento client, storage) |
| Phase 2 | ✅ Complete | Backtesting framework (strategies, engine, metrics) |
| Phase 3 | ✅ Complete | Risk management (position sizing, circuit breakers) |
| Phase 4 | ❌ TODO | Paper trading (IB) |
| Phase 5 | 🔄 NEXT | Backtest visualization & results dashboard |
| Phase 6 | ❌ TODO | AWS infrastructure |
| Phase 7 | ❌ Future | Rust execution engine (for millisecond trading) |

## Data Status ✅

Historical data downloaded and converted to Parquet:

| Schema | Date Range | Rows | Size |
|--------|------------|------|------|
| OHLCV-1m | 2023-12-10 to 2026-01-08 | 1,663,134 | 35 MB |
| OHLCV-1h | 2023-12-10 to 2026-01-08 | ~27,000 | ~1 MB |
| OHLCV-1d | 2023-12-10 to 2026-01-08 | ~750 | ~50 KB |
| OHLCV-1s | 2025-01-09 to 2026-01-08 | ~15M | ~500 MB |

Location: `data/raw/MCL_FUT/{year}/{date}_ohlcv-{schema}.parquet`

---

## Phase 3: Risk Management ✅ COMPLETE

### Files Created

| File | Purpose |
|------|---------|
| `src/risk/position_sizer.py` | Position sizing (fixed fractional, volatility-based, Kelly) |
| `src/risk/risk_manager.py` | Risk limits, circuit breakers, trade validation |
| `src/risk/stop_loss.py` | Stop-loss strategies (fixed, trailing, ATR-based) |
| `src/risk/__init__.py` | Module exports |
| `src/risk/CLAUDE.md` | Module documentation |

### Key Features Implemented

- **PositionSizer**: Fixed fractional, volatility-adjusted, Kelly criterion sizing
- **RiskManager**: Pre-trade validation, daily loss limits, circuit breakers
- **StopLossManager**: Fixed %, fixed $, ATR-based, trailing stops

### Backtest Engine Integration

The backtest engine now supports risk management via config:

```python
from src.backtest.engine import run_backtest

result = run_backtest(
    data=ohlcv_data,
    strategy=momentum_strategy,
    use_risk_manager=True,      # Enable risk management (default)
    risk_per_trade=0.02,        # 2% risk per trade
    daily_loss_limit=0.05,      # 5% daily loss limit
    max_position_contracts=5,   # Max 5 contracts
)
```

---

## Phase 4: Paper Trading (Interactive Brokers)

### Files to Create

| File | Purpose |
|------|---------|
| `src/execution/ib_client.py` | IB TWS/Gateway connection |
| `src/execution/order_manager.py` | Order lifecycle management |
| `src/execution/execution_engine.py` | Signal-to-order orchestration |
| `src/execution/__init__.py` | Module exports |
| `src/execution/CLAUDE.md` | Module documentation |
| `scripts/run_paper_trading.py` | Paper trading CLI |

### ib_client.py

Use `ib-async` library (maintained fork of ib_insync).

```python
from ib_async import IB, Future, MarketOrder, LimitOrder

class IBClient:
    def __init__(self, host, port, client_id):
        self.ib = IB()

    async def connect(self):
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)

    def get_mcl_contract(self, expiry=None) -> Future:
        return Future("MCL", expiry or self._next_expiry(), "NYMEX")

    async def place_market_order(self, contract, quantity, action) -> Trade
    async def place_limit_order(self, contract, quantity, action, price) -> Trade
    async def cancel_order(self, order_id)
    async def get_positions(self) -> list[Position]
    async def get_account_summary(self) -> dict
```

### order_manager.py

```python
class OrderManager:
    def create_entry_order(self, signal, price, quantity) -> Order
    def create_exit_order(self, position, price) -> Order
    def create_stop_order(self, position, stop_price) -> Order
    def track_fill(self, trade) -> None
    def reconcile_positions(self) -> None
```

### execution_engine.py

```python
class ExecutionEngine:
    def __init__(self, ib_client, risk_manager, strategy):
        ...

    async def run(self, data_feed):
        """Main trading loop"""
        while True:
            bar = await data_feed.next()
            signal = self.strategy.generate_signal(bar)
            if self.risk_manager.check_trade_allowed(signal):
                await self.execute_signal(signal)

    async def execute_signal(self, signal):
        """Convert signal to orders"""
```

### IB Connection Settings

| Mode | TWS Port | Gateway Port |
|------|----------|--------------|
| Paper | 7497 | 4002 |
| Live | 7496 | 4001 |

---

## Phase 5: Backtest Visualization & Results Dashboard (NEXT)

### Overview

Create a comprehensive backtest results visualization system that:
1. Opens an interactive window with charts after each backtest run
2. Saves detailed results to structured files for AI analysis
3. Enables strategy optimization through data-driven insights

### Files to Create

| File | Purpose |
|------|---------|
| `src/visualization/charts.py` | Interactive Plotly charts (equity curve, drawdown, trades) |
| `src/visualization/report.py` | Generate HTML/PDF reports |
| `src/visualization/window.py` | Pop-up visualization window |
| `src/visualization/__init__.py` | Module exports |
| `src/visualization/CLAUDE.md` | Module documentation |
| `results/` | Directory for all backtest results |

### Results Directory Structure

```
results/
├── backtests/
│   └── {timestamp}_{strategy}_{symbol}/
│       ├── summary.json          # Key metrics for AI analysis
│       ├── trades.csv            # All trades with full details
│       ├── equity_curve.csv      # Timestamped equity values
│       ├── daily_pnl.csv         # Daily P&L breakdown
│       ├── drawdowns.csv         # Drawdown periods
│       ├── signals.csv           # All signals generated
│       ├── parameters.json       # Strategy parameters used
│       ├── report.html           # Human-readable report
│       └── charts/
│           ├── equity_curve.html
│           ├── drawdown.html
│           ├── trade_distribution.html
│           └── monthly_returns.html
├── optimizations/
│   └── {timestamp}_{strategy}/
│       ├── parameter_sweep.csv   # All parameter combinations tested
│       ├── best_params.json      # Optimal parameters found
│       └── sensitivity.csv       # Parameter sensitivity analysis
└── comparisons/
    └── {timestamp}/
        ├── strategy_comparison.csv
        └── comparison_report.html
```

### summary.json Schema (AI-Friendly)

```json
{
  "run_id": "2026-01-10_143052_ema_MCL_FUT",
  "timestamp": "2026-01-10T14:30:52Z",
  "symbol": "MCL_FUT",
  "strategy": {
    "name": "EMA Crossover",
    "class": "MomentumStrategy",
    "parameters": {
      "fast_period": 20,
      "slow_period": 50,
      "atr_period": 14
    }
  },
  "data_range": {
    "start": "2023-12-10",
    "end": "2026-01-08",
    "trading_days": 649,
    "total_bars": 1663134
  },
  "performance": {
    "total_return_pct": 12.5,
    "total_return_usd": 625.0,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.8,
    "max_drawdown_pct": 8.5,
    "max_drawdown_usd": 425.0,
    "calmar_ratio": 1.47,
    "win_rate": 0.55,
    "profit_factor": 1.4,
    "avg_trade_pnl": 15.5,
    "avg_winner": 45.0,
    "avg_loser": -25.0,
    "largest_winner": 150.0,
    "largest_loser": -80.0,
    "total_trades": 120,
    "winning_trades": 66,
    "losing_trades": 54
  },
  "risk_metrics": {
    "risk_per_trade": 0.02,
    "daily_loss_limit": 0.05,
    "trades_rejected": 5,
    "circuit_breaker_triggered": 0
  },
  "time_analysis": {
    "avg_hold_time_minutes": 45,
    "best_hour": 10,
    "worst_hour": 15,
    "best_day_of_week": "Tuesday",
    "worst_day_of_week": "Friday"
  },
  "recommendations": [
    "Consider tightening stops - avg loser exceeds 2x risk target",
    "Performance degrades after 14:00 - consider time filter",
    "Win rate drops on Fridays - reduce position size"
  ]
}
```

### Visualization Window Features

```python
# src/visualization/window.py
class BacktestViewer:
    """Interactive backtest results viewer."""

    def show(self, result: BacktestResult):
        """Open interactive visualization window."""
        # Uses Plotly Dash or PyQt for desktop window

    def create_dashboard(self, result: BacktestResult) -> dash.Dash:
        """Create Dash app with:
        - Equity curve with trade markers
        - Drawdown chart
        - Monthly returns heatmap
        - Trade P&L distribution
        - Rolling Sharpe ratio
        - Win/loss streaks
        - Time-of-day analysis
        - Interactive trade table
        """
```

### CLI Integration

Update `scripts/run_backtest.py`:

```python
@click.option("--show", is_flag=True, help="Open visualization window")
@click.option("--save-results", is_flag=True, default=True, help="Save detailed results")
@click.option("--results-dir", default="results/backtests", help="Results directory")

# After backtest completes:
if save_results:
    result_saver.save_all(result, results_dir)

if show:
    viewer = BacktestViewer()
    viewer.show(result)  # Opens interactive window
```

### AI Optimization Support

```python
# src/visualization/optimizer.py
class StrategyOptimizer:
    """Generate optimization insights for AI analysis."""

    def analyze_results(self, results_dir: str) -> dict:
        """Analyze all backtests and generate recommendations."""

    def parameter_sensitivity(self, param_name: str) -> pd.DataFrame:
        """Show how changing one parameter affects performance."""

    def generate_next_experiments(self) -> list[dict]:
        """Suggest next parameter combinations to test."""
```

### alerts.py (Future - Phase 5b)

```python
class AlertManager:
    def send_slack(self, message, channel)
    def send_email(self, subject, body, recipient)
    def alert_on_drawdown(self, current_dd, threshold)
    def alert_on_trade(self, trade)
    def alert_on_error(self, error)
```

---

## Phase 6: AWS Infrastructure

### Terraform Modules to Create

```
infrastructure/terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── provider.tf
├── modules/
│   ├── networking/
│   │   ├── main.tf      # VPC, subnets, IGW
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── ec2/
│   │   ├── main.tf      # EC2 in Chicago Local Zone
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── security/
│       ├── main.tf      # Security groups, IAM
│       ├── variables.tf
│       └── outputs.tf
└── environments/
    ├── dev/terraform.tfvars
    └── prod/terraform.tfvars
```

### Docker Files to Create

```
infrastructure/docker/
├── Dockerfile           # Multi-stage production build
├── Dockerfile.dev       # Development with hot reload
├── docker-compose.yml   # Local: app + IB Gateway
└── docker-compose.prod.yml
```

### Key Terraform Resources

```hcl
# Enable Chicago Local Zone
resource "aws_ec2_availability_zone_group" "chicago" {
  group_name    = "us-east-1-chi-2"
  opt_in_status = "opted-in"
}

# Subnet in Local Zone
resource "aws_subnet" "chicago_trading" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "us-east-1-chi-2a"
}

# EC2 Instance
resource "aws_instance" "trading" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "c6in.xlarge"
  subnet_id              = aws_subnet.chicago_trading.id
  vpc_security_group_ids = [aws_security_group.trading.id]
}
```

---

## Phase 7: Rust Execution Engine (Future)

### Why Rust?

Python limitations for HFT:
- GIL prevents true parallelism
- Garbage collection pauses
- ~100μs minimum latency

Rust advantages:
- Zero-cost abstractions
- No GC, predictable latency
- ~1-10μs achievable
- Memory safety without runtime cost

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Python Layer                       │
│  - Research, backtesting, monitoring                │
│  - Strategy development                              │
│  - Dashboard UI                                      │
└─────────────────────────────────────────────────────┘
                         │
                         │ PyO3 FFI
                         ▼
┌─────────────────────────────────────────────────────┐
│                   Rust Layer                         │
│  - Market data ingestion (TCP/multicast)            │
│  - Order book management                             │
│  - Signal generation (hot path)                      │
│  - Order routing                                     │
│  - FIX protocol                                      │
└─────────────────────────────────────────────────────┘
```

### Key Rust Crates

- `tokio` - Async runtime
- `crossbeam` - Lock-free data structures
- `mio` - Low-level I/O
- `pyo3` - Python bindings
- `serde` - Serialization

### Target Latencies

| Component | Python | Rust Target |
|-----------|--------|-------------|
| Signal generation | 1-10ms | <100μs |
| Order submission | 10-50ms | <1ms |
| Full round-trip | 50-200ms | <5ms |

---

## Execution Priority

1. ~~**Phase 3** (Risk Management) - Required before paper trading~~ ✅ COMPLETE
2. **Phase 5** (Visualization) - **NEXT** - Analyze backtest results, optimize strategy
3. **Phase 4** (Paper Trading) - Validate strategy in real market (after profitable backtest)
4. **Phase 6** (Infrastructure) - Deploy to AWS for lower latency
5. **Phase 7** (Rust) - Only after strategy is profitable in paper trading

---

## Testing Requirements

Each phase should include:

1. Unit tests in `tests/unit/`
2. Integration tests in `tests/integration/`
3. Manual verification steps documented

### Test Commands

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific module
pytest tests/unit/test_risk_manager.py
```
