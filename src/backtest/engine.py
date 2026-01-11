"""Vectorized backtesting engine."""

from dataclasses import dataclass, field

import pandas as pd

from config.logging_config import get_logger
from src.backtest.metrics import PerformanceMetrics, calculate_metrics
from src.backtest.portfolio import Portfolio, TradeDirection
from src.risk.risk_manager import RiskManager
from src.risk.scalping_risk import ScalpingRiskConfig, ScalpingRiskManager
from src.strategy.base_strategy import BaseStrategy, Signal
from src.strategy.scalping_strategy import ScalpingStrategy

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""

    initial_capital: float = 5_000.0
    contract_multiplier: float = 100.0  # MCL = 100 barrels
    tick_size: float = 0.01  # MCL tick size
    position_size: int = 1  # Contracts per trade (used when risk manager disabled)
    slippage_ticks: int = 1  # Slippage in ticks

    # IBKR Commission Structure (per contract, per side)
    # Total cost = ibkr_commission + exchange_fee + regulatory_fee + give_up_fee
    ibkr_commission: float = 0.85  # IBKR tiered commission (≤1000 contracts/month)
    exchange_fee: float = 0.50  # NYMEX fee recovery: $0.50 MCL, $1.50 CL (Tier 1)
    regulatory_fee: float = 0.02  # NYMEX regulatory fee recovery (Tier 1)
    give_up_fee: float = 0.00  # Give-up surcharge (waived when IBKR is prime+executing)

    # Legacy field for backwards compatibility (computed from above)
    commission_per_contract: float = field(init=False)

    # Risk management settings
    use_risk_manager: bool = True  # Enable/disable risk management
    max_position_contracts: int = 5  # Maximum contracts per position
    risk_per_trade: float = 0.02  # Risk 2% of equity per trade
    daily_loss_limit: float = 0.05  # Stop trading at 5% daily loss

    def __post_init__(self):
        """Calculate total commission per contract from IBKR fee components."""
        self.commission_per_contract = (
            self.ibkr_commission + self.exchange_fee + self.regulatory_fee + self.give_up_fee
        )


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: pd.DataFrame
    signals: pd.Series
    strategy: BaseStrategy
    config: BacktestConfig


class BacktestEngine:
    """Vectorized backtesting engine.

    Processes OHLCV data with a strategy to generate signals,
    then simulates trade execution with realistic costs.
    """

    def __init__(self, config: BacktestConfig | None = None):
        """Initialize backtest engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()

        # Initialize risk manager if enabled
        if self.config.use_risk_manager:
            self.risk_manager: RiskManager | None = RiskManager(
                initial_capital=self.config.initial_capital,
                max_position_contracts=self.config.max_position_contracts,
                risk_per_trade=self.config.risk_per_trade,
                daily_loss_limit=self.config.daily_loss_limit,
                contract_multiplier=self.config.contract_multiplier,
                tick_size=self.config.tick_size,
            )
        else:
            self.risk_manager = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with datetime index
            strategy: Strategy instance to test
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with performance metrics and details
        """
        logger.info(
            "starting_backtest",
            strategy=strategy.name,
            data_rows=len(data),
            initial_capital=self.config.initial_capital,
            use_risk_manager=self.config.use_risk_manager,
        )

        # Filter data by date range
        df = data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if df.empty:
            raise ValueError("No data in specified date range")

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Generate signals
        signals = strategy.generate_signals(df)
        logger.info("signals_generated", total=len(signals), non_zero=(signals != 0).sum())

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            contract_multiplier=self.config.contract_multiplier,
            commission_per_contract=self.config.commission_per_contract,
        )

        # Run simulation (use scalping mode for ScalpingStrategy)
        if isinstance(strategy, ScalpingStrategy):
            logger.info("using_scalping_simulation_mode")
            self._simulate_scalping(df, signals, strategy, portfolio)
        else:
            self._simulate(df, signals, strategy, portfolio)

        # Get results
        equity_curve = portfolio.get_equity_series()
        trades_df = portfolio.get_trades_df()

        # Calculate metrics
        metrics = calculate_metrics(
            equity_series=equity_curve,
            trades_df=trades_df,
            initial_capital=self.config.initial_capital,
            signals=signals,
        )

        logger.info(
            "backtest_complete",
            total_return=metrics.total_return,
            total_trades=metrics.total_trades,
            sharpe=metrics.sharpe_ratio,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades_df,
            signals=signals,
            strategy=strategy,
            config=self.config,
        )

    def _simulate(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        strategy: BaseStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Simulate trading based on signals.

        Args:
            data: OHLCV data
            signals: Trading signals
            strategy: Strategy for stop-loss calculation
            portfolio: Portfolio to track state
        """
        current_signal = Signal.FLAT
        slippage = self.config.slippage_ticks * self.config.tick_size
        trades_rejected = 0

        for i, (timestamp, row) in enumerate(data.iterrows()):
            close_price = row["close"]
            high_price = row["high"]
            low_price = row["low"]

            new_signal = Signal(signals.iloc[i])

            # Check stop-loss if in position
            if portfolio.position is not None:
                stop_hit = False
                exit_price = close_price

                if portfolio.position.stop_loss is not None:
                    if portfolio.position.direction == TradeDirection.LONG:
                        if low_price <= portfolio.position.stop_loss:
                            stop_hit = True
                            exit_price = portfolio.position.stop_loss - slippage
                    else:  # SHORT
                        if high_price >= portfolio.position.stop_loss:
                            stop_hit = True
                            exit_price = portfolio.position.stop_loss + slippage

                if stop_hit:
                    trade = portfolio.close_position(exit_price, timestamp)
                    # Update risk manager with realized P&L
                    if self.risk_manager is not None and trade is not None:
                        self.risk_manager.update_daily_pnl(trade.pnl, timestamp)
                    current_signal = Signal.FLAT

            # Handle signal changes
            if new_signal != current_signal:
                # Close existing position if signal changes
                if portfolio.position is not None:
                    if new_signal == Signal.FLAT or new_signal != current_signal:
                        # Apply slippage based on direction
                        if portfolio.position.direction == TradeDirection.LONG:
                            exit_price = close_price - slippage
                        else:
                            exit_price = close_price + slippage
                        trade = portfolio.close_position(exit_price, timestamp)
                        # Update risk manager with realized P&L
                        if self.risk_manager is not None and trade is not None:
                            self.risk_manager.update_daily_pnl(trade.pnl, timestamp)

                # Open new position
                if new_signal != Signal.FLAT and portfolio.position is None:
                    direction = (
                        TradeDirection.LONG if new_signal == Signal.LONG else TradeDirection.SHORT
                    )

                    # Apply slippage to entry
                    if direction == TradeDirection.LONG:
                        entry_price = close_price + slippage
                    else:
                        entry_price = close_price - slippage

                    # Calculate stop-loss
                    stop_loss = strategy.get_stop_loss(
                        data.iloc[: i + 1], new_signal, entry_price
                    )

                    # Risk management validation
                    if self.risk_manager is not None:
                        validation = self.risk_manager.validate_trade(
                            signal=new_signal,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss,
                            current_equity=portfolio.get_equity(close_price),
                            timestamp=timestamp,
                        )

                        if not validation.allowed:
                            logger.debug(
                                "trade_rejected",
                                reason=validation.reason,
                                timestamp=timestamp,
                            )
                            trades_rejected += 1
                            # Don't update current_signal - stay flat
                            portfolio.record_equity(timestamp, close_price)
                            continue

                        position_size = validation.position_size
                    else:
                        position_size = self.config.position_size

                    portfolio.open_position(
                        direction=direction,
                        price=entry_price,
                        quantity=position_size,
                        timestamp=timestamp,
                        stop_loss=stop_loss,
                    )

                current_signal = new_signal

            # Record equity
            portfolio.record_equity(timestamp, close_price)

        # Close any remaining position at end
        if portfolio.position is not None:
            final_price = data.iloc[-1]["close"]
            if portfolio.position.direction == TradeDirection.LONG:
                final_price -= slippage
            else:
                final_price += slippage
            trade = portfolio.close_position(final_price, data.index[-1])
            if self.risk_manager is not None and trade is not None:
                self.risk_manager.update_daily_pnl(trade.pnl, data.index[-1])

        if trades_rejected > 0:
            logger.info("simulation_trades_rejected", count=trades_rejected)

    def _simulate_scalping(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        strategy: ScalpingStrategy,
        portfolio: Portfolio,
    ) -> None:
        """Simulate scalping strategy with multi-level exits.

        Supports:
        - Partial position exits at multiple profit targets
        - Breakeven stop management
        - Time-based exits
        - Trailing stops

        Args:
            data: OHLCV data
            signals: Trading signals
            strategy: ScalpingStrategy instance
            portfolio: Portfolio to track state
        """
        current_signal = Signal.FLAT
        slippage = self.config.slippage_ticks * self.config.tick_size
        trades_rejected = 0
        partial_exits = 0

        # Use scalping risk manager if available
        scalping_risk = None
        if self.risk_manager is not None:
            # Create scalping risk manager with same base settings
            scalping_config = ScalpingRiskConfig(
                initial_capital=self.config.initial_capital,
                max_position_contracts=self.config.max_position_contracts,
                risk_per_trade=min(self.config.risk_per_trade, 0.005),  # Cap at 0.5% for scalping
                daily_loss_limit=min(self.config.daily_loss_limit, 0.03),  # Cap at 3%
                max_drawdown_limit=0.10,  # 10% max drawdown
                contract_multiplier=self.config.contract_multiplier,
                tick_size=self.config.tick_size,
            )
            scalping_risk = ScalpingRiskManager(scalping_config)

        for i, (timestamp, row) in enumerate(data.iterrows()):
            close_price = row["close"]
            high_price = row["high"]
            low_price = row["low"]

            new_signal = Signal(signals.iloc[i])

            # Check exit conditions if in position
            if portfolio.position is not None:
                # Get current ATR for exit calculations
                current_atr = strategy.current_atr
                if current_atr <= 0:
                    current_atr = close_price * 0.002  # Fallback

                # Check all exit conditions
                exit_result = strategy.check_exit_conditions(
                    position_direction=Signal.LONG if portfolio.position.direction == TradeDirection.LONG else Signal.SHORT,
                    entry_price=portfolio.position.entry_price,
                    entry_time=portfolio.position.entry_time,
                    current_price=close_price,
                    current_high=high_price,
                    current_low=low_price,
                    current_time=timestamp,
                    current_atr=current_atr,
                    position_state=portfolio.position.state,
                    current_stop=portfolio.position.stop_loss or 0,
                    tick_size=self.config.tick_size,
                )

                if exit_result is not None:
                    action = exit_result.get("action")

                    if action == "close_full":
                        exit_price = exit_result["price"]
                        # Apply slippage
                        if portfolio.position.direction == TradeDirection.LONG:
                            exit_price -= slippage
                        else:
                            exit_price += slippage

                        trade = portfolio.close_position(exit_price, timestamp)

                        # Update risk manager
                        if scalping_risk is not None and trade is not None:
                            scalping_risk.record_trade_result(
                                pnl=trade.pnl,
                                timestamp=timestamp,
                                current_equity=portfolio.get_equity(close_price),
                            )

                        current_signal = Signal.FLAT

                    elif action == "close_partial":
                        exit_price = exit_result["price"]
                        percentage = exit_result["percentage"]

                        # Apply slippage
                        if portfolio.position.direction == TradeDirection.LONG:
                            exit_price -= slippage
                        else:
                            exit_price += slippage

                        partial_exit = portfolio.close_partial(
                            price=exit_price,
                            timestamp=timestamp,
                            percentage=percentage,
                            exit_type=exit_result.get("exit_type", "partial"),
                        )

                        if partial_exit is not None:
                            partial_exits += 1

                            # Update stop and state
                            if "new_stop" in exit_result:
                                portfolio.update_stop_loss(exit_result["new_stop"])
                            if "new_state" in exit_result:
                                portfolio.update_position_state(exit_result["new_state"])

                            # Check if position fully closed
                            if portfolio.position is None:
                                current_signal = Signal.FLAT

                    elif action == "update_stop":
                        # Just update stop loss and state
                        if "new_stop" in exit_result:
                            portfolio.update_stop_loss(exit_result["new_stop"])
                        if "new_state" in exit_result:
                            portfolio.update_position_state(exit_result["new_state"])

            # Handle signal changes (only if not already in position or signal reverses)
            if new_signal != current_signal and portfolio.position is None:
                # Open new position
                if new_signal != Signal.FLAT:
                    direction = (
                        TradeDirection.LONG if new_signal == Signal.LONG else TradeDirection.SHORT
                    )

                    # Apply slippage to entry
                    if direction == TradeDirection.LONG:
                        entry_price = close_price + slippage
                    else:
                        entry_price = close_price - slippage

                    # Calculate stop-loss
                    stop_loss = strategy.get_stop_loss(
                        data.iloc[: i + 1], new_signal, entry_price
                    )

                    # Risk management validation
                    if scalping_risk is not None:
                        validation = scalping_risk.validate_scalping_trade(
                            signal=new_signal,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss,
                            current_equity=portfolio.get_equity(close_price),
                            timestamp=timestamp,
                        )

                        if not validation.allowed:
                            logger.debug(
                                "scalping_trade_rejected",
                                reason=validation.reason,
                                timestamp=timestamp,
                            )
                            trades_rejected += 1
                            portfolio.record_equity(timestamp, close_price)
                            continue

                        position_size = validation.position_size
                    else:
                        position_size = self.config.position_size

                    portfolio.open_position(
                        direction=direction,
                        price=entry_price,
                        quantity=position_size,
                        timestamp=timestamp,
                        stop_loss=stop_loss,
                    )

                    current_signal = new_signal

            # Record equity
            portfolio.record_equity(timestamp, close_price)

        # Close any remaining position at end
        if portfolio.position is not None:
            final_price = data.iloc[-1]["close"]
            if portfolio.position.direction == TradeDirection.LONG:
                final_price -= slippage
            else:
                final_price += slippage
            trade = portfolio.close_position(final_price, data.index[-1])
            if scalping_risk is not None and trade is not None:
                scalping_risk.record_trade_result(
                    pnl=trade.pnl,
                    timestamp=data.index[-1],
                    current_equity=portfolio.get_equity(final_price),
                )

        if trades_rejected > 0:
            logger.info("scalping_trades_rejected", count=trades_rejected)
        if partial_exits > 0:
            logger.info("scalping_partial_exits", count=partial_exits)


def run_backtest(
    data: pd.DataFrame,
    strategy: BaseStrategy,
    initial_capital: float = 5_000.0,
    contract_multiplier: float = 100.0,
    commission: float = 2.25,
    slippage_ticks: int = 1,
    use_risk_manager: bool = True,
    risk_per_trade: float = 0.02,
    daily_loss_limit: float = 0.05,
    max_position_contracts: int = 5,
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        data: OHLCV DataFrame
        strategy: Strategy to test
        initial_capital: Starting capital
        contract_multiplier: Contract multiplier (MCL = 100)
        commission: Commission per contract
        slippage_ticks: Slippage in ticks
        use_risk_manager: Enable risk management (default True)
        risk_per_trade: Risk per trade as fraction (default 0.02 = 2%)
        daily_loss_limit: Daily loss limit as fraction (default 0.05 = 5%)
        max_position_contracts: Maximum contracts per position (default 5)

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
        commission_per_contract=commission,
        slippage_ticks=slippage_ticks,
        use_risk_manager=use_risk_manager,
        risk_per_trade=risk_per_trade,
        daily_loss_limit=daily_loss_limit,
        max_position_contracts=max_position_contracts,
    )

    engine = BacktestEngine(config)
    return engine.run(data, strategy)
