"""Central risk management and enforcement."""

from dataclasses import dataclass
from datetime import date, datetime

from config.logging_config import get_logger
from src.risk.position_sizer import PositionSizer
from src.strategy.base_strategy import Signal

logger = get_logger(__name__)


@dataclass
class TradeValidation:
    """Result of pre-trade validation."""

    allowed: bool
    reason: str
    position_size: int


class RiskManager:
    """Risk management and enforcement for MCL futures trading.

    Responsibilities:
    - Pre-trade validation (daily limits, position size)
    - Position sizing using PositionSizer
    - Daily P&L tracking
    - Circuit breaker activation

    Usage:
        risk_manager = RiskManager(
            initial_capital=5000,
            risk_per_trade=0.02,
            daily_loss_limit=0.05,
        )

        validation = risk_manager.validate_trade(
            signal=Signal.LONG,
            entry_price=75.50,
            stop_loss_price=75.30,
            current_equity=5000,
            timestamp=datetime.now(),
        )

        if validation.allowed:
            # Execute trade with validation.position_size contracts
    """

    def __init__(
        self,
        initial_capital: float = 5_000.0,
        max_position_contracts: int = 5,
        risk_per_trade: float = 0.02,
        daily_loss_limit: float = 0.05,
        contract_multiplier: float = 100.0,
        tick_size: float = 0.01,
    ):
        """Initialize risk manager.

        Args:
            initial_capital: Starting account capital
            max_position_contracts: Maximum contracts per position
            risk_per_trade: Risk per trade as fraction (e.g., 0.02 for 2%)
            daily_loss_limit: Daily loss limit as fraction (e.g., 0.05 for 5%)
            contract_multiplier: Contract multiplier (MCL = 100)
            tick_size: Minimum price increment (MCL = $0.01)
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if max_position_contracts <= 0:
            raise ValueError("max_position_contracts must be positive")
        if risk_per_trade <= 0 or risk_per_trade > 0.10:
            raise ValueError("risk_per_trade must be between 0 and 0.10 (10%)")
        if daily_loss_limit <= 0 or daily_loss_limit > 0.20:
            raise ValueError("daily_loss_limit must be between 0 and 0.20 (20%)")

        self.initial_capital = initial_capital
        self.max_position_contracts = max_position_contracts
        self.risk_per_trade = risk_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.contract_multiplier = contract_multiplier
        self.tick_size = tick_size

        # Initialize position sizer
        self.position_sizer = PositionSizer(
            contract_multiplier=contract_multiplier,
            tick_size=tick_size,
            max_contracts=max_position_contracts,
        )

        # Daily tracking state
        self._day_start_equity: float = initial_capital
        self._daily_pnl: float = 0.0
        self._current_date: date | None = None
        self._trades_today: int = 0

        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_reason: str | None = None

        logger.info(
            "risk_manager_initialized",
            initial_capital=initial_capital,
            max_position_contracts=max_position_contracts,
            risk_per_trade=risk_per_trade,
            daily_loss_limit=daily_loss_limit,
        )

    def validate_trade(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss_price: float | None,
        current_equity: float,
        timestamp: datetime,
    ) -> TradeValidation:
        """Validate a trade before execution.

        Checks:
        1. Circuit breaker not active
        2. Daily loss limit not exceeded
        3. Valid stop-loss provided
        4. Position size > 0

        Args:
            signal: Trading signal (LONG, SHORT, FLAT)
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price (None rejects trade)
            current_equity: Current account equity
            timestamp: Current timestamp

        Returns:
            TradeValidation with allowed, reason, and position_size
        """
        # Update daily tracking if new day
        self._check_new_day(timestamp, current_equity)

        # Check for FLAT signal (no trade needed)
        if signal == Signal.FLAT:
            return TradeValidation(
                allowed=False,
                reason="FLAT signal - no trade",
                position_size=0,
            )

        # Check circuit breaker (no logging here - already logged when triggered)
        if self._circuit_breaker_active:
            return TradeValidation(
                allowed=False,
                reason=f"circuit breaker active: {self._circuit_breaker_reason}",
                position_size=0,
            )

        # Check daily loss limit
        daily_limit_ok, daily_reason = self.check_daily_limit(current_equity, timestamp)
        if not daily_limit_ok:
            self.trigger_circuit_breaker(daily_reason)
            return TradeValidation(
                allowed=False,
                reason=daily_reason,
                position_size=0,
            )

        # Check stop-loss provided
        if stop_loss_price is None:
            return TradeValidation(
                allowed=False,
                reason="stop-loss price required",
                position_size=0,
            )

        # Calculate position size
        position_size = self.get_position_size(
            signal=signal,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            current_equity=current_equity,
        )

        # Reject if position size is 0
        if position_size == 0:
            return TradeValidation(
                allowed=False,
                reason="position size calculated as 0 (stop too wide or equity too low)",
                position_size=0,
            )

        logger.info(
            "trade_validated",
            signal=signal.name if hasattr(signal, "name") else str(signal),
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            position_size=position_size,
            current_equity=current_equity,
        )

        return TradeValidation(
            allowed=True,
            reason="trade approved",
            position_size=position_size,
        )

    def get_position_size(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss_price: float,
        current_equity: float,
    ) -> int:
        """Calculate position size using fixed fractional method.

        Args:
            signal: Trading signal
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price
            current_equity: Current account equity

        Returns:
            Number of contracts to trade (0 if invalid)
        """
        if signal == Signal.FLAT:
            return 0

        result = self.position_sizer.fixed_fractional(
            equity=current_equity,
            risk_per_trade=self.risk_per_trade,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
        )

        if result.reason:
            logger.debug(
                "position_size_note",
                contracts=result.contracts,
                reason=result.reason,
            )

        return result.contracts

    def check_daily_limit(
        self,
        current_equity: float,
        timestamp: datetime,
    ) -> tuple[bool, str]:
        """Check if daily loss limit has been exceeded.

        Args:
            current_equity: Current account equity
            timestamp: Current timestamp

        Returns:
            Tuple of (trading_allowed, reason)
        """
        self._check_new_day(timestamp, current_equity)

        # Calculate current daily loss
        daily_loss = self._day_start_equity - current_equity
        daily_loss_pct = daily_loss / self._day_start_equity if self._day_start_equity > 0 else 0

        if daily_loss_pct >= self.daily_loss_limit:
            reason = (
                f"daily loss limit exceeded: "
                f"${daily_loss:.2f} ({daily_loss_pct:.1%}) >= "
                f"{self.daily_loss_limit:.1%} limit"
            )
            logger.warning(
                "daily_limit_exceeded",
                day_start_equity=self._day_start_equity,
                current_equity=current_equity,
                daily_loss=daily_loss,
                daily_loss_pct=daily_loss_pct,
                limit=self.daily_loss_limit,
            )
            return False, reason

        return True, "within daily limit"

    def update_daily_pnl(
        self,
        realized_pnl: float,
        timestamp: datetime,
    ) -> None:
        """Update daily P&L tracking after a trade closes.

        Args:
            realized_pnl: Realized P&L from closed trade
            timestamp: Trade close timestamp
        """
        self._check_new_day(timestamp, self._day_start_equity + self._daily_pnl)

        self._daily_pnl += realized_pnl
        self._trades_today += 1

        logger.debug(
            "daily_pnl_updated",
            realized_pnl=realized_pnl,
            daily_pnl=self._daily_pnl,
            trades_today=self._trades_today,
        )

    def _check_new_day(self, timestamp: datetime, current_equity: float) -> None:
        """Check if it's a new trading day and reset metrics.

        Args:
            timestamp: Current timestamp
            current_equity: Current equity for new day start
        """
        current_day = timestamp.date()

        if self._current_date is None or current_day > self._current_date:
            if self._current_date is not None:
                logger.info(
                    "new_trading_day",
                    previous_date=self._current_date.isoformat(),
                    new_date=current_day.isoformat(),
                    previous_daily_pnl=self._daily_pnl,
                    trades_previous_day=self._trades_today,
                )

            self._current_date = current_day
            self._day_start_equity = current_equity
            self._daily_pnl = 0.0
            self._trades_today = 0

            # Reset circuit breaker on new day
            if self._circuit_breaker_active:
                logger.info(
                    "circuit_breaker_reset_new_day",
                    previous_reason=self._circuit_breaker_reason,
                )
                self._circuit_breaker_active = False
                self._circuit_breaker_reason = None

    def reset_daily_metrics(self, timestamp: datetime, equity: float) -> None:
        """Manually reset daily tracking metrics.

        Args:
            timestamp: Timestamp for the reset
            equity: Equity to use as day start
        """
        self._current_date = timestamp.date()
        self._day_start_equity = equity
        self._daily_pnl = 0.0
        self._trades_today = 0

        logger.info(
            "daily_metrics_reset",
            date=self._current_date.isoformat(),
            day_start_equity=equity,
        )

    def trigger_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker to halt all trading.

        Circuit breaker stops all trading until manually reset or new day.

        Args:
            reason: Reason for circuit breaker activation
        """
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason

        logger.critical(
            "circuit_breaker_triggered",
            reason=reason,
            day_start_equity=self._day_start_equity,
            daily_pnl=self._daily_pnl,
            trades_today=self._trades_today,
        )

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker.

        Use with caution - should only be done after reviewing
        the reason for the trigger.
        """
        if self._circuit_breaker_active:
            logger.warning(
                "circuit_breaker_manual_reset",
                previous_reason=self._circuit_breaker_reason,
            )

        self._circuit_breaker_active = False
        self._circuit_breaker_reason = None

    @property
    def is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        return self._circuit_breaker_active

    @property
    def circuit_breaker_reason(self) -> str | None:
        """Get the reason for circuit breaker activation."""
        return self._circuit_breaker_reason

    @property
    def daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._daily_pnl

    @property
    def trades_today(self) -> int:
        """Get number of trades today."""
        return self._trades_today

    @property
    def day_start_equity(self) -> float:
        """Get equity at start of current trading day."""
        return self._day_start_equity

    def get_status(self) -> dict:
        """Get current risk manager status.

        Returns:
            Dictionary with current state and metrics
        """
        return {
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "current_date": self._current_date.isoformat() if self._current_date else None,
            "day_start_equity": self._day_start_equity,
            "daily_pnl": self._daily_pnl,
            "trades_today": self._trades_today,
            "risk_per_trade": self.risk_per_trade,
            "daily_loss_limit": self.daily_loss_limit,
            "max_position_contracts": self.max_position_contracts,
        }
