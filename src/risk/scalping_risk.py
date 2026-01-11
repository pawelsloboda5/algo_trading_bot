"""Enhanced risk management for scalping strategies.

Extends base RiskManager with:
- Maximum drawdown tracking and enforcement (10% default)
- Consecutive loss tracking with cooldown
- More conservative position sizing for HF trading
- Trade frequency limits
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from config.logging_config import get_logger
from src.risk.risk_manager import RiskManager, TradeValidation
from src.strategy.base_strategy import Signal

logger = get_logger(__name__)


@dataclass
class ScalpingRiskConfig:
    """Configuration for scalping risk management."""

    # Core risk settings
    initial_capital: float = 5_000.0
    max_position_contracts: int = 5
    risk_per_trade: float = 0.005  # 0.5% per trade (conservative for scalping)
    daily_loss_limit: float = 0.03  # 3% daily limit

    # Max drawdown settings
    max_drawdown_limit: float = 0.10  # 10% maximum drawdown
    drawdown_recovery_pct: float = 0.5  # Resume trading at 50% of max drawdown

    # Consecutive loss management
    max_consecutive_losses: int = 5  # Stop after 5 losses in a row
    cooldown_after_losses_seconds: int = 60  # 60 second cooldown after max losses

    # Trade frequency limits
    max_trades_per_hour: int = 30  # Max trades per hour
    min_seconds_between_trades: int = 5  # Minimum 5 seconds between trades

    # Contract settings
    contract_multiplier: float = 100.0  # MCL = 100 barrels
    tick_size: float = 0.01  # MCL tick = $0.01


class ScalpingRiskManager(RiskManager):
    """Enhanced risk manager for scalping with max drawdown enforcement.

    Key features:
    - Tracks peak equity and current drawdown
    - Stops trading when max drawdown (10%) is reached
    - Consecutive loss tracking with forced cooldown
    - Trade frequency limiting
    - More conservative position sizing

    Example:
        config = ScalpingRiskConfig(
            initial_capital=5000,
            max_drawdown_limit=0.10,  # 10% max drawdown
            max_consecutive_losses=5,
        )
        risk_manager = ScalpingRiskManager(config)

        # Validate each trade
        validation = risk_manager.validate_scalping_trade(
            signal=Signal.LONG,
            entry_price=75.50,
            stop_loss_price=75.45,  # Tight stop
            current_equity=4800,
            timestamp=datetime.now(),
        )
    """

    def __init__(self, config: ScalpingRiskConfig | None = None):
        """Initialize scalping risk manager.

        Args:
            config: Scalping risk configuration (uses defaults if None)
        """
        config = config or ScalpingRiskConfig()

        # Initialize parent RiskManager
        super().__init__(
            initial_capital=config.initial_capital,
            max_position_contracts=config.max_position_contracts,
            risk_per_trade=config.risk_per_trade,
            daily_loss_limit=config.daily_loss_limit,
            contract_multiplier=config.contract_multiplier,
            tick_size=config.tick_size,
        )

        self.config = config

        # Drawdown tracking
        self._peak_equity = config.initial_capital
        self._current_drawdown = 0.0
        self._max_drawdown_hit = False

        # Consecutive loss tracking
        self._consecutive_losses = 0
        self._last_trade_was_loss = False
        self._in_cooldown = False
        self._cooldown_end_time: datetime | None = None

        # Trade frequency tracking
        self._hourly_trades: list[datetime] = []
        self._last_trade_time: datetime | None = None

        logger.info(
            "scalping_risk_manager_initialized",
            max_drawdown_limit=config.max_drawdown_limit,
            max_consecutive_losses=config.max_consecutive_losses,
            risk_per_trade=config.risk_per_trade,
            max_trades_per_hour=config.max_trades_per_hour,
        )

    def validate_scalping_trade(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss_price: float | None,
        current_equity: float,
        timestamp: datetime,
    ) -> TradeValidation:
        """Validate a scalping trade with enhanced risk checks.

        Performs all standard validation plus:
        - Max drawdown check
        - Consecutive loss check
        - Cooldown check
        - Trade frequency check

        Args:
            signal: Trading signal
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price
            current_equity: Current account equity
            timestamp: Current timestamp

        Returns:
            TradeValidation with allowed, reason, and position_size
        """
        # Update peak equity and drawdown
        self._update_drawdown(current_equity)

        # Check max drawdown limit
        if self._max_drawdown_hit:
            return TradeValidation(
                allowed=False,
                reason=f"max drawdown limit reached: {self._current_drawdown:.1%} >= {self.config.max_drawdown_limit:.1%}",
                position_size=0,
            )

        # Check if in cooldown after consecutive losses
        if self._in_cooldown:
            if timestamp < self._cooldown_end_time:
                remaining = (self._cooldown_end_time - timestamp).total_seconds()
                return TradeValidation(
                    allowed=False,
                    reason=f"in cooldown after {self.config.max_consecutive_losses} consecutive losses ({remaining:.0f}s remaining)",
                    position_size=0,
                )
            else:
                # Cooldown expired
                self._exit_cooldown()

        # Check consecutive losses (without cooldown)
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._enter_cooldown(timestamp)
            return TradeValidation(
                allowed=False,
                reason=f"reached {self._consecutive_losses} consecutive losses, entering cooldown",
                position_size=0,
            )

        # Check trade frequency
        freq_ok, freq_reason = self._check_trade_frequency(timestamp)
        if not freq_ok:
            return TradeValidation(
                allowed=False,
                reason=freq_reason,
                position_size=0,
            )

        # Perform standard validation
        validation = self.validate_trade(
            signal=signal,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            current_equity=current_equity,
            timestamp=timestamp,
        )

        # Record trade time if allowed
        if validation.allowed:
            self._record_trade(timestamp)

        return validation

    def _update_drawdown(self, current_equity: float) -> None:
        """Update peak equity and current drawdown.

        Args:
            current_equity: Current account equity
        """
        # Update peak equity if we have new high
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

            # Check if we've recovered from drawdown trigger
            if self._max_drawdown_hit:
                recovery_threshold = self.config.max_drawdown_limit * self.config.drawdown_recovery_pct
                if self._current_drawdown <= recovery_threshold:
                    logger.info(
                        "max_drawdown_recovered",
                        current_equity=current_equity,
                        peak_equity=self._peak_equity,
                        current_drawdown=self._current_drawdown,
                    )
                    self._max_drawdown_hit = False

        # Calculate current drawdown
        if self._peak_equity > 0:
            self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            self._current_drawdown = 0.0

        # Check if max drawdown hit
        if self._current_drawdown >= self.config.max_drawdown_limit and not self._max_drawdown_hit:
            self._max_drawdown_hit = True
            logger.critical(
                "max_drawdown_limit_reached",
                current_equity=current_equity,
                peak_equity=self._peak_equity,
                drawdown_pct=self._current_drawdown,
                limit=self.config.max_drawdown_limit,
            )
            self.trigger_circuit_breaker(
                f"max drawdown {self._current_drawdown:.1%} >= {self.config.max_drawdown_limit:.1%}"
            )

    def record_trade_result(
        self,
        pnl: float,
        timestamp: datetime,
        current_equity: float,
    ) -> None:
        """Record trade result for consecutive loss tracking.

        Call this after each trade closes.

        Args:
            pnl: Trade P&L (positive = win, negative = loss)
            timestamp: Trade close timestamp
            current_equity: Current equity after trade
        """
        # Update daily P&L tracking
        self.update_daily_pnl(pnl, timestamp)

        # Update drawdown
        self._update_drawdown(current_equity)

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
            self._last_trade_was_loss = True

            logger.debug(
                "consecutive_loss_recorded",
                pnl=pnl,
                consecutive_losses=self._consecutive_losses,
                max_allowed=self.config.max_consecutive_losses,
            )

            # Check if we should enter cooldown
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._enter_cooldown(timestamp)
        else:
            # Win breaks the streak
            if self._consecutive_losses > 0:
                logger.debug(
                    "losing_streak_broken",
                    previous_consecutive_losses=self._consecutive_losses,
                    winning_pnl=pnl,
                )
            self._consecutive_losses = 0
            self._last_trade_was_loss = False

    def _enter_cooldown(self, timestamp: datetime) -> None:
        """Enter cooldown period after consecutive losses."""
        self._in_cooldown = True
        self._cooldown_end_time = timestamp + timedelta(
            seconds=self.config.cooldown_after_losses_seconds
        )

        logger.warning(
            "entering_cooldown",
            consecutive_losses=self._consecutive_losses,
            cooldown_seconds=self.config.cooldown_after_losses_seconds,
            cooldown_end=self._cooldown_end_time.isoformat(),
        )

    def _exit_cooldown(self) -> None:
        """Exit cooldown period and reset consecutive losses."""
        logger.info(
            "exiting_cooldown",
            previous_consecutive_losses=self._consecutive_losses,
        )
        self._in_cooldown = False
        self._cooldown_end_time = None
        self._consecutive_losses = 0  # Reset after cooldown

    def _check_trade_frequency(
        self, timestamp: datetime
    ) -> tuple[bool, str]:
        """Check if trade frequency limits allow another trade.

        Args:
            timestamp: Current timestamp

        Returns:
            Tuple of (allowed, reason)
        """
        # Check minimum time between trades
        if self._last_trade_time is not None:
            seconds_since_last = (timestamp - self._last_trade_time).total_seconds()
            if seconds_since_last < self.config.min_seconds_between_trades:
                return (
                    False,
                    f"minimum {self.config.min_seconds_between_trades}s between trades "
                    f"({seconds_since_last:.1f}s since last trade)",
                )

        # Check hourly trade limit
        one_hour_ago = timestamp - timedelta(hours=1)
        self._hourly_trades = [t for t in self._hourly_trades if t > one_hour_ago]

        if len(self._hourly_trades) >= self.config.max_trades_per_hour:
            return (
                False,
                f"hourly trade limit reached: {len(self._hourly_trades)} >= {self.config.max_trades_per_hour}",
            )

        return True, "within frequency limits"

    def _record_trade(self, timestamp: datetime) -> None:
        """Record a trade for frequency tracking.

        Args:
            timestamp: Trade timestamp
        """
        self._last_trade_time = timestamp
        self._hourly_trades.append(timestamp)

    def reset_consecutive_losses(self) -> None:
        """Manually reset consecutive loss counter."""
        logger.info(
            "consecutive_losses_manual_reset",
            previous_count=self._consecutive_losses,
        )
        self._consecutive_losses = 0
        self._in_cooldown = False
        self._cooldown_end_time = None

    def reset_drawdown_tracking(self, current_equity: float) -> None:
        """Reset drawdown tracking (use after significant account changes).

        Args:
            current_equity: New equity baseline
        """
        logger.info(
            "drawdown_tracking_reset",
            previous_peak=self._peak_equity,
            new_peak=current_equity,
            previous_drawdown=self._current_drawdown,
        )
        self._peak_equity = current_equity
        self._current_drawdown = 0.0
        self._max_drawdown_hit = False

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as fraction (0.1 = 10%)."""
        return self._current_drawdown

    @property
    def peak_equity(self) -> float:
        """Peak equity (high water mark)."""
        return self._peak_equity

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss count."""
        return self._consecutive_losses

    @property
    def is_in_cooldown(self) -> bool:
        """Whether currently in cooldown period."""
        return self._in_cooldown

    @property
    def is_max_drawdown_hit(self) -> bool:
        """Whether max drawdown limit has been hit."""
        return self._max_drawdown_hit

    def get_scalping_status(self) -> dict:
        """Get current scalping risk manager status.

        Returns:
            Dictionary with current state and metrics
        """
        base_status = self.get_status()
        base_status.update({
            "peak_equity": self._peak_equity,
            "current_drawdown": self._current_drawdown,
            "max_drawdown_limit": self.config.max_drawdown_limit,
            "max_drawdown_hit": self._max_drawdown_hit,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self.config.max_consecutive_losses,
            "in_cooldown": self._in_cooldown,
            "cooldown_end_time": self._cooldown_end_time.isoformat() if self._cooldown_end_time else None,
            "hourly_trades": len(self._hourly_trades),
            "max_trades_per_hour": self.config.max_trades_per_hour,
        })
        return base_status
