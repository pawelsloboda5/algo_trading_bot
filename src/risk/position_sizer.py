"""Position sizing algorithms for risk-based trading."""

import math
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    contracts: int
    risk_amount: float
    method: str
    reason: str | None = None


class PositionSizer:
    """Position sizing calculator for MCL futures trading.

    Provides multiple sizing methods:
    - Fixed fractional: Risk X% of equity per trade
    - Fixed contracts: Always trade N contracts
    - Volatility adjusted: Size based on ATR
    - Kelly criterion: Optimal f based on win rate
    """

    def __init__(
        self,
        contract_multiplier: float = 100.0,
        tick_size: float = 0.01,
        max_contracts: int = 5,
    ):
        """Initialize position sizer.

        Args:
            contract_multiplier: Contract multiplier (MCL = 100 barrels)
            tick_size: Minimum price increment (MCL = $0.01)
            max_contracts: Maximum contracts per position
        """
        if contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if max_contracts <= 0:
            raise ValueError("max_contracts must be positive")

        self.contract_multiplier = contract_multiplier
        self.tick_size = tick_size
        self.max_contracts = max_contracts

    def fixed_fractional(
        self,
        equity: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> PositionSizeResult:
        """Calculate position size based on fixed fractional risk.

        Risk X% of equity per trade, with size determined by stop distance.

        Formula:
        - stop_distance = |entry_price - stop_loss_price|
        - dollar_risk_per_contract = stop_distance * multiplier
        - risk_amount = equity * risk_per_trade
        - contracts = floor(risk_amount / dollar_risk_per_contract)

        Example (MCL with $5,000 equity, 2% risk):
        - Entry: $75.50, Stop: $75.30
        - Stop distance: $0.20 * 100 = $20/contract
        - Risk amount: $5,000 * 0.02 = $100
        - Contracts: floor(100 / 20) = 5

        Args:
            equity: Current account equity
            risk_per_trade: Risk per trade as fraction (e.g., 0.02 for 2%)
            entry_price: Planned entry price
            stop_loss_price: Stop-loss price

        Returns:
            PositionSizeResult with calculated contracts
        """
        if equity <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="fixed_fractional",
                reason="equity must be positive",
            )

        if risk_per_trade <= 0 or risk_per_trade > 1:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="fixed_fractional",
                reason="risk_per_trade must be between 0 and 1",
            )

        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance < self.tick_size:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="fixed_fractional",
                reason="stop_distance too small (less than tick size)",
            )

        dollar_risk_per_contract = stop_distance * self.contract_multiplier
        risk_amount = equity * risk_per_trade
        raw_contracts = risk_amount / dollar_risk_per_contract
        contracts = math.floor(raw_contracts)

        reason = None
        if contracts > self.max_contracts:
            reason = f"capped from {contracts} to {self.max_contracts}"
            contracts = self.max_contracts
        elif contracts == 0:
            reason = "stop distance too large for risk amount"

        logger.debug(
            "fixed_fractional_sizing",
            equity=equity,
            risk_per_trade=risk_per_trade,
            stop_distance=stop_distance,
            risk_amount=risk_amount,
            raw_contracts=raw_contracts,
            final_contracts=contracts,
        )

        return PositionSizeResult(
            contracts=contracts,
            risk_amount=risk_amount,
            method="fixed_fractional",
            reason=reason,
        )

    def fixed_contracts(self, num_contracts: int) -> PositionSizeResult:
        """Return fixed number of contracts.

        Simple sizing method that always trades the same quantity.

        Args:
            num_contracts: Number of contracts to trade

        Returns:
            PositionSizeResult with fixed contracts
        """
        if num_contracts <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="fixed_contracts",
                reason="num_contracts must be positive",
            )

        contracts = min(num_contracts, self.max_contracts)
        reason = None
        if contracts < num_contracts:
            reason = f"capped from {num_contracts} to {self.max_contracts}"

        return PositionSizeResult(
            contracts=contracts,
            risk_amount=0.0,  # Not calculated for fixed contracts
            method="fixed_contracts",
            reason=reason,
        )

    def volatility_adjusted(
        self,
        equity: float,
        current_atr: float,
        risk_target: float = 0.02,
    ) -> PositionSizeResult:
        """Calculate position size based on volatility (ATR).

        Size positions to target a specific volatility exposure.

        Formula:
        - dollar_volatility = ATR * multiplier
        - risk_amount = equity * risk_target
        - contracts = floor(risk_amount / dollar_volatility)

        Args:
            equity: Current account equity
            current_atr: Current Average True Range value
            risk_target: Target risk as fraction of equity (default 2%)

        Returns:
            PositionSizeResult with calculated contracts
        """
        if equity <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="volatility_adjusted",
                reason="equity must be positive",
            )

        if current_atr <= 0:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="volatility_adjusted",
                reason="ATR must be positive",
            )

        if risk_target <= 0 or risk_target > 1:
            return PositionSizeResult(
                contracts=0,
                risk_amount=0.0,
                method="volatility_adjusted",
                reason="risk_target must be between 0 and 1",
            )

        dollar_volatility = current_atr * self.contract_multiplier
        risk_amount = equity * risk_target
        raw_contracts = risk_amount / dollar_volatility
        contracts = math.floor(raw_contracts)

        reason = None
        if contracts > self.max_contracts:
            reason = f"capped from {contracts} to {self.max_contracts}"
            contracts = self.max_contracts
        elif contracts == 0:
            reason = "volatility too high for risk target"

        logger.debug(
            "volatility_adjusted_sizing",
            equity=equity,
            current_atr=current_atr,
            risk_target=risk_target,
            dollar_volatility=dollar_volatility,
            risk_amount=risk_amount,
            final_contracts=contracts,
        )

        return PositionSizeResult(
            contracts=contracts,
            risk_amount=risk_amount,
            method="volatility_adjusted",
            reason=reason,
        )

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
    ) -> float:
        """Calculate optimal position size using Kelly criterion.

        Kelly formula: f* = (p * b - q) / b
        where:
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        - b = ratio of avg win to avg loss

        Note: Returns a FRACTION of capital to risk, not contracts.
        Use with fixed_fractional() to convert to contract count.

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            fraction: Kelly fraction (0.5 = half-Kelly for safety)

        Returns:
            Optimal fraction of capital to risk per trade
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(
                "kelly_invalid_win_rate",
                win_rate=win_rate,
                message="win_rate must be between 0 and 1",
            )
            return 0.0

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning(
                "kelly_invalid_amounts",
                avg_win=avg_win,
                avg_loss=avg_loss,
                message="avg_win and avg_loss must be positive",
            )
            return 0.0

        if fraction <= 0 or fraction > 1:
            logger.warning(
                "kelly_invalid_fraction",
                fraction=fraction,
                message="fraction must be between 0 and 1",
            )
            return 0.0

        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        kelly_f = (p * b - q) / b

        if kelly_f <= 0:
            logger.info(
                "kelly_negative",
                kelly_f=kelly_f,
                message="negative Kelly suggests no edge",
            )
            return 0.0

        optimal_f = kelly_f * fraction

        # Cap at reasonable maximum (never risk more than 25%)
        optimal_f = min(optimal_f, 0.25)

        logger.debug(
            "kelly_criterion",
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            full_kelly=kelly_f,
            fraction=fraction,
            optimal_f=optimal_f,
        )

        return optimal_f
