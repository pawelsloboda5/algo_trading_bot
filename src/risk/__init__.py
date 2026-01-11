"""Risk management modules for MCL futures trading.

This module provides:
- Position sizing algorithms (fixed fractional, volatility-based, Kelly)
- Risk enforcement (daily limits, circuit breakers)
- Stop-loss strategies (fixed, trailing, ATR-based)

Usage:
    from src.risk import RiskManager, PositionSizer, StopLossManager

    # Create risk manager
    risk_manager = RiskManager(
        initial_capital=5000,
        risk_per_trade=0.02,
        daily_loss_limit=0.05,
    )

    # Validate trade before execution
    validation = risk_manager.validate_trade(
        signal=Signal.LONG,
        entry_price=75.50,
        stop_loss_price=75.30,
        current_equity=5000,
        timestamp=datetime.now(),
    )

    if validation.allowed:
        # Execute with validation.position_size contracts
        pass
"""

from src.risk.position_sizer import PositionSizeResult, PositionSizer
from src.risk.risk_manager import RiskManager, TradeValidation
from src.risk.stop_loss import StopLossManager, StopLossResult, StopLossType

__all__ = [
    # Position sizing
    "PositionSizer",
    "PositionSizeResult",
    # Risk management
    "RiskManager",
    "TradeValidation",
    # Stop-loss
    "StopLossManager",
    "StopLossResult",
    "StopLossType",
]
