"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Databento
    databento_api_key: str = Field(..., description="Databento API key")
    databento_dataset: str = Field(
        default="GLBX.MDP3", description="Databento dataset for CME futures"
    )

    # Interactive Brokers
    ib_host: str = Field(default="127.0.0.1", description="IB TWS/Gateway host")
    ib_port: int = Field(default=7497, description="IB port (7497=TWS paper, 4002=Gateway paper)")
    ib_client_id: int = Field(default=1, description="IB client ID")
    ib_account: str | None = Field(default=None, description="IB account ID")

    # Trading Symbol (MCL = Micro WTI Crude Oil)
    symbol: str = Field(default="MCL", description="Trading symbol (MCL = Micro Crude Oil)")
    exchange: str = Field(default="NYMEX", description="Exchange for futures")
    contract_multiplier: float = Field(
        default=100.0, description="Contract multiplier (MCL = 100 barrels, CL = 1000)"
    )
    tick_size: float = Field(default=0.01, description="Minimum price increment")

    # Databento Schema for market data
    default_schema: Literal["ohlcv-1m", "ohlcv-1s", "mbp-1", "mbp-10", "trades"] = Field(
        default="ohlcv-1m", description="Default Databento schema"
    )

    # Risk Management
    max_position_contracts: int = Field(default=5, description="Maximum contracts per position")
    risk_per_trade: float = Field(
        default=0.02, ge=0.001, le=0.10, description="Risk per trade as fraction of equity"
    )
    daily_loss_limit: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Daily loss limit as fraction of equity"
    )
    initial_capital: float = Field(default=5_000.0, description="Starting capital in USD")

    # Strategy Parameters
    fast_period: int = Field(default=20, ge=5, le=50, description="Fast EMA period")
    slow_period: int = Field(default=50, ge=20, le=200, description="Slow EMA period")
    atr_period: int = Field(default=14, ge=5, le=30, description="ATR period for volatility")

    # Data Storage
    data_dir: Path = Field(default=Path("data"), description="Data storage directory")
    log_dir: Path = Field(default=Path("logs"), description="Log storage directory")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Environment
    environment: Literal["development", "paper", "production"] = Field(
        default="development", description="Runtime environment"
    )

    @field_validator("slow_period")
    @classmethod
    def slow_must_be_greater_than_fast(cls, v: int, info) -> int:
        """Ensure slow period is greater than fast period."""
        fast = info.data.get("fast_period", 20)
        if v <= fast:
            raise ValueError(f"slow_period ({v}) must be greater than fast_period ({fast})")
        return v

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.ib_port in (7497, 4002)

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw data directory."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed data directory."""
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        """Path to cache directory."""
        return self.data_dir / "cache"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
