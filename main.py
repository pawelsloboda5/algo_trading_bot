"""Algo Trading Bot - Main entry point."""

import sys
from pathlib import Path

import click

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Algo Trading Bot - Crude Oil Futures Trading System."""
    pass


@cli.command()
def info() -> None:
    """Show configuration and system info."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger = get_logger(__name__)

    click.echo("Algo Trading Bot Configuration")
    click.echo("=" * 40)
    click.echo(f"Environment: {settings.environment}")
    click.echo(f"Symbol: {settings.symbol}")
    click.echo(f"Exchange: {settings.exchange}")
    click.echo()
    click.echo("Data Settings:")
    click.echo(f"  Databento Dataset: {settings.databento_dataset}")
    click.echo(f"  Data Directory: {settings.data_dir}")
    click.echo()
    click.echo("IB Connection:")
    click.echo(f"  Host: {settings.ib_host}")
    click.echo(f"  Port: {settings.ib_port}")
    click.echo(f"  Paper Trading: {settings.is_paper_trading}")
    click.echo()
    click.echo("Risk Parameters:")
    click.echo(f"  Initial Capital: ${settings.initial_capital:,.0f}")
    click.echo(f"  Risk Per Trade: {settings.risk_per_trade:.1%}")
    click.echo(f"  Daily Loss Limit: {settings.daily_loss_limit:.1%}")
    click.echo(f"  Max Contracts: {settings.max_position_contracts}")
    click.echo()
    click.echo("Strategy Parameters:")
    click.echo(f"  Fast EMA: {settings.fast_period}")
    click.echo(f"  Slow EMA: {settings.slow_period}")
    click.echo(f"  ATR Period: {settings.atr_period}")

    logger.info("config_displayed", environment=settings.environment)


@cli.command()
@click.option("--symbol", default="CL.FUT", help="Symbol to download")
@click.option("--days", default=30, help="Number of days to download")
def download(symbol: str, days: int) -> None:
    """Download historical market data."""
    from datetime import datetime, timedelta

    from src.data.databento_client import DatabentoClient
    from src.data.storage import DataStorage

    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)

    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    start = end - timedelta(days=days)

    click.echo(f"Downloading {symbol} from {start.date()} to {end.date()}")

    client = DatabentoClient(settings.databento_api_key, settings.databento_dataset)

    try:
        df = client.get_historical_bars([symbol], start, end, schema="ohlcv-1m")
        click.echo(f"Downloaded {len(df):,} rows")

        storage = DataStorage(settings.raw_data_dir)
        storage.save_bulk(df, symbol, "ohlcv-1m")
        click.echo("Data saved successfully")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
