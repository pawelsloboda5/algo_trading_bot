"""Download historical market data from Databento."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.data.databento_client import DatabentoClient
from src.data.storage import DataStorage

logger = get_logger(__name__)


@click.command()
@click.option(
    "--symbol",
    default="CL.FUT",
    help="Symbol to download (default: CL.FUT for continuous crude oil)",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Start date (YYYY-MM-DD). Default: 30 days ago",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="End date (YYYY-MM-DD). Default: yesterday",
)
@click.option(
    "--schema",
    default="ohlcv-1m",
    type=click.Choice(["ohlcv-1s", "ohlcv-1m", "ohlcv-1h", "ohlcv-1d", "trades"]),
    help="Data schema (default: ohlcv-1m)",
)
@click.option(
    "--estimate-only",
    is_flag=True,
    help="Only show cost estimate, don't download",
)
def main(
    symbol: str,
    start: datetime | None,
    end: datetime | None,
    schema: str,
    estimate_only: bool,
) -> None:
    """Download historical market data from Databento.

    Examples:
        # Download last 30 days of 1-minute bars for CL
        python scripts/download_historical.py

        # Download specific date range
        python scripts/download_historical.py --start 2024-01-01 --end 2024-06-30

        # Get cost estimate only
        python scripts/download_historical.py --start 2024-01-01 --end 2024-12-31 --estimate-only

        # Download 1-second bars
        python scripts/download_historical.py --schema ohlcv-1s --start 2024-12-01 --end 2024-12-31
    """
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)

    # Set default dates
    if end is None:
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    if start is None:
        start = end - timedelta(days=30)

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Schema: {schema}")
    click.echo(f"Date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    click.echo()

    # Initialize client
    client = DatabentoClient(
        api_key=settings.databento_api_key,
        dataset=settings.databento_dataset,
    )

    # Get cost estimate
    try:
        cost = client.get_cost_estimate(
            symbols=[symbol],
            schema=schema,
            start=start,
            end=end,
        )
        click.echo(f"Estimated cost: ${cost:.2f}")

        if estimate_only:
            return

        # Confirm download
        if cost > 0:
            if not click.confirm("Proceed with download?"):
                click.echo("Download cancelled.")
                return

    except Exception as e:
        logger.error("cost_estimate_failed", error=str(e))
        click.echo(f"Error getting cost estimate: {e}", err=True)
        if not click.confirm("Continue without cost estimate?"):
            return

    # Download data
    click.echo()
    click.echo("Downloading data...")

    try:
        df = client.get_historical_bars(
            symbols=[symbol],
            start=start,
            end=end,
            schema=schema,
        )

        if df.empty:
            click.echo("No data returned.", err=True)
            return

        click.echo(f"Downloaded {len(df):,} rows")

        # Save to storage
        storage = DataStorage(settings.raw_data_dir)
        saved_paths = storage.save_bulk(df, symbol, schema)

        click.echo(f"Saved to {len(saved_paths)} files")
        click.echo(f"Storage location: {settings.raw_data_dir}")

        # Show summary
        click.echo()
        click.echo("Data summary:")
        click.echo(f"  First timestamp: {df['ts_event'].min()}")
        click.echo(f"  Last timestamp: {df['ts_event'].max()}")
        click.echo(f"  Total rows: {len(df):,}")

        if "close" in df.columns:
            click.echo(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        if "volume" in df.columns:
            click.echo(f"  Total volume: {df['volume'].sum():,}")

    except Exception as e:
        logger.error("download_failed", error=str(e))
        click.echo(f"Download failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
