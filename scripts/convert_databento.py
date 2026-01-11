"""Convert Databento DBN files to Parquet for backtesting."""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from src.data.storage import DataStorage

logger = get_logger(__name__)


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input DBN file or directory containing DBN files",
)
@click.option(
    "--schema",
    "-s",
    type=click.Choice(["ohlcv-1s", "ohlcv-1m", "ohlcv-1h", "ohlcv-1d", "mbp-1"]),
    required=True,
    help="Data schema type",
)
@click.option(
    "--symbol",
    default="MCL_FUT",
    help="Symbol name for storage (default: MCL_FUT)",
)
def main(input_path: str, schema: str, symbol: str) -> None:
    """Convert Databento DBN files to Parquet format for backtesting.

    Examples:
        # Convert single file
        python scripts/convert_databento.py -i data/databento_downloads/file.dbn.zst -s ohlcv-1m

        # Convert directory of files
        python scripts/convert_databento.py -i data/databento_downloads/ohlcv-1s/ -s ohlcv-1s

        # Convert with custom symbol
        python scripts/convert_databento.py -i file.dbn.zst -s ohlcv-1m --symbol CL_FUT
    """
    try:
        import databento as db
    except ImportError:
        click.echo("Error: databento package not installed.", err=True)
        click.echo("Install with: pip install databento", err=True)
        sys.exit(1)

    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_dir=settings.log_dir)

    input_path = Path(input_path)
    storage = DataStorage(settings.raw_data_dir)

    click.echo("=" * 60)
    click.echo("DATABENTO DBN TO PARQUET CONVERTER")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"Input: {input_path}")
    click.echo(f"Schema: {schema}")
    click.echo(f"Symbol: {symbol}")
    click.echo(f"Output: {settings.raw_data_dir}")
    click.echo()

    # Find all DBN files (search recursively)
    if input_path.is_file():
        dbn_files = [input_path]
    else:
        # Search recursively for DBN files
        dbn_files = list(input_path.rglob("*.dbn.zst")) + list(input_path.rglob("*.dbn"))
        dbn_files = sorted(dbn_files)

    if not dbn_files:
        click.echo("No DBN files found.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(dbn_files)} DBN file(s)")
    click.echo()

    total_rows = 0
    total_files_saved = 0

    with click.progressbar(dbn_files, label="Converting") as files:
        for dbn_file in files:
            try:
                # Load DBN file
                data = db.DBNStore.from_file(str(dbn_file))
                df = data.to_df()

                if df.empty:
                    logger.warning("empty_file", file=str(dbn_file))
                    continue

                # Databento puts ts_event in the index, reset it to a column
                if df.index.name == "ts_event":
                    df = df.reset_index()

                # Databento uses ts_event as the primary timestamp
                if "ts_event" not in df.columns:
                    # Try to find timestamp column
                    ts_cols = [c for c in df.columns if "ts" in c.lower()]
                    if ts_cols:
                        df = df.rename(columns={ts_cols[0]: "ts_event"})
                    else:
                        click.echo(f"No timestamp column in {dbn_file.name}", err=True)
                        continue

                # Save using bulk method (splits by date automatically)
                saved_paths = storage.save_bulk(df, symbol, schema, timestamp_col="ts_event")

                total_rows += len(df)
                total_files_saved += len(saved_paths)

            except Exception as e:
                logger.error("conversion_failed", file=str(dbn_file), error=str(e))
                click.echo(f"\nError converting {dbn_file.name}: {e}", err=True)

    click.echo()
    click.echo("=" * 60)
    click.echo("CONVERSION COMPLETE")
    click.echo("=" * 60)
    click.echo(f"Total rows converted: {total_rows:,}")
    click.echo(f"Parquet files created: {total_files_saved}")
    click.echo()

    # Show storage info
    info = storage.get_storage_info(symbol)
    click.echo(f"Storage location: {info['base_dir']}")
    click.echo(f"Total files: {info['total_files']}")
    click.echo(f"Total size: {info['total_size_mb']} MB")
    click.echo()

    # Show available dates
    dates = storage.list_available_dates(symbol, schema)
    if dates:
        click.echo(f"Available date range: {dates[0].date()} to {dates[-1].date()}")
        click.echo(f"Trading days: {len(dates)}")
    click.echo()
    click.echo("You can now run backtests with:")
    click.echo(f"  python scripts/run_backtest.py --symbol {symbol}")


if __name__ == "__main__":
    main()
