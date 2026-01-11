"""Data storage utilities for Parquet files."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.logging_config import get_logger

logger = get_logger(__name__)


class DataStorage:
    """Storage manager for market data using Parquet format.

    Organizes data by symbol and date for efficient querying.
    Uses Apache Parquet for columnar storage with compression.
    """

    def __init__(self, base_dir: Path):
        """Initialize storage manager.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("storage_initialized", base_dir=str(self.base_dir))

    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get directory path for a symbol."""
        # Sanitize symbol for filesystem
        safe_symbol = symbol.replace("/", "_").replace(".", "_")
        return self.base_dir / safe_symbol

    def _get_file_path(self, symbol: str, date: datetime, schema: str) -> Path:
        """Get file path for a specific date's data.

        Args:
            symbol: Trading symbol
            date: Date of data
            schema: Data schema (e.g., "ohlcv-1m", "trades")

        Returns:
            Path to Parquet file
        """
        symbol_dir = self._get_symbol_dir(symbol)
        year_dir = symbol_dir / str(date.year)
        filename = f"{date.strftime('%Y-%m-%d')}_{schema}.parquet"
        return year_dir / filename

    def save_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        date: datetime,
        schema: str = "ohlcv-1m",
    ) -> Path:
        """Save DataFrame to Parquet file.

        Args:
            df: DataFrame to save
            symbol: Trading symbol
            date: Date of data
            schema: Data schema

        Returns:
            Path to saved file
        """
        file_path = self._get_file_path(symbol, date, schema)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to PyArrow table for efficient storage
        table = pa.Table.from_pandas(df)

        # Write with compression
        pq.write_table(
            table,
            file_path,
            compression="snappy",
            use_dictionary=True,
        )

        logger.info(
            "data_saved",
            symbol=symbol,
            date=date.strftime("%Y-%m-%d"),
            rows=len(df),
            path=str(file_path),
        )
        return file_path

    def load_dataframe(
        self,
        symbol: str,
        date: datetime,
        schema: str = "ohlcv-1m",
    ) -> pd.DataFrame | None:
        """Load DataFrame from Parquet file.

        Args:
            symbol: Trading symbol
            date: Date of data
            schema: Data schema

        Returns:
            DataFrame or None if file doesn't exist
        """
        file_path = self._get_file_path(symbol, date, schema)

        if not file_path.exists():
            logger.warning("file_not_found", path=str(file_path))
            return None

        df = pd.read_parquet(file_path)
        logger.info("data_loaded", symbol=symbol, date=date.strftime("%Y-%m-%d"), rows=len(df))
        return df

    def load_date_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        schema: str = "ohlcv-1m",
    ) -> pd.DataFrame:
        """Load data for a date range.

        Args:
            symbol: Trading symbol
            start: Start date (inclusive)
            end: End date (inclusive)
            schema: Data schema

        Returns:
            Combined DataFrame for the date range
        """
        dfs = []
        current = start

        while current <= end:
            df = self.load_dataframe(symbol, current, schema)
            if df is not None:
                dfs.append(df)
            current = current.replace(day=current.day + 1) if current.day < 28 else datetime(
                current.year + (1 if current.month == 12 else 0),
                (current.month % 12) + 1,
                1,
            )

        if not dfs:
            logger.warning("no_data_in_range", symbol=symbol, start=str(start), end=str(end))
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("ts_event").reset_index(drop=True)

        logger.info(
            "date_range_loaded",
            symbol=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            rows=len(combined),
        )
        return combined

    def save_bulk(
        self,
        df: pd.DataFrame,
        symbol: str,
        schema: str = "ohlcv-1m",
        timestamp_col: str = "ts_event",
    ) -> list[Path]:
        """Save bulk data, splitting by date.

        Args:
            df: DataFrame with data spanning multiple dates
            symbol: Trading symbol
            schema: Data schema
            timestamp_col: Column name containing timestamps

        Returns:
            List of paths to saved files
        """
        if df.empty:
            logger.warning("empty_dataframe", symbol=symbol)
            return []

        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract date for grouping
        df["_date"] = df[timestamp_col].dt.date

        saved_paths = []
        for date, group in df.groupby("_date"):
            group = group.drop(columns=["_date"])
            date_dt = datetime.combine(date, datetime.min.time())
            path = self.save_dataframe(group, symbol, date_dt, schema)
            saved_paths.append(path)

        logger.info("bulk_save_complete", symbol=symbol, files_saved=len(saved_paths))
        return saved_paths

    def list_available_dates(self, symbol: str, schema: str = "ohlcv-1m") -> list[datetime]:
        """List available dates for a symbol.

        Args:
            symbol: Trading symbol
            schema: Data schema

        Returns:
            List of dates with available data
        """
        symbol_dir = self._get_symbol_dir(symbol)

        if not symbol_dir.exists():
            return []

        dates = []
        for file_path in symbol_dir.rglob(f"*_{schema}.parquet"):
            try:
                date_str = file_path.stem.split("_")[0]
                date = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date)
            except (ValueError, IndexError):
                continue

        return sorted(dates)

    def get_storage_info(self, symbol: str | None = None) -> dict:
        """Get storage statistics.

        Args:
            symbol: Optional symbol to filter (None for all)

        Returns:
            Dictionary with storage statistics
        """
        if symbol:
            dirs = [self._get_symbol_dir(symbol)]
        else:
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        total_files = 0
        total_size = 0

        for symbol_dir in dirs:
            for file_path in symbol_dir.rglob("*.parquet"):
                total_files += 1
                total_size += file_path.stat().st_size

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_dir": str(self.base_dir),
        }
