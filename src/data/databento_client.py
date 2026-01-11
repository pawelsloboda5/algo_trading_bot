"""Databento API client for market data."""

from datetime import datetime
from pathlib import Path

import databento as db
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class DatabentoClient:
    """Client for fetching market data from Databento.

    Handles historical data downloads and live streaming for CME futures.
    Uses the GLBX.MDP3 dataset for CME Globex market data.
    """

    def __init__(self, api_key: str, dataset: str = "GLBX.MDP3"):
        """Initialize Databento client.

        Args:
            api_key: Databento API key
            dataset: Databento dataset (default: GLBX.MDP3 for CME)
        """
        self.api_key = api_key
        self.dataset = dataset
        self._historical = db.Historical(api_key)
        logger.info("databento_client_initialized", dataset=dataset)

    def get_cost_estimate(
        self,
        symbols: list[str],
        schema: str,
        start: datetime,
        end: datetime,
    ) -> float:
        """Get cost estimate for a data request.

        Args:
            symbols: List of symbols (e.g., ["CL.FUT"])
            schema: Data schema (e.g., "ohlcv-1m", "trades", "mbp-1")
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            Estimated cost in USD
        """
        try:
            cost = self._historical.metadata.get_cost(
                dataset=self.dataset,
                symbols=symbols,
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            logger.info(
                "cost_estimate",
                symbols=symbols,
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
                cost_usd=cost,
            )
            return cost
        except Exception as e:
            logger.error("cost_estimate_failed", error=str(e))
            raise

    def get_historical_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "ohlcv-1m",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars.

        Args:
            symbols: List of symbols (e.g., ["CL.FUT"] for continuous front month)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            schema: Bar schema (default: "ohlcv-1m" for 1-minute bars)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(
            "fetching_historical_bars",
            symbols=symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            schema=schema,
        )

        try:
            data = self._historical.timeseries.get_range(
                dataset=self.dataset,
                symbols=symbols,
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
            )

            df = data.to_df()
            logger.info("historical_bars_fetched", rows=len(df))
            return df

        except Exception as e:
            logger.error("historical_bars_failed", error=str(e))
            raise

    def download_to_file(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        output_path: Path,
        schema: str = "ohlcv-1m",
    ) -> Path:
        """Download historical data directly to a file.

        More efficient for large downloads as it streams to disk.

        Args:
            symbols: List of symbols
            start: Start datetime (UTC)
            end: End datetime (UTC)
            output_path: Path for output file (.dbn.zst format)
            schema: Data schema

        Returns:
            Path to downloaded file
        """
        logger.info(
            "downloading_to_file",
            symbols=symbols,
            start=start.isoformat(),
            end=end.isoformat(),
            output_path=str(output_path),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._historical.timeseries.get_range(
                dataset=self.dataset,
                symbols=symbols,
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
                path=str(output_path),
            )
            logger.info("download_complete", output_path=str(output_path))
            return output_path

        except Exception as e:
            logger.error("download_failed", error=str(e))
            raise

    def get_available_symbols(self, symbol_pattern: str = "CL") -> list[dict]:
        """Get available symbols matching a pattern.

        Args:
            symbol_pattern: Pattern to match (e.g., "CL" for crude oil)

        Returns:
            List of symbol info dictionaries
        """
        try:
            # Get instrument definitions
            symbols = self._historical.metadata.list_symbols(
                dataset=self.dataset,
                symbol_pattern=f"{symbol_pattern}*",
            )
            logger.info("symbols_fetched", count=len(symbols), pattern=symbol_pattern)
            return symbols
        except Exception as e:
            logger.error("symbols_fetch_failed", error=str(e))
            raise

    @staticmethod
    def get_continuous_symbol(root: str = "CL") -> str:
        """Get continuous front-month futures symbol.

        Args:
            root: Root symbol (e.g., "CL" for crude oil)

        Returns:
            Continuous contract symbol for Databento
        """
        # Databento uses .FUT suffix for continuous front month
        return f"{root}.FUT"

    @staticmethod
    def get_specific_contract(root: str, year: int, month: int) -> str:
        """Get specific futures contract symbol.

        Args:
            root: Root symbol (e.g., "CL")
            year: Contract year (e.g., 2024)
            month: Contract month (1-12)

        Returns:
            Specific contract symbol
        """
        # CME futures month codes
        month_codes = {
            1: "F",  # January
            2: "G",  # February
            3: "H",  # March
            4: "J",  # April
            5: "K",  # May
            6: "M",  # June
            7: "N",  # July
            8: "Q",  # August
            9: "U",  # September
            10: "V",  # October
            11: "X",  # November
            12: "Z",  # December
        }
        month_code = month_codes[month]
        year_suffix = str(year)[-2:]  # Last 2 digits
        return f"{root}{month_code}{year_suffix}"
