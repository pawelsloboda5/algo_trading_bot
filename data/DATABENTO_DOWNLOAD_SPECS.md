# Databento Download Specifications for MCL

## Dataset Overview

| # | Schema | Time Range | Est. Size | Purpose |
|---|--------|------------|-----------|---------|
| 1 | OHLCV-1m | 3 years (2022-01-01 to 2025-01-09) | ~50-100 MB | Primary backtesting |
| 2 | OHLCV-1h | 3 years (2022-01-01 to 2025-01-09) | ~5-10 MB | Trend context |
| 3 | OHLCV-1d | 3 years (2022-01-01 to 2025-01-09) | ~500 KB | Regime detection |
| 4 | OHLCV-1s | 1 year (2025-01-09 to 2026-01-09) | ~5-10 GB | Precise timing |
| 5 | MBP-1 | 1 year (2025-01-09 to 2026-01-09) | ~22.8 GB | Spread/slippage |

---

## Exact Databento Web UI Settings

### Dataset 1: OHLCV-1m (3 Years) - PRIMARY
```
Dataset:        GLBX.MDP3
Symbols:        MCL.FUT
Schema:         OHLCV-1m ($0.01/GB)
Time Range:     2022-01-01 to 2025-01-09
Encoding:       Databento Binary Encoding (DBN)
Compression:    zstd
Delivery:       Direct download
Split:          By duration - Month
```
**Save to:** `data/databento_downloads/MCL_FUT_ohlcv-1m_3yr.dbn.zst`

---

### Dataset 2: OHLCV-1h (3 Years)
```
Dataset:        GLBX.MDP3
Symbols:        MCL.FUT
Schema:         OHLCV-1h ($0.01/GB)
Time Range:     2022-01-01 to 2025-01-09
Encoding:       Databento Binary Encoding (DBN)
Compression:    zstd
Delivery:       Direct download
Split:          None (single file is fine, small dataset)
```
**Save to:** `data/databento_downloads/MCL_FUT_ohlcv-1h_3yr.dbn.zst`

---

### Dataset 3: OHLCV-1d (3 Years)
```
Dataset:        GLBX.MDP3
Symbols:        MCL.FUT
Schema:         OHLCV-1d ($0.01/GB)
Time Range:     2022-01-01 to 2025-01-09
Encoding:       Databento Binary Encoding (DBN)
Compression:    zstd
Delivery:       Direct download
Split:          None (single file, tiny dataset)
```
**Save to:** `data/databento_downloads/MCL_FUT_ohlcv-1d_3yr.dbn.zst`

---

### Dataset 4: OHLCV-1s (1 Year) - LARGE
```
Dataset:        GLBX.MDP3
Symbols:        MCL.FUT
Schema:         OHLCV-1s ($0.01/GB)
Time Range:     2025-01-09 to 2026-01-09
Encoding:       Databento Binary Encoding (DBN)
Compression:    zstd
Delivery:       Direct download
Split:          By duration - Day (IMPORTANT: large dataset)
```
**Save to:** `data/databento_downloads/ohlcv-1s/` (will be many daily files)

---

### Dataset 5: MBP-1 (1 Year) - LARGEST
```
Dataset:        GLBX.MDP3
Symbols:        MCL.FUT
Schema:         MBP-1 ($1.80/GB)
Time Range:     2025-01-09 to 2026-01-09
Encoding:       Databento Binary Encoding (DBN)
Compression:    zstd
Delivery:       Direct download
Split:          By duration - Day (CRITICAL: 22.8 GB dataset)
```
**Save to:** `data/databento_downloads/mbp-1/` (will be many daily files)

---

## Folder Structure

```
data/
├── databento_downloads/        <- Put raw Databento files here
│   ├── MCL_FUT_ohlcv-1m_3yr.dbn.zst
│   ├── MCL_FUT_ohlcv-1h_3yr.dbn.zst
│   ├── MCL_FUT_ohlcv-1d_3yr.dbn.zst
│   ├── ohlcv-1s/               <- Daily split files
│   │   ├── 2025-01-09.dbn.zst
│   │   ├── 2025-01-10.dbn.zst
│   │   └── ...
│   └── mbp-1/                  <- Daily split files
│       ├── 2025-01-09.dbn.zst
│       ├── 2025-01-10.dbn.zst
│       └── ...
├── raw/                        <- Converted Parquet files (for backtesting)
│   └── MCL_FUT/
│       ├── ohlcv-1m/
│       ├── ohlcv-1h/
│       ├── ohlcv-1d/
│       ├── ohlcv-1s/
│       └── mbp-1/
├── processed/                  <- Feature-engineered data
└── cache/                      <- Temporary cache
```

---

## Download Priority Order

1. **OHLCV-1m** (3 years) - Essential for backtesting
2. **OHLCV-1h** (3 years) - Multi-timeframe analysis
3. **OHLCV-1d** (3 years) - Regime detection
4. **OHLCV-1s** (1 year) - Precise entry timing (optional for now)
5. **MBP-1** (1 year) - Live trading prep (optional for now)

---

## Cost Estimate

| Schema | Size | Price/GB | Est. Cost |
|--------|------|----------|-----------|
| OHLCV-1m | ~100 MB | $0.01 | ~$0.001 |
| OHLCV-1h | ~10 MB | $0.01 | ~$0.0001 |
| OHLCV-1d | ~500 KB | $0.01 | ~$0.00001 |
| OHLCV-1s | ~10 GB | $0.01 | ~$0.10 |
| MBP-1 | 22.8 GB | $1.80 | ~$41 |

**Total (all 5):** ~$41 (mostly MBP-1)
**OHLCV only:** ~$0.10

---

## After Download: Convert to Parquet

Once downloaded, convert DBN to Parquet for backtesting:

### Option 1: Use the Conversion Script (Recommended)

```bash
# Convert single file
python scripts/convert_databento.py -i data/databento_downloads/file.dbn.zst -s ohlcv-1m

# Convert entire directory (e.g., daily OHLCV-1s files)
python scripts/convert_databento.py -i data/databento_downloads/ohlcv-1s/ -s ohlcv-1s

# Convert MBP-1 directory
python scripts/convert_databento.py -i data/databento_downloads/mbp-1/ -s mbp-1
```

### Option 2: Manual Python

```python
import databento as db

# Load DBN file
data = db.DBNStore.from_file("data/databento_downloads/MCL_FUT_ohlcv-1m_3yr.dbn.zst")

# Convert to DataFrame
df = data.to_df()

# Save as Parquet (our backtest engine reads this)
df.to_parquet("data/raw/MCL_FUT/ohlcv-1m/2022-2025.parquet")
```

### Option 3: Download via API Script
```bash
python scripts/download_historical.py --symbol MCL.FUT --schema ohlcv-1m --start 2022-01-01
```

---

## AWS S3 Scaling (Future)

When you scale to AWS, this structure maps directly to S3:

```
s3://your-bucket/data/raw/MCL_FUT/ohlcv-1m/2022/01/01.parquet
s3://your-bucket/data/raw/MCL_FUT/ohlcv-1m/2022/01/02.parquet
...
```

The Parquet format is ideal for:
- Columnar storage (efficient for time-series queries)
- Compression (typically 5-10x)
- Partitioning (by date for fast range queries)
- AWS Athena/Glue integration
