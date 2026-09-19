"""Auditable Yahoo adjusted-price snapshot; no imputation or silent fallback.

The initial download used Yahoo's public chart endpoint in a real browser
because direct Python requests were region-blocked. Raw responses are archived.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "VNQ"]
START = "2021-09-19"
SPLIT = "2024-09-19"
END = "2026-09-19"  # Exclusive; includes the last completed US session.
PRICE_START = "2021-09-17"  # Prior close for the first in-window return.


def build_snapshot(raw_dir, output_dir):
    """Parse nine archived responses and preserve raw-file and CSV checksums."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    if {row["ticker"] for row in manifest} != set(TICKERS) or len(manifest) != len(TICKERS):
        raise ValueError("Expected exactly nine distinct ETF sources")
    columns, sources = {}, []
    for ticker in TICKERS:
        content = (raw_dir / f"{ticker}.json").read_bytes()
        payload = json.loads(content)["chart"]
        if payload["error"] is not None or len(payload["result"]) != 1:
            raise ValueError(f"Invalid Yahoo response for {ticker}")
        result = payload["result"][0]
        if result["meta"]["symbol"] != ticker or result["meta"]["currency"] != "USD":
            raise ValueError(f"Unexpected identity/currency for {ticker}")
        # Convert exchange timestamps to calendar dates before aligning ETFs.
        dates = pd.to_datetime(result["timestamp"], unit="s", utc=True)
        dates = dates.tz_convert("America/New_York").tz_localize(None).normalize()
        series = pd.Series(result["indicators"]["adjclose"][0]["adjclose"], index=dates, name=ticker)
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered dates for {ticker}")
        columns[ticker] = series
        source = next(row.copy() for row in manifest if row["ticker"] == ticker)
        source["sha256"] = hashlib.sha256(content).hexdigest()
        sources.append(source)
    prices = pd.DataFrame(columns).sort_index()
    missing = int(prices.isna().sum().sum())
    if missing or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Incomplete/nonpositive snapshot: investigate; do not impute")
    prices.index.name = "Date"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "adjusted_close.csv"
    prices.to_csv(csv_path, float_format="%.17g")
    metadata = {
        "provider": "Yahoo Finance", "field": "indicators.adjclose[0].adjclose",
        "retrieval_method": "Public chart endpoint via real browser; direct yfinance was blocked",
        "tickers": TICKERS, "currency": "USD", "window_start_inclusive": START,
        "window_end_exclusive": END, "split_date": SPLIT,
        "price_start": prices.index[0].date().isoformat(),
        "price_end": prices.index[-1].date().isoformat(),
        "price_rows": len(prices), "missing_cells": missing, "dropped_rows": 0,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "sources": sources,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return prices, metadata


def load_snapshot(data_dir):
    """Verify the saved bytes before loading; vendor revisions cannot slip in."""
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    csv_path = data_dir / "adjusted_close.csv"
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    prices = pd.read_csv(csv_path, index_col="Date", parse_dates=True, float_precision="round_trip")
    prices.index = pd.DatetimeIndex(prices.index).as_unit("s")
    if list(prices.columns) != TICKERS:
        raise ValueError("Unexpected ETF columns")
    return prices


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prices, metadata = build_snapshot(root / "data/raw", root / "data")
    print(f"Verified {prices.shape[0]} dates x {prices.shape[1]} ETFs; {metadata['missing_cells']} missing cells")
    print(f"CSV SHA-256: {metadata['sha256']}")
