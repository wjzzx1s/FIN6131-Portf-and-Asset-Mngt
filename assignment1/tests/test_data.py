"""Check the real archived Yahoo snapshot, not fabricated market data."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_archive_to_verified_prices(tmp_path):
    """Nine complete ETF histories must survive the CSV/hash round trip."""
    path = ROOT / "data_io.py"
    assert path.exists(), "Data snapshot loader has not been implemented"
    spec = importlib.util.spec_from_file_location("data_io", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prices, metadata = module.build_snapshot(ROOT / "data/raw", tmp_path)
    assert list(prices.columns) == module.TICKERS
    assert prices.shape == (1256, 9)
    assert prices.index[0] == pd.Timestamp("2021-09-17")
    assert prices.index[-1] == pd.Timestamp("2026-09-18")
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices > 0).all().all()
    assert metadata["missing_cells"] == 0
    assert metadata["dropped_rows"] == 0
    pd.testing.assert_frame_equal(prices, module.load_snapshot(tmp_path), check_freq=False)
    with (tmp_path / "adjusted_close.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        module.load_snapshot(tmp_path)
