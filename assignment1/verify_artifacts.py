"""Fail the build if a deliverable is incomplete or inconsistent."""
from pathlib import Path
import hashlib
import json
import re
import sys
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
assert Path(sys.prefix).resolve() == ROOT.parent / ".venv", "Use the repository .venv"
metadata = json.loads((ROOT / "data/metadata.json").read_text())
assert hashlib.sha256((ROOT / "data/adjusted_close.csv").read_bytes()).hexdigest() == metadata["sha256"]
assert len(metadata["sources"]) == len(set(metadata["tickers"])) == 9
for source in metadata["sources"]:
    assert hashlib.sha256((ROOT / "data/raw" / source["file"]).read_bytes()).hexdigest() == source["sha256"]

notebook = nbformat.read(ROOT / "portfolio_analysis.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells), "Unexecuted cells"
assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
assert any((ROOT / "mpt.py").read_text().strip() == c.source.strip() for c in code_cells), "Stale embedded core"
result = json.loads((ROOT / "outputs/results.json").read_text())
assert result["snapshot_sha256"] == metadata["sha256"] and not result["refreshed"]
assert result["sample_sizes"] == {"full":1255, "in_sample":754, "out_of_sample":501}
for check in ["frontier_dominance", "independent_solver_agreement", "no_lookahead", "long_only_random_check"]:
    assert result["checks"][check]
for regime in ["full_weights", "in_sample_weights"]:
    for name, weights in result[regime].items():
        assert np.isclose(sum(weights), 1, atol=1e-8)
        if name.startswith("L") or name == "Equal":
            assert min(weights) >= -1e-8

# Independently recalculate holdout wealth and core metrics from exported returns.
daily = pd.read_csv(ROOT / "outputs/out_of_sample_daily_simple_returns.csv", index_col=0)
for name in daily:
    returns = daily[name].to_numpy()
    wealth = np.r_[1., np.cumprod(1+returns)]
    actual = result["out_of_sample"][name]
    assert np.isclose(actual["total_return"], wealth[-1]-1)
    assert np.isclose(actual["cagr"], wealth[-1]**(252/len(returns))-1)
    assert np.isclose(actual["max_drawdown"], (wealth/np.maximum.accumulate(wealth)-1).min())

pdf = PdfReader(ROOT / "report.pdf")
assert 2 <= len(pdf.pages) <= 3, f"Report has {len(pdf.pages)} pages, expected 2–3"
text = "\n".join(page.extract_text() for page in pdf.pages)
assert "??" not in text and len(text) > 5000, "Missing report content or unresolved references"
log = (ROOT / "report.log").read_text()
assert not re.search(r"^!|LaTeX Warning|Package \S+ Warning|Overfull \\[hv]box", log, re.M), "LaTeX errors, warnings, or overflow"
# Spot-check numerical tokens in the rendered artifact against notebook results.
for name, values in result["full_sample"].items():
    for field in ["return", "volatility"]:
        assert f"{100*values[field]:.2f}" in text, f"Missing/stale {name} {field}"
for name, values in result["out_of_sample"].items():
    assert f"{100*values['cagr']:.2f}" in text, f"Missing/stale {name} CAGR"

print(json.dumps({"report_pages":len(pdf.pages), "unresolved_references":text.count("??"),
                  "executed_code_cells":len(code_cells), "verified_etfs":len(metadata["tickers"]),
                  "sample_sizes":result["sample_sizes"],
                  "max_independent_weight_error":result["checks"]["max_independent_weight_error"],
                  "snapshot_sha256":metadata["sha256"]}, indent=2))
