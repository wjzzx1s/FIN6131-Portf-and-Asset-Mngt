"""End-to-end acceptance: execute the deliverable without companion files."""
from pathlib import Path
import json
import subprocess
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_standalone_notebook_executes(tmp_path):
    builder = ROOT / "build_notebook.py"
    assert builder.exists(), "Notebook generator has not been implemented"
    submitted = ROOT / "portfolio_analysis.ipynb"
    original = submitted.read_bytes() if submitted.exists() else None
    target = tmp_path / "portfolio_analysis.ipynb"
    subprocess.run([sys.executable, str(builder), "--output", str(target)], check=True)
    assert target.exists(), "Generator must honor the isolated output path"
    assert (submitted.read_bytes() if submitted.exists() else None) == original
    notebook = nbformat.read(target, as_version=4)
    # Force the executing test interpreter: never launch a global Python kernel.
    manager = KernelManager(kernel_name="python3", ip="127.0.0.1")
    assert manager.kernel_spec is not None
    manager.kernel_spec.argv[0] = sys.executable
    NotebookClient(notebook, km=manager, timeout=240, resources={"metadata": {"path": str(tmp_path)}}).execute(cleanup_kc=True)
    assert not manager.has_kernel, "Notebook execution must shut down its kernel"
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    result = json.loads((tmp_path / "outputs/results.json").read_text())
    assert result["sample_sizes"] == {"full": 1255, "in_sample": 754, "out_of_sample": 501}
    assert len(result["full_sample"]) == 4
    assert len(result["out_of_sample"]) == 5
    assert result["checks"]["frontier_dominance"]
    assert result["checks"]["no_lookahead"]
    assert result["checks"]["independent_solver_agreement"]
    assert (tmp_path / "outputs/figures/frontiers.pdf").stat().st_size > 1000
    assert (tmp_path / "outputs/figures/oos_wealth.pdf").stat().st_size > 1000
