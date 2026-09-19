"""Execute with the calling repository interpreter, then save checked outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

ROOT = Path(__file__).resolve().parent
path = ROOT / "portfolio_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3", ip="127.0.0.1")
assert manager.kernel_spec is not None
manager.kernel_spec.argv[0] = sys.executable
NotebookClient(notebook, km=manager, timeout=240,
               resources={"metadata": {"path": str(ROOT)}}).execute(cleanup_kc=True)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
for cell in notebook.cells:
    # Execution timestamps are not analytical output and create noisy Git diffs.
    cell.metadata.pop("execution", None)
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Executed {len(code_cells)} code cells without errors using {sys.executable}")
