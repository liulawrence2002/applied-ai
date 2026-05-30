"""Dump source of W2 - Explainability.ipynb cell-by-cell."""
import nbformat
from pathlib import Path

nb_path = Path(r"C:\Users\liula\Documents\applied ai\project 1\W2 - Explainability.ipynb")
nb = nbformat.read(str(nb_path), as_version=4)
print(f"cells: {len(nb.cells)}")
for i, c in enumerate(nb.cells):
    src = c.source if isinstance(c.source, str) else "".join(c.source)
    print(f"\n=== cell {i} ({c.cell_type}) [{len(src)} chars] ===")
    print(src)
