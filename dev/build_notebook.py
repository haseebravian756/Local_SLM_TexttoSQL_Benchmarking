"""Build kaggle/bird_minidev_text2sql.ipynb from dev/notebook_source.py.

Usage:  py dev/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "dev" / "notebook_source.py"
OUT = ROOT / "kaggle" / "bird_minidev_text2sql.ipynb"


def parse_cells(text):
    """Split a '# %%'-style source file into (cell_type, source) pairs. Lines before the first marker are ignored."""
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
            continue
        if kind == "markdown":
            line = line[2:] if line.startswith("# ") else line.lstrip("#")
        if kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    out = []
    for kind, lines in cells:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        out.append((kind, "\n".join(lines)))
    return out


def to_ipynb(cells):
    nb_cells = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    cells = parse_cells(SRC.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_ipynb(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(k == "code" for k, _ in cells)
    print(f"Wrote {OUT} ({len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown)")
