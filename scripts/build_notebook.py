#!/usr/bin/env python
"""Build notebooks/ml_notebook3.ipynb from notebooks/ml_notebook3.py.

The .py file is the source of record: it is what gets reviewed, diffed and run
locally. The .ipynb is a derived artefact for uploading to Kaggle. Cells are
split on the `# %%` markers, so the two stay in step by construction.

    python scripts/build_notebook.py [--check]

--check exits non-zero if the .ipynb on disk differs from what the .py implies,
which is what stops a stale notebook being uploaded after a source edit.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "ml_notebook3.py"
OUT = ROOT / "notebooks" / "ml_notebook3.ipynb"


def build(text):
    chunks, cur = [], []
    for line in text.split("\n"):
        if line.strip() == "# %%":
            if any(c.strip() for c in cur):
                chunks.append(cur)
            cur = []
        else:
            cur.append(line)
    if any(c.strip() for c in cur):
        chunks.append(cur)

    cells = []
    for ch in chunks:
        while ch and not ch[0].strip():
            ch.pop(0)
        while ch and not ch[-1].strip():
            ch.pop()
        src = "\n".join(ch)
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [],
                      "source": [l + "\n" for l in src.split("\n")[:-1]]
                                + [src.split("\n")[-1]]})
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    nb = build(SRC.read_text(encoding="utf-8"))
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != text:
            print(f"STALE: {OUT.name} does not match {SRC.name} — run "
                  "`python scripts/build_notebook.py`")
            return 1
        print(f"OK: {OUT.name} is in step with {SRC.name} ({len(nb['cells'])} cells)")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} — {len(nb['cells'])} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
