"""Load the cleaning functions out of ``data_cleaning.ipynb``.

The cleaning logic deliberately lives in the notebook rather than in a module:
every decision about the data should be visible in one place, next to the
analysis that motivated it. That would normally put it beyond the reach of tests,
which is how careful data handling quietly rots.

This executes the notebook's function-definition cell -- the one that defines
``build_model_panel`` -- and hands back its namespace. Nothing in the notebook
depends on this file; it only reads.
"""

import json
import os

import numpy as np
import pandas as pd

from entsoe_tp.areas import lookup

NOTEBOOK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cleaning.ipynb")

# The cell is identified by what it defines, not by its position, so inserting a
# cell above it does not silently start testing something else.
MARKER = "def build_model_panel("


def load(notebook=NOTEBOOK):
    """Return the namespace of the notebook's cleaning-function cell."""
    with open(notebook, encoding="utf-8") as handle:
        cells = json.load(handle)["cells"]

    sources = [
        "".join(c["source"]) for c in cells
        if c["cell_type"] == "code" and MARKER in "".join(c["source"])
    ]
    if len(sources) != 1:
        raise AssertionError(
            f"expected exactly one cell defining {MARKER!r} in {notebook}, "
            f"found {len(sources)}")

    namespace = {"pd": pd, "np": np, "json": json, "lookup": lookup,
                 "display": lambda *a, **k: None}
    exec(compile(sources[0], notebook, "exec"), namespace)
    return namespace
