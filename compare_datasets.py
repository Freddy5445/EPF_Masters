"""
Compare a freshly built model CSV against one built by the old pipeline.

Cleaning moved into data_cleaning.ipynb, and a few things changed with it (see
DATA_CLEANING.md, "What changed in the move"). This says whether those changes
actually touched your data, so you can decide whether a backtest has to be
re-run rather than guessing.

    python compare_datasets.py datasets/DK1_clean_load-windsolar.csv old/DK1_clean.csv

Exit code 0 means the two files agree everywhere they overlap.
"""

import argparse
import sys

import pandas as pd


def load(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new")
    parser.add_argument("old")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)

    new, old = load(args.new), load(args.old)

    print(f"new: {len(new):,} hours, {new.index.min()} -> {new.index.max()}")
    print(f"old: {len(old):,} hours, {old.index.min()} -> {old.index.max()}")

    if list(new.columns) != list(old.columns):
        print(f"\ncolumn names differ:\n  new {list(new.columns)}\n  old {list(old.columns)}")
        print("Comparing by position, since read_data binds columns positionally.")
        old.columns = new.columns

    common = new.index.intersection(old.index)
    print(f"\noverlap: {len(common):,} hours")
    only_new = len(new.index.difference(old.index))
    only_old = len(old.index.difference(new.index))
    if only_new or only_old:
        print(f"  {only_new:,} hour(s) only in new, {only_old:,} only in old "
              f"-- the usable span moved, so the test period may differ")

    a = new.loc[common]
    b = old.loc[common]
    diff = (a - b).abs()
    changed = diff.gt(args.tolerance)

    total = int(changed.to_numpy().sum())
    print(f"\nvalues differing by more than {args.tolerance:g}: {total:,} "
          f"of {a.size:,} ({100 * total / max(a.size, 1):.4f}%)")

    if total:
        print("\nby column:")
        for column in a.columns:
            n = int(changed[column].sum())
            if n:
                worst = diff[column].max()
                print(f"  {column:<45} {n:>7,}  max abs diff {worst:.6g}")

        rows = changed.any(axis=1)
        print(f"\n{int(rows.sum()):,} hour(s) affected, "
              f"{rows[rows].index.normalize().nunique():,} distinct day(s)")
        print("first few:")
        for ts in rows[rows].index[:8]:
            print(f"  {ts}")

        print("\nA changed value inside a calibration window moves every LASSO fit that")
        print("sees it, so forecasts can differ on days whose own inputs are unchanged.")
        print("If any of these fall on or before your test period, re-run the backtest.")
        return 1

    print("\nIdentical. A re-run would reproduce the same forecasts; no need.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
