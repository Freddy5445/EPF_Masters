"""
Download the full Nord Pool (NP) day-ahead electricity market dataset using the
epftoolbox ``read_data`` function and save it as a single CSV.

``read_data`` returns a train/test split, so the full dataset is reconstructed by
concatenating the two parts back together in chronological order.
"""

import os
import sys

import pandas as pd

# Make the epftoolbox package importable whether or not it is pip-installed.
# The package lives at <this dir>/epftoolbox/epftoolbox, so its import root is
# the outer <this dir>/epftoolbox folder.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EPFTOOLBOX_ROOT = os.path.join(THIS_DIR, "epftoolbox")
if EPFTOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, EPFTOOLBOX_ROOT)

from epftoolbox.data import read_data  # noqa: E402


def main():
    # Directory where read_data caches the raw NP.csv (downloaded on first run).
    datasets_dir = os.path.join(THIS_DIR, "datasets")

    # read_data splits the data into train/test. years_test=0 keeps everything
    # in the training set so no rows are lost when we recombine.
    df_train, df_test = read_data(path=datasets_dir, dataset="NP", years_test=0)

    # Reconstruct the full dataset in chronological order.
    df_full = pd.concat([df_train, df_test]).sort_index()
    df_full.index.name = "Date"

    output_path = os.path.join(THIS_DIR, "NP_full.csv")
    df_full.to_csv(output_path)

    print(f"Full Nord Pool dataset saved to: {output_path}")
    print(f"Rows: {len(df_full)}")
    print(f"Date range: {df_full.index.min()} -> {df_full.index.max()}")
    print(f"Columns: {list(df_full.columns)}")


if __name__ == "__main__":
    main()
