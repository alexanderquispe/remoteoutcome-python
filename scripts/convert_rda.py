"""
Script to convert R .rda files to Parquet format for Python testing.
"""

import pyreadr
import pandas as pd
import numpy as np
import os


def convert_pred_real_ycons():
    """Convert pred_real_Ycons.rda to parquet."""
    # Path to R data file
    rda_path = r"C:\Users\Alexander\Documents\GitHub\remoteoutcome\data\pred_real_Ycons.rda"

    # Output path
    output_dir = r"C:\Users\Alexander\Documents\GitHub\remoteoutcome-python\data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pred_real_Ycons.parquet")

    # Read R data
    result = pyreadr.read_r(rda_path)

    # Get the dataframe
    df = result['pred_real_Ycons']

    print(f"Loaded data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Get attributes (theta_init is stored as an attribute in R)
    # We'll need to get this separately

    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = convert_pred_real_ycons()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
