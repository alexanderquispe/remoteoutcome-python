"""
Quickstart example for remoteoutcome package.

This example demonstrates how to use the RSV Treatment Effect Estimator
with pre-computed predictions.
"""

import sys
import os

# Add package to path if not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from remoteoutcome import rsv_estimate, get_theta_init


def main():
    """Run the quickstart example."""
    # Load the test dataset
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'pred_real_Ycons.parquet'
    )
    df = pd.read_parquet(data_path)

    # Convert Y to numeric
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

    print("Loaded data:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print()

    # Compute theta_init from the data
    observations = pd.DataFrame({
        'Y': df['Y'].values,
        'D': df['D'].values,
        'S_e': df['S_e'].values,
        'S_o': df['S_o'].values
    })
    predictions = pd.DataFrame({
        'Y': df['pred_Y'].values,
        'D': df['pred_D'].values,
        'S_e': df['pred_S_e'].values,
        'S_o': df['pred_S_o'].values
    })

    theta_init = get_theta_init(observations, predictions)
    print(f"Computed theta_init: {theta_init:.6f}")
    print()

    # Run RSV estimation with bootstrap SE
    print("Running RSV estimation with bootstrap SE (this may take a minute)...")
    result = rsv_estimate(
        Y=df['Y'].values,
        D=df['D'].values,
        S_e=df['S_e'].values,
        S_o=df['S_o'].values,
        pred_Y=df['pred_Y'].values,
        pred_D=df['pred_D'].values,
        pred_S_e=df['pred_S_e'].values,
        pred_S_o=df['pred_S_o'].values,
        method='predictions',
        theta_init=theta_init,
        eps=1e-2,
        se=True,
        se_params={
            'B': 1000,
            'fix_seed': True,
            'clusters': df['clusters'].values
        },
        n_jobs=1
    )

    # Print results
    print()
    print(result)
    print()

    # Confidence intervals
    ci_90 = result.confint(level=0.90)
    ci_95 = result.confint(level=0.95)

    print(f"90% CI: [{ci_90[0]:.6f}, {ci_90[1]:.6f}]")
    print(f"95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
    print()

    # Detailed summary
    print("Detailed summary:")
    print(result.summary())


if __name__ == "__main__":
    main()
