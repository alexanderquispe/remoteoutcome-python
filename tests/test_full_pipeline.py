"""
Test the full RSV estimation pipeline with smartcard data.

This test uses the raw smartcard data to test:
1. Data loading and merging
2. create_data_real() function
3. RF prediction fitting
4. RSV estimation with method='none'
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from remoteoutcome import (
    rsv_estimate,
    create_data_real,
    fit_predictions_rf,
    rsv_compute,
    get_theta_init
)


def load_smartcard_data():
    """Load and merge smartcard data."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    p1_path = os.path.join(data_dir, 'smartcard_data_p1.parquet')
    p2_path = os.path.join(data_dir, 'smartcard_data_p2.parquet')

    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        pytest.skip("Smartcard data files not found")

    df1 = pd.read_parquet(p1_path)
    df2 = pd.read_parquet(p2_path)

    # Merge on shrid2
    df = df1.merge(df2, on='shrid2', how='inner')

    return df


class TestFullPipeline:
    """Test the full RSV estimation pipeline."""

    @pytest.fixture
    def smartcard_data(self):
        """Load smartcard data."""
        return load_smartcard_data()

    def test_data_loading(self, smartcard_data):
        """Test that smartcard data loads correctly."""
        assert len(smartcard_data) == 8312
        assert 'shrid2' in smartcard_data.columns
        assert 'D' in smartcard_data.columns
        assert 'Ycons' in smartcard_data.columns
        assert 'Sample (Smartcard)' in smartcard_data.columns

    def test_create_data_real(self, smartcard_data):
        """Test create_data_real function."""
        data_real = create_data_real(smartcard_data)

        # Check columns
        assert 'S' in data_real.columns
        assert 'D' in data_real.columns
        assert 'Ycons' in data_real.columns
        assert 'clusters' in data_real.columns

        # Check sample indicators
        assert set(data_real['S'].dropna().unique()) == {'e', 'o', 'both'}

        # Check that D is NA for observational-only
        obs_only = data_real[data_real['S'] == 'o']
        assert obs_only['D'].isna().all()

        # Check that Y is NA for experimental-only
        exp_only = data_real[data_real['S'] == 'e']
        assert exp_only['Ycons'].isna().all()

    def test_rf_prediction_fitting(self, smartcard_data):
        """Test RF prediction fitting on a small subset."""
        data_real = create_data_real(smartcard_data)

        # Extract variables
        Y = data_real['Ycons'].values
        D = data_real['D'].values

        # Get R (remotely sensed variables) - use a subset for speed
        r_cols = [c for c in data_real.columns if c.startswith('luminosity_')][:5]
        if len(r_cols) < 5:
            r_cols = [c for c in data_real.columns if c.startswith('satellite_')][:5]
        R = data_real[r_cols].values

        # Create sample indicators
        S_e = (~np.isnan(D)) & (np.sum(np.isnan(R), axis=1) == 0)
        S_o = (~np.isnan(Y)) & (np.sum(np.isnan(R), axis=1) == 0)

        # Use a small subset for quick testing
        n_subset = 500
        indices = np.random.choice(len(Y), size=n_subset, replace=False)

        # Fit predictions
        result = fit_predictions_rf(
            R=R[indices],
            Y=Y[indices],
            D=D[indices],
            S_e=S_e[indices].astype(int),
            S_o=S_o[indices].astype(int),
            ml_params={'ntree': 10, 'seed': 42},  # Small for speed
            n_jobs=1
        )

        # Check outputs
        assert 'theta_init' in result
        assert 'predictions' in result
        assert np.isfinite(result['theta_init'])
        assert len(result['predictions']) == n_subset

    def test_method_none_small(self, smartcard_data):
        """Test method='none' on a small subset."""
        data_real = create_data_real(smartcard_data)

        # Extract variables
        Y = data_real['Ycons'].values
        D = data_real['D'].values
        clusters = data_real['clusters'].values

        # Get R - use only luminosity features for speed
        r_cols = [c for c in data_real.columns if c.startswith('luminosity_')]
        R = data_real[r_cols].values

        # Create sample indicators
        S_e = (~np.isnan(D)) & (np.sum(np.isnan(R), axis=1) == 0)
        S_o = (~np.isnan(Y)) & (np.sum(np.isnan(R), axis=1) == 0)

        # Use a small subset for quick testing
        n_subset = 300
        np.random.seed(42)
        indices = np.random.choice(len(Y), size=n_subset, replace=False)

        # Run RSV estimation
        result = rsv_estimate(
            Y=Y[indices],
            D=D[indices],
            S_e=S_e[indices].astype(int),
            S_o=S_o[indices].astype(int),
            R=R[indices],
            method='none',
            ml_params={'ntree': 10, 'seed': 42},
            se=False,  # Skip bootstrap for speed
            n_jobs=1
        )

        # Check result
        assert result.coef is not None
        assert np.isfinite(result.coef)
        assert result.n_exp is not None
        assert result.n_obs is not None
        print(f"\nSmall subset result: coef={result.coef:.4f}")


class TestMethodComparison:
    """Compare different estimation methods."""

    @pytest.fixture
    def test_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n = 200

        # Generate synthetic data
        R = np.random.randn(n, 3)
        S_e = np.random.binomial(1, 0.7, n)
        S_o = np.random.binomial(1, 0.6, n)

        # Generate D and Y based on R
        D = np.where(S_e == 1, np.random.binomial(1, 0.5, n), np.nan)
        Y = np.where(S_o == 1, np.random.binomial(1, 0.3, n), np.nan)

        return {
            'Y': Y,
            'D': D,
            'S_e': S_e,
            'S_o': S_o,
            'R': R
        }

    def test_method_none_runs(self, test_data):
        """Test that method='none' runs without error."""
        result = rsv_estimate(
            **test_data,
            method='none',
            ml_params={'ntree': 10, 'seed': 42},
            se=False,
            n_jobs=1
        )
        assert np.isfinite(result.coef)
        print(f"method='none': coef={result.coef:.4f}")

    def test_method_split_runs(self, test_data):
        """Test that method='split' runs without error."""
        result = rsv_estimate(
            **test_data,
            method='split',
            ml_params={'ntree': 10, 'seed': 42, 'train_ratio': 0.5},
            se=False,
            n_jobs=1
        )
        assert np.isfinite(result.coef)
        print(f"method='split': coef={result.coef:.4f}")

    def test_method_crossfit_runs(self, test_data):
        """Test that method='crossfit' runs without error."""
        result = rsv_estimate(
            **test_data,
            method='crossfit',
            ml_params={'ntree': 10, 'seed': 42, 'nfolds': 3},
            se=False,
            n_jobs=1
        )
        assert np.isfinite(result.coef)
        print(f"method='crossfit': coef={result.coef:.4f}")


if __name__ == "__main__":
    # Quick test
    print("Loading smartcard data...")
    try:
        df = load_smartcard_data()
        print(f"Loaded {len(df)} rows")

        print("\nCreating data_real...")
        data_real = create_data_real(df)

        print("\nRunning quick estimation test...")
        Y = data_real['Ycons'].values
        D = data_real['D'].values

        r_cols = [c for c in data_real.columns if c.startswith('luminosity_')]
        R = data_real[r_cols].values

        S_e = (~np.isnan(D)) & (np.sum(np.isnan(R), axis=1) == 0)
        S_o = (~np.isnan(Y)) & (np.sum(np.isnan(R), axis=1) == 0)

        # Small subset
        n_subset = 300
        np.random.seed(42)
        indices = np.random.choice(len(Y), size=n_subset, replace=False)

        result = rsv_estimate(
            Y=Y[indices],
            D=D[indices],
            S_e=S_e[indices].astype(int),
            S_o=S_o[indices].astype(int),
            R=R[indices],
            method='none',
            ml_params={'ntree': 10, 'seed': 42},
            se=False,
            n_jobs=1
        )

        print(f"\nResult: coef={result.coef:.4f}")
        print("Test passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
