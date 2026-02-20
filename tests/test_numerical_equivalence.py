"""
Test numerical equivalence with R package results.

Expected results from README:
| Method      | Coefficient | SE     | n_exp | n_obs | n_both |
|-------------|-------------|--------|-------|-------|--------|
| crossfit    | -0.1082     | -      | -     | -     | -      |
| split       | -0.1086     | 0.0959 | 3032  | 2575  | 1451   |
| none        | -0.0135     | 0.0120 | 6055  | 5186  | 2929   |
| predictions | -0.0135     | 0.0120 | 6055  | 5186  | 2929   |

90% CI (none method): [-0.03319573, 0.006176239]
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from remoteoutcome import rsv_estimate, rsv_compute, get_theta_init


def load_pred_real_ycons():
    """Load the test dataset."""
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'pred_real_Ycons.parquet'
    )
    df = pd.read_parquet(data_path)

    # Convert Y to numeric (it may be object type)
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

    return df


class TestNumericalEquivalence:
    """Test numerical equivalence with R results."""

    @pytest.fixture
    def data(self):
        """Load test data."""
        return load_pred_real_ycons()

    def test_data_loading(self, data):
        """Test that data loads correctly."""
        assert len(data) == 8312
        assert set(data.columns) == {
            'Y', 'D', 'S_e', 'S_o', 'pred_Y', 'pred_D',
            'pred_S_e', 'pred_S_o', 'clusters'
        }

    def test_sample_sizes(self, data):
        """Test that sample sizes match expected values."""
        n_exp = (data['S_e'] == 1).sum()
        n_obs = (data['S_o'] == 1).sum()
        n_both = ((data['S_e'] == 1) & (data['S_o'] == 1)).sum()

        assert n_exp == 6055, f"Expected n_exp=6055, got {n_exp}"
        assert n_obs == 5186, f"Expected n_obs=5186, got {n_obs}"
        assert n_both == 2929, f"Expected n_both=2929, got {n_both}"

    def test_predictions_method_coefficient(self, data):
        """
        Test method='predictions' produces correct coefficient.

        Expected: coef = -0.0135
        """
        # First, compute theta_init from the data
        observations = pd.DataFrame({
            'Y': data['Y'].values,
            'D': data['D'].values,
            'S_e': data['S_e'].values,
            'S_o': data['S_o'].values
        })
        predictions = pd.DataFrame({
            'Y': data['pred_Y'].values,
            'D': data['pred_D'].values,
            'S_e': data['pred_S_e'].values,
            'S_o': data['pred_S_o'].values
        })

        theta_init = get_theta_init(observations, predictions)
        print(f"Computed theta_init: {theta_init}")

        # Now run rsv_compute
        result = rsv_compute(
            observations=observations,
            predictions=predictions,
            theta_init=theta_init,
            eps=1e-2  # Same as R code
        )

        print(f"Coefficient: {result['coef']}")
        print(f"n_exp: {result['n_exp']}")
        print(f"n_obs: {result['n_obs']}")
        print(f"n_both: {result['n_both']}")

        # Check coefficient
        assert abs(result['coef'] - (-0.0135)) < 0.001, \
            f"Expected coef=-0.0135, got {result['coef']}"

        # Check sample sizes
        assert result['n_exp'] == 6055
        assert result['n_obs'] == 5186
        assert result['n_both'] == 2929

    def test_predictions_method_with_bootstrap(self, data):
        """
        Test method='predictions' with bootstrap SE.

        Expected: SE = 0.0120
        """
        theta_init = get_theta_init(
            observations=pd.DataFrame({
                'Y': data['Y'].values,
                'D': data['D'].values,
                'S_e': data['S_e'].values,
                'S_o': data['S_o'].values
            }),
            predictions=pd.DataFrame({
                'Y': data['pred_Y'].values,
                'D': data['pred_D'].values,
                'S_e': data['pred_S_e'].values,
                'S_o': data['pred_S_o'].values
            })
        )

        result = rsv_estimate(
            Y=data['Y'].values,
            D=data['D'].values,
            S_e=data['S_e'].values,
            S_o=data['S_o'].values,
            pred_Y=data['pred_Y'].values,
            pred_D=data['pred_D'].values,
            pred_S_e=data['pred_S_e'].values,
            pred_S_o=data['pred_S_o'].values,
            method='predictions',
            theta_init=theta_init,
            eps=1e-2,
            se=True,
            se_params={
                'B': 1000,
                'fix_seed': True,
                'clusters': data['clusters'].values
            },
            n_jobs=1
        )

        print(f"Coefficient: {result.coef}")
        print(f"SE: {result.se}")

        # Check coefficient
        assert abs(result.coef - (-0.0135)) < 0.001, \
            f"Expected coef=-0.0135, got {result.coef}"

        # Check SE (with wider tolerance due to bootstrap variability)
        assert abs(result.se - 0.0120) < 0.005, \
            f"Expected SE=0.0120, got {result.se}"

    def test_confidence_interval(self, data):
        """
        Test 90% confidence interval.

        Expected 90% CI: [-0.03319573, 0.006176239]
        """
        theta_init = get_theta_init(
            observations=pd.DataFrame({
                'Y': data['Y'].values,
                'D': data['D'].values,
                'S_e': data['S_e'].values,
                'S_o': data['S_o'].values
            }),
            predictions=pd.DataFrame({
                'Y': data['pred_Y'].values,
                'D': data['pred_D'].values,
                'S_e': data['pred_S_e'].values,
                'S_o': data['pred_S_o'].values
            })
        )

        result = rsv_estimate(
            Y=data['Y'].values,
            D=data['D'].values,
            S_e=data['S_e'].values,
            S_o=data['S_o'].values,
            pred_Y=data['pred_Y'].values,
            pred_D=data['pred_D'].values,
            pred_S_e=data['pred_S_e'].values,
            pred_S_o=data['pred_S_o'].values,
            method='predictions',
            theta_init=theta_init,
            eps=1e-2,
            se=True,
            se_params={
                'B': 1000,
                'fix_seed': True,
                'clusters': data['clusters'].values
            },
            n_jobs=1
        )

        ci = result.confint(level=0.90)
        print(f"90% CI: [{ci[0]}, {ci[1]}]")

        # Check CI bounds (with tolerance)
        expected_lower = -0.03319573
        expected_upper = 0.006176239

        # CI depends on SE, so we use wider tolerance
        assert abs(ci[0] - expected_lower) < 0.005, \
            f"Expected CI lower={expected_lower}, got {ci[0]}"
        assert abs(ci[1] - expected_upper) < 0.005, \
            f"Expected CI upper={expected_upper}, got {ci[1]}"


if __name__ == "__main__":
    # Run quick test
    data = load_pred_real_ycons()
    print(f"Loaded data with shape: {data.shape}")

    # Compute theta_init
    observations = pd.DataFrame({
        'Y': data['Y'].values,
        'D': data['D'].values,
        'S_e': data['S_e'].values,
        'S_o': data['S_o'].values
    })
    predictions = pd.DataFrame({
        'Y': data['pred_Y'].values,
        'D': data['pred_D'].values,
        'S_e': data['pred_S_e'].values,
        'S_o': data['pred_S_o'].values
    })

    theta_init = get_theta_init(observations, predictions)
    print(f"\nComputed theta_init: {theta_init}")

    # Run rsv_compute
    result = rsv_compute(
        observations=observations,
        predictions=predictions,
        theta_init=theta_init,
        eps=1e-2
    )

    print(f"\nResults:")
    print(f"  Coefficient: {result['coef']:.6f}")
    print(f"  n_exp: {result['n_exp']}")
    print(f"  n_obs: {result['n_obs']}")
    print(f"  n_both: {result['n_both']}")

    print(f"\nExpected:")
    print(f"  Coefficient: -0.0135")
    print(f"  n_exp: 6055")
    print(f"  n_obs: 5186")
    print(f"  n_both: 2929")
