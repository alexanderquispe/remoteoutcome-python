"""
Tests for helper functions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from remoteoutcome.helpers import (
    get_joint,
    count_marginals,
    get_delta,
    get_delta_obs,
    get_sigma2,
    get_theta_init,
    rsv_compute
)


class TestGetJoint:
    """Tests for get_joint function."""

    def test_basic_joint_probabilities(self):
        """Test basic joint probability calculations."""
        df = pd.DataFrame({
            'D': [1, 0, 1, 0],
            'Y': [1, 1, 0, 0],
            'S_e': [1, 1, 1, 1],
            'S_o': [1, 1, 1, 1]
        })

        result = get_joint(df)

        # D1Se = D * S_e
        np.testing.assert_array_equal(result['D1Se'], [1, 0, 1, 0])
        # D0Se = (1 - D) * S_e
        np.testing.assert_array_equal(result['D0Se'], [0, 1, 0, 1])
        # Y1So = Y * S_o
        np.testing.assert_array_equal(result['Y1So'], [1, 1, 0, 0])
        # Y0So = (1 - Y) * S_o
        np.testing.assert_array_equal(result['Y0So'], [0, 0, 1, 1])

    def test_with_zero_indicators(self):
        """Test joint probabilities with zero sample indicators."""
        df = pd.DataFrame({
            'D': [1, 1, 1, 1],
            'Y': [1, 1, 1, 1],
            'S_e': [0, 0, 1, 1],
            'S_o': [1, 1, 0, 0]
        })

        result = get_joint(df)

        # D1Se = D * S_e = [0, 0, 1, 1]
        np.testing.assert_array_equal(result['D1Se'], [0, 0, 1, 1])
        # Y1So = Y * S_o = [1, 1, 0, 0]
        np.testing.assert_array_equal(result['Y1So'], [1, 1, 0, 0])


class TestCountMarginals:
    """Tests for count_marginals function."""

    def test_basic_marginals(self):
        """Test basic marginal probability calculations."""
        df = pd.DataFrame({
            'Y': [1, 1, 0, 0],
            'D': [1, 0, 1, 0],
            'S_e': [1, 1, 1, 1],
            'S_o': [1, 1, 1, 1]
        })

        result = count_marginals(df)

        # D1Se mean = mean([1, 0, 1, 0]) = 0.5
        assert result['D1Se'] == 0.5
        # D0Se mean = mean([0, 1, 0, 1]) = 0.5
        assert result['D0Se'] == 0.5
        # Y1So mean = mean([1, 1, 0, 0]) = 0.5
        assert result['Y1So'] == 0.5
        # Y0So mean = mean([0, 0, 1, 1]) = 0.5
        assert result['Y0So'] == 0.5


class TestGetSigma2:
    """Tests for get_sigma2 function."""

    def test_sigma2_lower_bound(self):
        """Test that sigma2 is lower-bounded by eps."""
        observations = pd.DataFrame({
            'Y': [1, 1, 0, 0],
            'D': [1, 0, 1, 0],
            'S_e': [1, 1, 1, 1],
            'S_o': [1, 1, 1, 1]
        })
        predictions = pd.DataFrame({
            'Y': [0.8, 0.7, 0.3, 0.2],
            'D': [0.6, 0.4, 0.6, 0.4],
            'S_e': [0.5, 0.5, 0.5, 0.5],
            'S_o': [0.5, 0.5, 0.5, 0.5]
        })

        eps = 1e-2
        result = get_sigma2(
            observations=observations,
            predictions=predictions,
            theta_init=0.1,
            eps=eps
        )

        # All values should be >= eps
        assert np.all(result >= eps)


class TestRsvCompute:
    """Tests for rsv_compute function."""

    def test_sample_sizes(self):
        """Test that sample sizes are computed correctly."""
        observations = pd.DataFrame({
            'Y': [1, 1, 0, 0, 1, 0],
            'D': [1, 0, 1, 0, np.nan, np.nan],
            'S_e': [1, 1, 1, 1, 0, 0],
            'S_o': [1, 1, 0, 0, 1, 1]
        })
        predictions = pd.DataFrame({
            'Y': [0.8, 0.7, 0.3, 0.2, 0.6, 0.4],
            'D': [0.6, 0.4, 0.6, 0.4, 0.5, 0.5],
            'S_e': [0.5, 0.5, 0.5, 0.5, 0.3, 0.3],
            'S_o': [0.5, 0.5, 0.3, 0.3, 0.7, 0.7]
        })

        result = rsv_compute(
            observations=observations,
            predictions=predictions,
            theta_init=0.1,
            eps=1e-4
        )

        assert result['n_exp'] == 4  # S_e == 1
        assert result['n_obs'] == 4  # S_o == 1
        assert result['n_both'] == 2  # S_e == 1 and S_o == 1

    def test_output_structure(self):
        """Test that rsv_compute returns all expected keys."""
        observations = pd.DataFrame({
            'Y': [1, 1, 0, 0],
            'D': [1, 0, 1, 0],
            'S_e': [1, 1, 1, 1],
            'S_o': [1, 1, 1, 1]
        })
        predictions = pd.DataFrame({
            'Y': [0.8, 0.7, 0.3, 0.2],
            'D': [0.6, 0.4, 0.6, 0.4],
            'S_e': [0.5, 0.5, 0.5, 0.5],
            'S_o': [0.5, 0.5, 0.5, 0.5]
        })

        result = rsv_compute(
            observations=observations,
            predictions=predictions,
            theta_init=0.1,
            eps=1e-4
        )

        expected_keys = {
            'coef', 'weights', 'n_obs', 'n_exp', 'n_both',
            'numerator', 'denominator', 'theta_init',
            'observations', 'predictions'
        }
        assert set(result.keys()) == expected_keys


class TestGetThetaInit:
    """Tests for get_theta_init function."""

    def test_theta_init_finite(self):
        """Test that theta_init returns a finite value."""
        observations = pd.DataFrame({
            'Y': [1, 1, 0, 0],
            'D': [1, 0, 1, 0],
            'S_e': [1, 1, 1, 1],
            'S_o': [1, 1, 1, 1]
        })
        predictions = pd.DataFrame({
            'Y': [0.8, 0.7, 0.3, 0.2],
            'D': [0.6, 0.4, 0.6, 0.4],
            'S_e': [0.5, 0.5, 0.5, 0.5],
            'S_o': [0.5, 0.5, 0.5, 0.5]
        })

        theta_init = get_theta_init(observations, predictions)

        assert np.isfinite(theta_init)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
