"""
remoteoutcome: RSV Treatment Effect Estimator

Estimates treatment effects using remotely sensed variables (RSVs) following
Rambachan, Singh, and Viviano (2025).
"""

from .estimator import rsv_estimate
from .result import RSVResult
from .helpers import (
    get_joint,
    count_marginals,
    get_delta,
    get_delta_obs,
    get_sigma2,
    get_theta_init,
    rsv_compute
)
from .prediction import fit_predictions_rf
from .bootstrap import rsv_bootstrap
from .data_constructors import create_data_real, create_data_synth

__version__ = "0.1.0"

__all__ = [
    # Main function
    "rsv_estimate",
    # Result class
    "RSVResult",
    # Helper functions
    "get_joint",
    "count_marginals",
    "get_delta",
    "get_delta_obs",
    "get_sigma2",
    "get_theta_init",
    "rsv_compute",
    # Prediction
    "fit_predictions_rf",
    # Bootstrap
    "rsv_bootstrap",
    # Data constructors
    "create_data_real",
    "create_data_synth",
]
