"""
Random Forest prediction fitting for RSV estimation.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .helpers import get_theta_init


def fit_predictions_rf(
    R: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    R_pred: Optional[np.ndarray] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Fit predictions using random forest.

    Fits random forest models to predict Y(R), D(R), S_e(R), and S_o(R) using
    the remotely sensed variable R as the predictor.

    Parameters
    ----------
    R : np.ndarray
        Remotely sensed variable (n_samples, n_features).
    Y : np.ndarray
        Outcome variable (binary, NaN for experimental-only sample).
    D : np.ndarray
        Treatment indicator (binary, NaN for observational-only sample).
    S_e : np.ndarray
        Experimental sample indicator (0 or 1).
    S_o : np.ndarray
        Observational sample indicator (0 or 1).
    R_pred : np.ndarray, optional
        Values of R on which to make predictions. Defaults to R.
    ml_params : dict, optional
        Parameters for random forest. Should include:
        - ntree: Number of trees (default 100)
        - classwt_Y: Class weights for pred_Y model (default {0: 10, 1: 1})
        - seed: Random seed for reproducibility (default None)
    n_jobs : int, default 1
        Number of jobs for parallel training.

    Returns
    -------
    dict
        Dictionary containing:
        - theta_init: Initial estimate of the treatment effect
        - predictions: DataFrame with columns Y, D, S_e, S_o
    """
    if R_pred is None:
        R_pred = R

    # Convert to numpy arrays
    R = np.atleast_2d(R)
    if R.shape[0] == 1 and len(Y) > 1:
        R = R.T
    R_pred = np.atleast_2d(R_pred)
    if R_pred.shape[0] == 1 and R_pred.shape[1] == R.shape[1]:
        pass  # Already correct shape
    elif R_pred.ndim == 1:
        R_pred = R_pred.reshape(-1, 1)
    elif R_pred.shape[0] == 1 and R_pred.shape[1] != R.shape[1]:
        R_pred = R_pred.T

    # Convert boolean to int if needed
    S_e = np.asarray(S_e, dtype=int)
    S_o = np.asarray(S_o, dtype=int)
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)

    # Extract ML parameters
    if ml_params is None:
        ml_params = {}

    ntree = ml_params.get('ntree', 100)
    classwt_Y = ml_params.get('classwt_Y', {0: 10, 1: 1})
    seed = ml_params.get('seed', None)

    # Indices for experimental and observational samples
    obs_idx = (S_o == 1)
    if np.sum(obs_idx) == 0:
        raise ValueError("No observations in observational sample")

    exp_idx = (S_e == 1)
    if np.sum(exp_idx) == 0:
        raise ValueError("No observations in experimental sample")

    # Fit E[Y | R, S_o = 1] on observations with Y available
    model_Y = RandomForestClassifier(
        n_estimators=ntree,
        class_weight=classwt_Y,
        n_jobs=n_jobs,
        random_state=seed
    )
    # Filter out NaN values in Y
    valid_y_idx = obs_idx & ~np.isnan(Y)
    model_Y.fit(R[valid_y_idx], Y[valid_y_idx].astype(int))

    # Fit E[D | R, S_e = 1] on observations with D available
    model_D = RandomForestClassifier(
        n_estimators=ntree,
        n_jobs=n_jobs,
        random_state=seed
    )
    # Filter out NaN values in D
    valid_d_idx = exp_idx & ~np.isnan(D)
    model_D.fit(R[valid_d_idx], D[valid_d_idx].astype(int))

    # Fit P(S_e = 1 | R) on full sample
    model_S_e = RandomForestClassifier(
        n_estimators=ntree,
        n_jobs=n_jobs,
        random_state=seed
    )
    model_S_e.fit(R, S_e)

    # Fit P(S_o = 1 | R) on full sample
    model_S_o = RandomForestClassifier(
        n_estimators=ntree,
        n_jobs=n_jobs,
        random_state=seed
    )
    model_S_o.fit(R, S_o)

    # Initial estimate for theta
    observations = pd.DataFrame({
        'Y': Y,
        'D': D,
        'S_e': S_e,
        'S_o': S_o
    })

    predictions_train = pd.DataFrame({
        'Y': model_Y.predict_proba(R)[:, 1],
        'D': model_D.predict_proba(R)[:, 1],
        'S_e': model_S_e.predict_proba(R)[:, 1],
        'S_o': model_S_o.predict_proba(R)[:, 1]
    })

    theta_init = get_theta_init(observations, predictions_train)

    # Generate predictions on R_pred
    predictions = pd.DataFrame({
        'Y': model_Y.predict_proba(R_pred)[:, 1],
        'D': model_D.predict_proba(R_pred)[:, 1],
        'S_e': model_S_e.predict_proba(R_pred)[:, 1],
        'S_o': model_S_o.predict_proba(R_pred)[:, 1]
    })

    return {
        'theta_init': theta_init,
        'predictions': predictions
    }
