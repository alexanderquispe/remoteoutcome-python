"""
RSV Helper Functions (Binary Outcome)
Following main text Algorithm 1 with Remark 3 generalization
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def get_joint(x: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute joint probabilities for treatment/outcome and sample membership.

    Parameters
    ----------
    x : pd.DataFrame
        DataFrame with columns 'D', 'Y', 'S_e', and 'S_o'.

    Returns
    -------
    dict
        Dictionary containing:
        - D1Se: P(D=1, S_e=1 | R) = P(D=1 | S_e=1, R) * P(S_e=1 | R)
        - D0Se: P(D=0, S_e=1 | R) = P(D=0 | S_e=1, R) * P(S_e=1 | R)
        - Y1So: P(Y=1, S_o=1 | R) = P(Y=1 | S_o=1, R) * P(S_o=1 | R)
        - Y0So: P(Y=0, S_o=1 | R) = P(Y=0 | S_o=1, R) * P(S_o=1 | R)
    """
    D = np.asarray(x['D'], dtype=float)
    Y = np.asarray(x['Y'], dtype=float)
    S_e = np.asarray(x['S_e'], dtype=float)
    S_o = np.asarray(x['S_o'], dtype=float)

    return {
        'D1Se': D * S_e,
        'D0Se': (1 - D) * S_e,
        'Y1So': Y * S_o,
        'Y0So': (1 - Y) * S_o
    }


def count_marginals(observations: pd.DataFrame) -> Dict[str, float]:
    """
    Compute sample means of joint probabilities.

    Used to calculate marginal probabilities for treatment/outcome and
    sample membership.

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame with columns 'D', 'Y', 'S_e', and 'S_o' containing
        observed values.

    Returns
    -------
    dict
        Dictionary containing mean values of joint probabilities.
    """
    # Create a copy to avoid modifying the original
    obs_copy = observations.copy()

    # Convert to binary indicators as in R: (Y == 1) & (S_o == 1)
    Y = np.asarray(obs_copy['Y'], dtype=float)
    D = np.asarray(obs_copy['D'], dtype=float)
    S_e = np.asarray(obs_copy['S_e'], dtype=float)
    S_o = np.asarray(obs_copy['S_o'], dtype=float)

    obs_copy['Y'] = ((Y == 1) & (S_o == 1)).astype(int)
    obs_copy['D'] = ((D == 1) & (S_e == 1)).astype(int)

    j = get_joint(obs_copy)

    return {k: np.mean(v) for k, v in j.items()}


def get_delta(observations: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute treatment and outcome variations for RSV estimation (Deltas).

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (observed values).
    predictions : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (predicted probabilities).

    Returns
    -------
    dict
        Dictionary containing:
        - e: Treatment variation in experimental sample
        - o: Outcome variation in observational sample
    """
    j = get_joint(predictions)
    count = count_marginals(observations)

    return {
        'e': j['D1Se'] / count['D1Se'] - j['D0Se'] / count['D0Se'],  # treatment variations from Se
        'o': j['Y1So'] / count['Y1So'] - j['Y0So'] / count['Y0So']   # outcome variations from So
    }


def get_delta_obs(observations: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute treatment and outcome variations from observations only.

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (observed values).

    Returns
    -------
    dict
        Dictionary containing:
        - e: Treatment variation in experimental sample
        - o: Outcome variation in observational sample
    """
    # Create a copy to avoid modifying the original
    obs_copy = observations.copy()

    Y = np.asarray(obs_copy['Y'], dtype=float)
    D = np.asarray(obs_copy['D'], dtype=float)
    S_e = np.asarray(obs_copy['S_e'], dtype=float)
    S_o = np.asarray(obs_copy['S_o'], dtype=float)

    obs_copy['Y'] = ((Y == 1) & (S_o == 1)).astype(int)
    obs_copy['D'] = ((D == 1) & (S_e == 1)).astype(int)

    j = get_joint(obs_copy)
    count = count_marginals(observations)

    return {
        'e': j['D1Se'] / count['D1Se'] - j['D0Se'] / count['D0Se'],  # treatment variations from Se
        'o': j['Y1So'] / count['Y1So'] - j['Y0So'] / count['Y0So']   # outcome variations from So
    }


def get_sigma2(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    theta_init: float,
    eps: Optional[float] = None,
    eps_prob: float = 0.01
) -> np.ndarray:
    """
    Compute variance component sigma-squared.

    Computes the variance component used in the efficient weight function
    for RSV estimation.

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (observed values).
    predictions : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (predicted probabilities).
    theta_init : float
        Initial estimate of the treatment effect.
    eps : float, optional
        Small positive constant for numerical stability.
        If None, uses quantile-based lower bound.
    eps_prob : float, default 0.01
        Probability for quantile-based lower bound if eps is None.

    Returns
    -------
    np.ndarray
        Vector of sigma-squared values, one for each observation.
    """
    j = get_joint(predictions)
    count = count_marginals(observations)

    sigma2 = (
        j['D1Se'] / (count['D1Se'] ** 2) +
        j['D0Se'] / (count['D0Se'] ** 2) +
        (theta_init ** 2) * (
            j['Y1So'] / (count['Y1So'] ** 2) +
            j['Y0So'] / (count['Y0So'] ** 2)
        )
    )

    # Lower bound on sigma for numerical stability
    if eps is None:
        eps = np.quantile(sigma2, eps_prob)

    sigma2 = np.maximum(sigma2, eps)

    return sigma2


def get_theta_init(observations: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Compute initial treatment effect estimate.

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (observed values).
    predictions : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (predicted probabilities).

    Returns
    -------
    float
        Initial treatment effect estimate.
    """
    # Create modified predictions with observed sample indicators
    pred_mod = predictions.copy()
    pred_mod['S_e'] = observations['S_e'].values
    pred_mod['S_o'] = observations['S_o'].values

    delta = get_delta(observations, pred_mod)

    # theta_init = mean(Delta_e * Delta_o) / mean(Delta_o^2)
    numerator = np.nanmean(delta['e'] * delta['o'])
    denominator = np.nanmean(delta['o'] ** 2)

    return numerator / denominator


def rsv_compute(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    theta_init: float,
    eps: Optional[float] = None,
    eps_prob: float = 0.01
) -> dict:
    """
    Compute RSV estimator from predictions.

    Implements Algorithm 1 from main text with Remark 3 generalization
    allowing observations to be in experimental only, observational only, or both.

    Parameters
    ----------
    observations : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (observed values).
    predictions : pd.DataFrame
        DataFrame containing Y, D, S_e, S_o (predicted probabilities).
    theta_init : float
        Initial estimate of the treatment effect.
    eps : float, optional
        Small constant for numerical stability.
    eps_prob : float, default 0.01
        Probability for quantile-based lower bound if eps is None.

    Returns
    -------
    dict
        Dictionary containing:
        - coef: Treatment effect estimate
        - weights: Efficient weights used in estimation
        - n_obs: Sample size in observational sample
        - n_exp: Sample size in experimental sample
        - n_both: Sample size in both samples
        - numerator: Numerator of the treatment effect estimate
        - denominator: Denominator of the treatment effect estimate
    """
    if len(observations) != len(predictions):
        raise ValueError(
            "observations and predictions must have the same number of rows"
        )

    # Convert boolean columns to numeric if needed
    obs = observations.copy()
    if obs['S_e'].dtype == bool:
        obs['S_e'] = obs['S_e'].astype(int)
    if obs['S_o'].dtype == bool:
        obs['S_o'] = obs['S_o'].astype(int)

    # Sample sizes
    S_e = np.asarray(obs['S_e'])
    S_o = np.asarray(obs['S_o'])
    n_exp = int(np.sum(S_e == 1))
    n_obs = int(np.sum(S_o == 1))
    n_both = int(np.sum((S_e == 1) & (S_o == 1)))

    # Compute sigma^2 (Step 2d)
    sigma2 = get_sigma2(
        observations=obs,
        predictions=predictions,
        theta_init=theta_init,
        eps=eps,
        eps_prob=eps_prob
    )

    # Efficient weight function
    delta_pred = get_delta(observations=obs, predictions=predictions)
    h = delta_pred['o'] / sigma2

    # Compute weighted estimator
    delta_obs = get_delta_obs(obs)
    numerator = np.mean(delta_obs['e'] * h)
    denominator = np.mean(delta_obs['o'] * h)
    coef = numerator / denominator

    return {
        'coef': coef,
        'weights': h,
        'n_obs': n_obs,
        'n_exp': n_exp,
        'n_both': n_both,
        'numerator': numerator,
        'denominator': denominator,
        'theta_init': theta_init,
        'observations': obs,
        'predictions': predictions
    }
