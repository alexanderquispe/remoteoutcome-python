"""
RSV Estimator Main Function
"""

from typing import Optional, Dict, Any, Union, Literal
import numpy as np
import pandas as pd

from .helpers import rsv_compute, get_theta_init
from .prediction import fit_predictions_rf
from .bootstrap import rsv_bootstrap
from .result import RSVResult


def rsv_estimate(
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    R: Optional[np.ndarray] = None,
    pred_Y: Optional[np.ndarray] = None,
    pred_D: Optional[np.ndarray] = None,
    pred_S_e: Optional[np.ndarray] = None,
    pred_S_o: Optional[np.ndarray] = None,
    theta_init: Optional[float] = None,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    method: Literal["crossfit", "split", "none", "predictions"] = "crossfit",
    ml_params: Optional[Dict[str, Any]] = None,
    se: bool = True,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> RSVResult:
    """
    RSV Treatment Effect Estimator.

    Estimates treatment effects using remotely sensed variables (RSVs) following
    Rambachan, Singh, and Viviano (2025). Implements Algorithm 1 from the main
    text for binary outcomes without pretreatment covariates.

    The function supports two interfaces:
    1. Provide fitted predictions (pred_Y, pred_D, pred_S_e, pred_S_o) directly.
    2. Provide raw data (Y, D, S, R) and the function fits predictions using
       random forests.

    Parameters
    ----------
    Y : np.ndarray
        Outcome variable (binary, NaN where not observed).
    D : np.ndarray
        Treatment indicator (binary, NaN where not observed).
    S_e : np.ndarray
        Experimental sample indicator (0 or 1).
    S_o : np.ndarray
        Observational sample indicator (0 or 1).
    R : np.ndarray, optional
        Remotely sensed variable. Required if predictions are not provided.
    pred_Y : np.ndarray, optional
        Predicted P(Y=1 | R, S_o=1). If provided, other predictions must also
        be provided.
    pred_D : np.ndarray, optional
        Predicted P(D=1 | R, S_e=1).
    pred_S_e : np.ndarray, optional
        Predicted P(S_e=1 | R).
    pred_S_o : np.ndarray, optional
        Predicted P(S_o=1 | R).
    theta_init : float, optional
        Initial estimate of the treatment effect on the train data.
    eps : float, optional
        Small constant for numerical stability of sigma2 estimate.
    eps_prob : float, default 0.01
        Probability for quantile-based lower bound if eps is None.
    method : str, default "crossfit"
        Prediction fitting method; one of "split", "crossfit", "none", or
        "predictions".
        - "split": simple sample split
        - "crossfit": K-fold cross-fitting
        - "none": use all data for training/testing
        - "predictions": use provided predictions
    ml_params : dict, optional
        Parameters for random forest:
        - ntree: Number of trees (default 100)
        - classwt_Y: Class weights for pred_Y model (default {0: 10, 1: 1})
        - seed: Random seed for reproducibility (default None)
        - nfolds: Number of folds for cross-fitting (default 5)
        - train_ratio: Proportion for training in sample split (default 0.5)
    se : bool, default True
        Compute standard errors via bootstrap.
    se_params : dict, optional
        Bootstrap parameters:
        - B: Number of bootstrap replications (default 1000)
        - fix_seed: If True, use deterministic seeding (default False)
        - clusters: Cluster identifiers for clustered sampling
    n_jobs : int, default 1
        Number of cores for parallel computation.

    Returns
    -------
    RSVResult
        Object containing estimation results with attributes:
        - coef: Treatment effect estimate
        - se: Standard error (if se=True)
        - n_obs: Sample size in observational sample
        - n_exp: Sample size in experimental sample
        - n_both: Sample size in both samples
        - method: Prediction fitting method used

    Examples
    --------
    >>> # Example with pre-computed predictions
    >>> result = rsv_estimate(
    ...     Y=data['Y'],
    ...     D=data['D'],
    ...     S_e=data['S_e'],
    ...     S_o=data['S_o'],
    ...     pred_Y=data['pred_Y'],
    ...     pred_D=data['pred_D'],
    ...     pred_S_e=data['pred_S_e'],
    ...     pred_S_o=data['pred_S_o'],
    ...     method='predictions',
    ...     theta_init=0.1
    ... )
    >>> print(result)
    """
    # Set default parameters
    if ml_params is None:
        ml_params = {}

    ml_defaults = {
        'ntree': 100,
        'classwt_Y': {0: 10, 1: 1},
        'seed': None,
        'train_ratio': 0.5,
        'nfolds': 5
    }
    ml_params = {**ml_defaults, **ml_params}

    if se_params is None:
        se_params = {}

    se_defaults = {
        'B': 1000,
        'fix_seed': False,
        'clusters': None
    }
    se_params = {**se_defaults, **se_params}

    # Check required inputs
    if Y is None or D is None or S_e is None or S_o is None:
        raise ValueError("Y, D, S_e, and S_o must be provided")

    # Convert inputs to numpy arrays
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    S_e = np.asarray(S_e)
    S_o = np.asarray(S_o)

    # Convert boolean to int if needed
    if S_e.dtype == bool:
        S_e = S_e.astype(int)
    if S_o.dtype == bool:
        S_o = S_o.astype(int)

    if method == "predictions":
        # Interface 1: User provides predictions
        if pred_Y is None or pred_D is None or pred_S_e is None or pred_S_o is None:
            raise ValueError(
                "pred_Y, pred_D, pred_S_e, pred_S_o must be provided when "
                "method='predictions'"
            )

        result = _rsv_from_predictions(
            Y=Y, D=D, S_e=S_e, S_o=S_o,
            pred_Y=pred_Y, pred_D=pred_D, pred_S_e=pred_S_e, pred_S_o=pred_S_o,
            theta_init=theta_init,
            eps=eps, eps_prob=eps_prob,
            ml_params=ml_params,
            se=se, se_params=se_params,
            n_jobs=n_jobs
        )

    else:
        # Interface 2: Fit predictions from raw data
        if R is None:
            raise ValueError("R must be provided to fit predictions from raw data")

        R = np.atleast_2d(R)
        if R.shape[0] == 1 and len(Y) > 1:
            R = R.T

        if method == "none":
            result = _rsv_fit_none(
                R=R, Y=Y, D=D, S_e=S_e, S_o=S_o,
                eps=eps, eps_prob=eps_prob,
                ml_params=ml_params,
                se=se, se_params=se_params,
                n_jobs=n_jobs
            )
        elif method == "split":
            result = _rsv_fit_split(
                R=R, Y=Y, D=D, S_e=S_e, S_o=S_o,
                eps=eps, eps_prob=eps_prob,
                ml_params=ml_params,
                se=se, se_params=se_params,
                n_jobs=n_jobs
            )
        elif method == "crossfit":
            result = _rsv_fit_crossfit(
                R=R, Y=Y, D=D, S_e=S_e, S_o=S_o,
                eps=eps, eps_prob=eps_prob,
                ml_params=ml_params,
                se=se, se_params=se_params,
                n_jobs=n_jobs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Set method
    result.method = method

    return result


def _rsv_from_predictions(
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    pred_Y: np.ndarray,
    pred_D: np.ndarray,
    pred_S_e: np.ndarray,
    pred_S_o: np.ndarray,
    theta_init: Optional[float] = None,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    ml_params: Optional[Dict[str, Any]] = None,
    se: bool = True,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> RSVResult:
    """
    Estimate from predictions without fitting.

    Uses all data as a test set.
    """
    pred_Y = np.asarray(pred_Y)
    pred_D = np.asarray(pred_D)
    pred_S_e = np.asarray(pred_S_e)
    pred_S_o = np.asarray(pred_S_o)

    if theta_init is None:
        n = len(Y)
        train_ratio = ml_params.get('train_ratio', 0.2) if ml_params else 0.2
        seed = ml_params.get('seed', None) if ml_params else None

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Create train/test split
        indices = np.arange(n)
        train_idx = np.random.choice(
            indices, size=int(train_ratio * n), replace=False
        )
        test_idx = np.setdiff1d(indices, train_idx)

        predictions_train = pd.DataFrame({
            'Y': pred_Y[train_idx],
            'D': pred_D[train_idx],
            'S_e': pred_S_e[train_idx],
            'S_o': pred_S_o[train_idx]
        })
        observations_train = pd.DataFrame({
            'Y': Y[train_idx],
            'D': D[train_idx],
            'S_e': S_e[train_idx],
            'S_o': S_o[train_idx]
        })

        theta_init = get_theta_init(
            observations=observations_train,
            predictions=predictions_train
        )

        predictions = pd.DataFrame({
            'Y': pred_Y[test_idx],
            'D': pred_D[test_idx],
            'S_e': pred_S_e[test_idx],
            'S_o': pred_S_o[test_idx]
        })
        observations = pd.DataFrame({
            'Y': Y[test_idx],
            'D': D[test_idx],
            'S_e': S_e[test_idx],
            'S_o': S_o[test_idx]
        })

        # Update clusters for test set
        if se_params and se_params.get('clusters') is not None:
            se_params = se_params.copy()
            se_params['clusters'] = np.asarray(se_params['clusters'])[test_idx]

    else:
        predictions = pd.DataFrame({
            'Y': pred_Y,
            'D': pred_D,
            'S_e': pred_S_e,
            'S_o': pred_S_o
        })
        observations = pd.DataFrame({
            'Y': Y,
            'D': D,
            'S_e': S_e,
            'S_o': S_o
        })

    # Compute RSV estimator
    result = rsv_compute(
        observations=observations,
        predictions=predictions,
        theta_init=theta_init,
        eps=eps,
        eps_prob=eps_prob
    )

    # Compute standard errors if requested
    se_value = None
    denominator_se = None
    clusters = None

    if se:
        boot_result = rsv_bootstrap(
            observations=observations,
            predictions=predictions,
            theta_init=theta_init,
            eps=eps,
            eps_prob=eps_prob,
            se_params=se_params,
            n_jobs=n_jobs
        )
        se_value = boot_result['se']
        denominator_se = boot_result['denominator_se']
        clusters = boot_result['clusters']

    return RSVResult(
        coef=result['coef'],
        se=se_value,
        n_exp=result['n_exp'],
        n_obs=result['n_obs'],
        n_both=result['n_both'],
        numerator=result['numerator'],
        denominator=result['denominator'],
        denominator_se=denominator_se,
        theta_init=result['theta_init'],
        weights=result['weights'],
        clusters=clusters
    )


def _rsv_fit_none(
    R: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    ml_params: Optional[Dict[str, Any]] = None,
    se: bool = True,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> RSVResult:
    """
    Fit predictions with no split.

    Uses all data for both training and testing.
    """
    # Fit on all data, predict on all data
    out = fit_predictions_rf(
        R=R, Y=Y, D=D, S_e=S_e, S_o=S_o,
        R_pred=R,
        ml_params=ml_params,
        n_jobs=n_jobs
    )

    theta_init = out['theta_init']
    predictions = out['predictions']
    observations = pd.DataFrame({
        'Y': Y,
        'D': D,
        'S_e': S_e,
        'S_o': S_o
    })

    # Compute RSV estimator
    result = rsv_compute(
        observations=observations,
        predictions=predictions,
        theta_init=theta_init,
        eps=eps,
        eps_prob=eps_prob
    )

    # Compute standard errors if requested
    se_value = None
    denominator_se = None
    clusters = None

    if se:
        boot_result = rsv_bootstrap(
            observations=observations,
            predictions=predictions,
            theta_init=theta_init,
            eps=eps,
            eps_prob=eps_prob,
            se_params=se_params,
            n_jobs=n_jobs
        )
        se_value = boot_result['se']
        denominator_se = boot_result['denominator_se']
        clusters = boot_result['clusters']

    return RSVResult(
        coef=result['coef'],
        se=se_value,
        n_exp=result['n_exp'],
        n_obs=result['n_obs'],
        n_both=result['n_both'],
        numerator=result['numerator'],
        denominator=result['denominator'],
        denominator_se=denominator_se,
        theta_init=result['theta_init'],
        weights=result['weights'],
        clusters=clusters
    )


def _rsv_fit_split(
    R: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    ml_params: Optional[Dict[str, Any]] = None,
    se: bool = True,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> RSVResult:
    """
    Fit predictions with sample splitting.

    Splits data into training and test sets, fits predictions on training set,
    and returns predictions on test set.
    """
    n = len(Y)
    train_ratio = ml_params.get('train_ratio', 0.5) if ml_params else 0.5
    seed = ml_params.get('seed', None) if ml_params else None

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Create train/test split
    indices = np.arange(n)
    train_idx = np.random.choice(
        indices, size=int(train_ratio * n), replace=False
    )
    test_idx = np.setdiff1d(indices, train_idx)

    # Fit on training data, predict on test data
    out = fit_predictions_rf(
        R=R[train_idx],
        Y=Y[train_idx],
        D=D[train_idx],
        S_e=S_e[train_idx],
        S_o=S_o[train_idx],
        R_pred=R[test_idx],
        ml_params=ml_params,
        n_jobs=n_jobs
    )

    theta_init = out['theta_init']
    predictions = out['predictions']
    observations = pd.DataFrame({
        'Y': Y[test_idx],
        'D': D[test_idx],
        'S_e': S_e[test_idx],
        'S_o': S_o[test_idx]
    })

    # Compute RSV estimator
    result = rsv_compute(
        observations=observations,
        predictions=predictions,
        theta_init=theta_init,
        eps=eps,
        eps_prob=eps_prob
    )

    # Compute standard errors if requested
    se_value = None
    denominator_se = None
    clusters = None

    if se:
        # Update clusters for test set
        se_params_test = se_params.copy() if se_params else {}
        if se_params_test.get('clusters') is not None:
            se_params_test['clusters'] = np.asarray(
                se_params_test['clusters']
            )[test_idx]

        boot_result = rsv_bootstrap(
            observations=observations,
            predictions=predictions,
            theta_init=theta_init,
            eps=eps,
            eps_prob=eps_prob,
            se_params=se_params_test,
            n_jobs=n_jobs
        )
        se_value = boot_result['se']
        denominator_se = boot_result['denominator_se']
        clusters = boot_result['clusters']

    return RSVResult(
        coef=result['coef'],
        se=se_value,
        n_exp=result['n_exp'],
        n_obs=result['n_obs'],
        n_both=result['n_both'],
        numerator=result['numerator'],
        denominator=result['denominator'],
        denominator_se=denominator_se,
        theta_init=result['theta_init'],
        weights=result['weights'],
        clusters=clusters
    )


def _rsv_fit_crossfit(
    R: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    S_e: np.ndarray,
    S_o: np.ndarray,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    ml_params: Optional[Dict[str, Any]] = None,
    se: bool = False,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> RSVResult:
    """
    Fit predictions with K-fold cross-fitting.

    Performs K-fold cross-fitting: splits data into K folds, fits predictions
    on K-1 folds, predicts on held-out fold, and repeats for all folds.
    """
    n = len(Y)
    nfolds = ml_params.get('nfolds', 5) if ml_params else 5
    seed = ml_params.get('seed', None) if ml_params else None

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Create folds
    fold_ids = np.random.permutation(np.tile(np.arange(nfolds), n // nfolds + 1)[:n])

    # Initialize results
    coefs = []
    fold_results = []

    # Cross-fitting loop
    for k in range(nfolds):
        # Training set: all folds except k
        train_idx = (fold_ids != k)
        # Test set: fold k
        test_idx = (fold_ids == k)

        # Fit on training folds, predict on test fold
        out_k = fit_predictions_rf(
            R=R[train_idx],
            Y=Y[train_idx],
            D=D[train_idx],
            S_e=S_e[train_idx],
            S_o=S_o[train_idx],
            R_pred=R[test_idx],
            ml_params=ml_params,
            n_jobs=n_jobs
        )

        theta_init_k = out_k['theta_init']
        predictions_k = out_k['predictions']
        observations_k = pd.DataFrame({
            'Y': Y[test_idx],
            'D': D[test_idx],
            'S_e': S_e[test_idx],
            'S_o': S_o[test_idx]
        })

        # Compute RSV estimator
        result_k = rsv_compute(
            observations=observations_k,
            predictions=predictions_k,
            theta_init=theta_init_k,
            eps=eps,
            eps_prob=eps_prob
        )

        # Compute standard errors if requested
        if se:
            se_params_k = se_params.copy() if se_params else {}
            if se_params_k.get('clusters') is not None:
                se_params_k['clusters'] = np.asarray(
                    se_params_k['clusters']
                )[test_idx]

            boot_result_k = rsv_bootstrap(
                observations=observations_k,
                predictions=predictions_k,
                theta_init=theta_init_k,
                eps=eps,
                eps_prob=eps_prob,
                se_params=se_params_k,
                n_jobs=n_jobs
            )
            result_k['se'] = boot_result_k['se']
            result_k['denominator_se'] = boot_result_k['denominator_se']

        fold_results.append(result_k)
        coefs.append(result_k['coef'])

    # Pooled Cross-Validated Estimator
    coef = np.nanmean(coefs)

    return RSVResult(
        coef=coef,
        se=None,  # SE not computed for cross-fit (requires special handling)
        n_exp=None,
        n_obs=None,
        n_both=None,
        fold_results=fold_results
    )
