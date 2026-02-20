"""
Cluster bootstrap for RSV standard errors.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .helpers import rsv_compute


def rsv_bootstrap(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    theta_init: float,
    eps: Optional[float] = None,
    eps_prob: float = 0.01,
    se_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Compute cluster-bootstrap standard errors for the RSV estimator.

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
    se_params : dict, optional
        Bootstrap parameters:
        - B: Number of bootstrap replications (default 1000)
        - fix_seed: If True, use deterministic seeding (default False)
        - clusters: Cluster identifiers for clustered sampling
    n_jobs : int, default 1
        Number of parallel jobs for bootstrap.

    Returns
    -------
    dict
        Dictionary containing:
        - se: Standard error of the treatment effect
        - denominator_se: Standard error of the denominator
        - clusters: Cluster identifiers used
    """
    n = len(observations)

    if se_params is None:
        se_params = {}

    B = se_params.get('B', 1000)
    fix_seed = se_params.get('fix_seed', False)
    clusters = se_params.get('clusters', None)

    # If no clusters provided, use individual-level bootstrap
    if clusters is None:
        clusters = np.arange(n)
    else:
        clusters = np.asarray(clusters)

    if len(clusters) != len(observations):
        raise ValueError(
            "clusters must have the same length as observations"
        )

    unique_clusters = np.unique(clusters)

    def run_one(b: int) -> Dict[str, float]:
        """Run a single bootstrap replication."""
        if fix_seed:
            np.random.seed(b)

        # Cluster bootstrap: sample clusters with replacement
        clusters_b = np.random.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True
        )

        # Get indices for selected clusters
        # R uses: which(clusters %in% clusters_b)
        # This is set membership - even if a cluster is sampled multiple times,
        # its observations only appear once in the bootstrap sample
        index_b = np.isin(clusters, clusters_b)

        observations_b = observations.loc[index_b].reset_index(drop=True)
        predictions_b = predictions.loc[index_b].reset_index(drop=True)

        try:
            out_b = rsv_compute(
                observations=observations_b,
                predictions=predictions_b,
                theta_init=theta_init,
                eps=eps,
                eps_prob=eps_prob
            )
            return {'coef': out_b['coef'], 'denominator': out_b['denominator']}
        except Exception:
            return {'coef': np.nan, 'denominator': np.nan}

    # Run bootstrap replications
    if n_jobs == 1:
        results = [run_one(b) for b in range(1, B + 1)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(b) for b in range(1, B + 1)
        )

    # Extract coefficients and denominators
    coefs = np.array([r['coef'] for r in results])
    denominators = np.array([r['denominator'] for r in results])

    # Compute SE
    se = np.nanstd(coefs, ddof=1)
    denominator_se = np.nanstd(denominators, ddof=1)

    return {
        'se': se,
        'denominator_se': denominator_se,
        'clusters': clusters
    }
