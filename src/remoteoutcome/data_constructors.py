"""
Data Constructor Functions.

Functions to create experimental/observational sample splits from smartcard_data.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def create_data_real(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create Real Experimental/Observational Split Dataset.

    Transforms smartcard_data into the real experimental/observational split
    where treatment is observed in "Experimental: Treated (2010)",
    "Experimental: Untreated (2011)", and "Experimental: Untreated (2012)"
    waves and outcomes are observed in the "Experimental: Untreated (2011)"
    and "Observational (N/A)" waves.

    Parameters
    ----------
    data : pd.DataFrame
        smartcard_data data frame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - shrid2: SHRUG village identifier
        - spillover_20km: Spillover indicator
        - S: Sample indicator ("e", "o", "both")
        - D: Treatment indicator (NaN when not observed)
        - Ycons, Ylowinc, Ymidinc: Outcome variables (NaN when not observed)
        - clusters: Cluster identifier
        - luminosity_*: VIIRS nighttime lights features
        - satellite_*: MOSAIKS satellite features
    """
    result = data.copy()

    # Sample indicator based on original sample assignment
    sample_col = 'Sample (Smartcard)'
    conditions = [
        result[sample_col] == "Experimental: Treated (2010)",
        result[sample_col] == "Experimental: Untreated (2011)",
        result[sample_col] == "Experimental: Untreated (2012)",
        result[sample_col] == "Observational (N/A)"
    ]
    choices = ["e", "both", "e", "o"]
    result['S'] = np.select(conditions, choices, default=np.nan)

    # Treatment observed only when S = "e" or S = "both"
    result.loc[~result['S'].isin(['both', 'e']), 'D'] = np.nan

    # Outcomes observed only when S = "o" or S = "both"
    for col in ['Ycons', 'Ylowinc', 'Ymidinc']:
        if col in result.columns:
            result.loc[~result['S'].isin(['both', 'o']), col] = np.nan

    # Select relevant columns
    cols_to_keep = ['shrid2', 'spillover_20km', 'S', 'D', 'Ycons', 'Ylowinc',
                    'Ymidinc', 'clusters']
    cols_to_keep.extend([c for c in result.columns if c.startswith('luminosity_')])
    cols_to_keep.extend([c for c in result.columns if c.startswith('satellite_')])
    cols_to_keep = [c for c in cols_to_keep if c in result.columns]

    result = result[cols_to_keep]

    # Validation checks
    n_exp = (result['S'] == 'e').sum()
    n_obs = (result['S'] == 'o').sum()
    n_both = (result['S'] == 'both').sum()

    if n_exp == 0 or n_obs == 0:
        import warnings
        warnings.warn(
            "Sample split resulted in empty experimental or observational sample"
        )

    print(
        f"Created data_real with {len(result)} observations:\n"
        f"  Experimental only: {n_exp}\n"
        f"  Observational only: {n_obs}\n"
        f"  Both samples: {n_both}"
    )

    return result


def create_data_synth(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create Synthetic Experimental/Observational Split Dataset.

    Transforms the complete dataset (smartcard_data) into a synthetic
    experimental/observational split where top 50% of the clusters are
    assigned to observational sample.

    Parameters
    ----------
    data : pd.DataFrame
        smartcard_data data frame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - shrid2: SHRUG village identifier
        - spillover_20km: Spillover indicator
        - S: Sample indicator ("e", "o", "both")
        - D: Treatment indicator (NaN when not observed)
        - Ycons, Ylowinc, Ymidinc: Outcome variables (NaN when not observed)
        - clusters: Cluster identifier
        - luminosity_*: VIIRS nighttime lights features
        - satellite_*: MOSAIKS satellite features
    """
    result = data.copy()

    # Assign half the clusters to observational sample
    clusters = result['clusters'].unique()
    obs_clusters = clusters[:len(clusters) // 2]

    # Sample indicator
    sample_col = 'Sample (Smartcard)'

    # Experimental sample: original experimental waves
    S_e = result[sample_col].isin([
        "Experimental: Treated (2010)",
        "Experimental: Untreated (2011)",
        "Experimental: Untreated (2012)"
    ])

    # Observational sample: randomly selected clusters
    S_o = result['clusters'].isin(obs_clusters)

    # Combined sample indicator
    conditions = [
        S_e & S_o,
        S_e & ~S_o,
        ~S_e & S_o
    ]
    choices = ["both", "e", "o"]
    result['S'] = np.select(conditions, choices, default=np.nan)

    # Treatment observed only when S = "e" or S = "both"
    result.loc[~result['S'].isin(['both', 'e']), 'D'] = np.nan

    # Outcomes observed only when S = "o" or S = "both"
    for col in ['Ycons', 'Ylowinc', 'Ymidinc']:
        if col in result.columns:
            result.loc[~result['S'].isin(['both', 'o']), col] = np.nan

    # Select relevant columns
    cols_to_keep = ['shrid2', 'spillover_20km', 'S', 'D', 'Ycons', 'Ylowinc',
                    'Ymidinc', 'clusters']
    cols_to_keep.extend([c for c in result.columns if c.startswith('luminosity_')])
    cols_to_keep.extend([c for c in result.columns if c.startswith('satellite_')])
    cols_to_keep = [c for c in cols_to_keep if c in result.columns]

    result = result[cols_to_keep]

    # Validation checks
    n_exp = (result['S'] == 'e').sum()
    n_obs = (result['S'] == 'o').sum()
    n_both = (result['S'] == 'both').sum()

    if n_exp == 0 or n_obs == 0:
        import warnings
        warnings.warn(
            "Sample split resulted in empty experimental or observational sample"
        )

    print(
        f"Created data_synth with {len(result)} observations:\n"
        f"  Experimental only: {n_exp}\n"
        f"  Observational only: {n_obs}\n"
        f"  Both samples: {n_both}\n"
        f"  Clusters in observational sample: {len(obs_clusters)} of "
        f"{len(clusters)} ({len(obs_clusters)/len(clusters)*100:.1f}%)"
    )

    return result
