"""
RSVResult class for storing and displaying RSV estimation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
from scipy import stats


@dataclass
class RSVResult:
    """
    Container for RSV treatment effect estimation results.

    Attributes
    ----------
    coef : float
        Treatment effect estimate.
    se : float, optional
        Standard error (if computed via bootstrap).
    n_exp : int, optional
        Sample size in experimental sample.
    n_obs : int, optional
        Sample size in observational sample.
    n_both : int, optional
        Sample size in both samples.
    method : str, optional
        Prediction fitting method used ('none', 'split', 'crossfit', 'predictions').
    numerator : float, optional
        Numerator of the treatment effect estimate.
    denominator : float, optional
        Denominator of the treatment effect estimate.
    denominator_se : float, optional
        Standard error of the denominator.
    theta_init : float, optional
        Initial treatment effect estimate.
    weights : np.ndarray, optional
        Efficient weights used in estimation.
    clusters : np.ndarray, optional
        Cluster identifiers used in bootstrap.
    fold_results : list, optional
        Results from each fold (for crossfit method).
    """
    coef: float
    se: Optional[float] = None
    n_exp: Optional[int] = None
    n_obs: Optional[int] = None
    n_both: Optional[int] = None
    method: Optional[str] = None
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    denominator_se: Optional[float] = None
    theta_init: Optional[float] = None
    weights: Optional[np.ndarray] = field(default=None, repr=False)
    clusters: Optional[np.ndarray] = field(default=None, repr=False)
    fold_results: Optional[List[Any]] = field(default=None, repr=False)

    def __str__(self) -> str:
        """Print method for RSV results."""
        lines = [
            "RSV Treatment Effect Estimate",
            "==============================",
            ""
        ]

        if self.se is not None:
            lines.append(f"Coefficient: {self.coef:.4f} (SE: {self.se:.4f})")
        else:
            lines.append(f"Coefficient: {self.coef:.4f}")

        if self.n_exp is not None and self.n_obs is not None:
            lines.append("")
            lines.append("Sample sizes:")
            lines.append(f"  Experimental: {self.n_exp}")
            lines.append(f"  Observational: {self.n_obs}")
            if self.n_both is not None and self.n_both > 0:
                lines.append(f"  Both: {self.n_both}")

        if self.method is not None:
            lines.append("")
            lines.append(f"Method: {self.method}")

        return "\n".join(lines)

    def summary(self) -> str:
        """
        Generate a detailed summary of the RSV estimation results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "RSV Treatment Effect Estimate",
            "==============================",
            ""
        ]

        # Coefficient table
        if self.se is not None:
            t_stat = self.coef / self.se
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

            lines.append("Coefficient:")
            lines.append(f"{'':>10}{'Estimate':>12}{'Std.Error':>12}{'t.value':>12}{'Pr(>|t|)':>12}")
            lines.append(f"{'D':>10}{self.coef:>12.4f}{self.se:>12.4f}{t_stat:>12.4f}{p_value:>12.4f}")
            lines.append("")
        else:
            lines.append(f"Coefficient: {self.coef:.4f}")
            lines.append("")

        # Sample information
        if self.n_exp is not None and self.n_obs is not None:
            lines.append("Sample sizes:")
            lines.append(f"  Experimental: {self.n_exp}")
            lines.append(f"  Observational: {self.n_obs}")
            if self.n_both is not None and self.n_both > 0:
                lines.append(f"  Both: {self.n_both}")
            lines.append("")

        # Method information
        if self.method is not None:
            lines.append(f"Prediction fitting method: {self.method}")

        return "\n".join(lines)

    def vcov(self) -> np.ndarray:
        """
        Extract variance-covariance matrix.

        Returns
        -------
        np.ndarray
            1x1 matrix containing the variance of the treatment effect.

        Raises
        ------
        ValueError
            If standard error is not available.
        """
        if self.se is None:
            raise ValueError("Standard error not available. Re-run with se=True.")
        return np.array([[self.se ** 2]])

    def confint(self, level: float = 0.95) -> tuple:
        """
        Compute confidence interval for the treatment effect.

        Parameters
        ----------
        level : float, default 0.95
            Confidence level.

        Returns
        -------
        tuple
            (lower bound, upper bound) of the confidence interval.

        Raises
        ------
        ValueError
            If standard error is not available.
        """
        if self.se is None:
            raise ValueError("Standard error not available. Re-run with se=True.")

        alpha = 1 - level
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = self.coef - self.se * z
        ci_upper = self.coef + self.se * z

        return (ci_lower, ci_upper)
