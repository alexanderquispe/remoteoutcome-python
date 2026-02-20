# remoteoutcome-python

Python implementation of the RSV Treatment Effect Estimator from [Rambachan, Singh, and Viviano (2025)](https://arxiv.org/abs/2501.02377).

This package estimates treatment effects using remotely sensed variables (RSVs) by combining experimental and observational data sources.

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from remoteoutcome import rsv_estimate

# With pre-computed predictions
result = rsv_estimate(
    Y=data['Y'],
    D=data['D'],
    S_e=data['S_e'],
    S_o=data['S_o'],
    pred_Y=data['pred_Y'],
    pred_D=data['pred_D'],
    pred_S_e=data['pred_S_e'],
    pred_S_o=data['pred_S_o'],
    method='predictions',
    theta_init=theta_init,
    eps=1e-2,
    se=True,
    se_params={'B': 1000, 'fix_seed': True, 'clusters': data['clusters']}
)

print(result)
# RSV Treatment Effect Estimate
# ==============================
#
# Coefficient: -0.0135 (SE: 0.0123)
#
# Sample sizes:
#   Experimental: 6055
#   Observational: 5186
#   Both: 2929
#
# Method: predictions

# Get confidence interval
ci = result.confint(level=0.90)
print(f"90% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

## Methods

The package supports four estimation methods:

| Method | Description |
|--------|-------------|
| `predictions` | Use pre-computed predictions provided by user |
| `none` | Fit predictions on all data, estimate on all data |
| `split` | Train/test split for predictions |
| `crossfit` | K-fold cross-fitting for predictions |

## Numerical Equivalence with R Package

This Python implementation produces results that match the original R package:

| Method | Coefficient | SE | n_exp | n_obs | n_both |
|--------|-------------|-----|-------|-------|--------|
| crossfit | -0.1082 | - | - | - | - |
| split | -0.1086 | 0.0959 | 3032 | 2575 | 1451 |
| none | -0.0135 | 0.0120 | 6055 | 5186 | 2929 |
| predictions | -0.0135 | 0.0120 | 6055 | 5186 | 2929 |

## API Reference

### Main Function

```python
rsv_estimate(
    Y,                  # Outcome variable (binary)
    D,                  # Treatment indicator (binary)
    S_e,                # Experimental sample indicator
    S_o,                # Observational sample indicator
    R=None,             # Remotely sensed variables (for fitting)
    pred_Y=None,        # Predicted P(Y=1|R, S_o=1)
    pred_D=None,        # Predicted P(D=1|R, S_e=1)
    pred_S_e=None,      # Predicted P(S_e=1|R)
    pred_S_o=None,      # Predicted P(S_o=1|R)
    theta_init=None,    # Initial treatment effect estimate
    eps=None,           # Numerical stability constant
    eps_prob=0.01,      # Quantile for eps if not provided
    method='crossfit',  # Estimation method
    ml_params=None,     # Random forest parameters
    se=True,            # Compute bootstrap SE
    se_params=None,     # Bootstrap parameters
    n_jobs=1            # Parallel jobs
)
```

### RSVResult Object

The `rsv_estimate` function returns an `RSVResult` object with:

- `coef`: Treatment effect estimate
- `se`: Standard error (if computed)
- `n_exp`: Experimental sample size
- `n_obs`: Observational sample size
- `n_both`: Sample size in both
- `method`: Method used
- `confint(level)`: Confidence interval method
- `vcov()`: Variance-covariance matrix

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.0.0

## Data Files

The package includes `pred_real_Ycons.parquet` for testing with pre-computed predictions.

For full pipeline testing with RF fitting, you need the smartcard data files. Convert them from the R package:

```python
import pyreadr
import pandas as pd

# Load from R package
r_data_path = "path/to/remoteoutcome/data/"
df1 = pyreadr.read_r(f"{r_data_path}/smartcard_data_p1.rda")['smartcard_data_p1']
df2 = pyreadr.read_r(f"{r_data_path}/smartcard_data_p2.rda")['smartcard_data_p2']

# Save as parquet
df1.to_parquet("data/smartcard_data_p1.parquet")
df2.to_parquet("data/smartcard_data_p2.parquet")
```

## Testing

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{rambachan2025remotely,
  title={Remotely Sensed Treatment Effects},
  author={Rambachan, Ashesh and Singh, Rahul and Viviano, Davide},
  journal={arXiv preprint arXiv:2501.02377},
  year={2025}
}
```

## License

MIT
