# rho-uncertainty

Correcting inference in spatial autoregressive (SAR) models for the estimation uncertainty in ρ.

The standard plug-in procedure treats ρ̂ as known, producing standard errors that are systematically too small. This package provides two corrections:

- **VIF** (instant, closed-form): captures 40-67% of the true correction
- **SANI** (seconds, numerical): captures ~98% of the true correction

## Installation
```bash
pip install rho-uncertainty
```

Or download `rho_uncertainty.py` directly and place it in your working directory.

## Usage with PySAL/spreg
```python
from spreg import ML_Lag
model = ML_Lag(y, X, w=w)

from rho_uncertainty import correct
correct(model)           # prints corrected regression table
correct(model, 'sani')   # higher accuracy (~10s)
```

## Usage with raw matrices
```python
from rho_uncertainty import rho_test

# W: spatial weights (n x n), X: design matrix (n x p, with intercept)
# y: dependent variable (n,), rho_hat: estimated rho
result = rho_test(W, X, y, rho_hat=0.3)
```

## Quick VIF-only correction
```python
from rho_uncertainty import vif_correction

factor = vif_correction(W, X, rho_hat)
se_corrected = se_naive * factor
```

## Example output
```
======================================================================
  SAR Regression with Rho-Uncertainty Correction
======================================================================
  Dependent variable: crime
  n = 49,  p = 3,  rho_hat = 0.4159
  Method: VIF

      Variable      Coef  SE(naive)    SE(VIF)    t(VIF)   p-value
  -----------------------------------------------------------------
        income    0.4795     0.1587     0.1597    3.0020    0.0043  ***
         hoval   -0.0760     0.1583     0.1589   -0.4784    0.6346
  -----------------------------------------------------------------
  Average variance inflation: +1.07%
======================================================================
```

## How it works

The VIF captures only 40-67% of the true variance correction due to a structural limitation called *phantom substitution*. SANI bypasses this through Rao-Blackwellized numerical integration.

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7

## License

MIT
