# rho-uncertainty

Correcting inference in spatial autoregressive (SAR) models for the estimation uncertainty in ρ.

The standard plug-in procedure treats ρ̂ as known, producing standard errors that are systematically too small. This package provides two corrections:

- **VIF** (instant, closed-form): captures 40-67% of the true correction
- **SANI** (seconds, numerical): captures ~98% of the true correction

## Installation
```bash
