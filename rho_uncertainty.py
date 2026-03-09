"""
rho_uncertainty: Correcting SAR inference for rho-estimation uncertainty
========================================================================

The standard two-step procedure in spatial autoregressive (SAR) models
— estimate rho by profile MLE, then treat it as known — understates
standard errors for beta. This package corrects it.

Install:
    pip install rho-uncertainty

Usage with PySAL/spreg:
    from spreg import ML_Lag
    model = ML_Lag(y, X, w=w)

    from rho_uncertainty import correct
    correct(model)           # prints corrected table
    correct(model, 'sani')   # higher accuracy (takes ~10s)

Usage with raw matrices:
    from rho_uncertainty import rho_test
    result = rho_test(W, X, y, rho_hat=0.3)

Reference:
    [Author] (2026). "How Much Does Estimating rho Distort Inference
    on beta?" Journal of Econometrics, forthcoming.

License: MIT
"""

__version__ = "0.1.0"
__all__ = ['correct', 'rho_test', 'rho_test_all', 'vif_correction']

import numpy as np
from scipy import linalg, optimize
from scipy.special import roots_hermite, roots_genlaguerre
from scipy.special import gamma as gamma_fn
import warnings


# ======================================================================
#  TOP-LEVEL API: correct()
# ======================================================================

def correct(model, method='vif', sani_draws=300, verbose=True):
    """
    Correct a fitted SAR model for rho-uncertainty. Works with:
      - PySAL spreg.ML_Lag
      - PySAL spreg.ML_Error (partial support)
      - Any object with .rho, .x, .y, .w attributes

    Parameters
    ----------
    model : spreg result object
        A fitted SAR model from spreg.ML_Lag or similar.
    method : str
        'vif' (instant, default) or 'sani' (accurate, ~10s) or 'both'.
    sani_draws : int
        Number of spherical draws for SANI (default 300).
    verbose : bool
        Print corrected regression table (default True).

    Returns
    -------
    List of RhoTestResult, one per non-intercept coefficient.

    Example
    -------
    >>> from spreg import ML_Lag
    >>> from libpysal.weights import Queen
    >>> w = Queen.from_dataframe(gdf)
    >>> model = ML_Lag(gdf[['crime']].values, gdf[['income','hoval']].values, w=w)
    >>> from rho_uncertainty import correct
    >>> correct(model)
    """
    W, X, y, rho_hat, var_names = _extract_from_model(model)
    results = rho_test_all(W, X, y, rho_hat=rho_hat, method=method,
                           sani_draws=sani_draws, verbose=False)

    # Attach variable names if available
    for i, r in enumerate(results):
        if var_names and i + 1 < len(var_names):
            r.var_name = var_names[i + 1]
        else:
            r.var_name = f"x{r.test_idx}"

    if verbose:
        _print_corrected_table(results, model, method)

    return results


def _extract_from_model(model):
    """Extract W, X, y, rho_hat from various model objects."""
    var_names = None

    # --- Try spreg.ML_Lag ---
    if hasattr(model, 'rho') and hasattr(model, 'x') and hasattr(model, 'y'):
        rho_hat = float(model.rho)
        X = np.asarray(model.x)
        y = np.asarray(model.y).ravel()

        # Get W matrix from libpysal weights object
        if hasattr(model, 'w'):
            w_obj = model.w
            if hasattr(w_obj, 'full'):
                W = np.asarray(w_obj.full()[0])
            elif hasattr(w_obj, 'sparse'):
                W = np.asarray(w_obj.sparse.toarray())
            else:
                raise ValueError(
                    "Cannot convert spatial weights to dense matrix. "
                    "Pass W matrix directly using rho_test()."
                )
        else:
            raise ValueError(
                "Model has no .w attribute. "
                "Pass W matrix directly using rho_test(W, X, y, rho_hat)."
            )

        # Try to get variable names
        if hasattr(model, 'name_x'):
            var_names = list(model.name_x)
        elif hasattr(model, 'name_x_e'):
            var_names = list(model.name_x_e)

        return W, X, y, rho_hat, var_names

    # --- Try dict-like input ---
    if isinstance(model, dict):
        required = ['W', 'X', 'y', 'rho_hat']
        for k in required:
            if k not in model:
                raise ValueError(f"Dict input missing key '{k}'. "
                                 f"Required: {required}")
        return (np.asarray(model['W']), np.asarray(model['X']),
                np.asarray(model['y']).ravel(), float(model['rho_hat']),
                model.get('var_names', None))

    raise TypeError(
        f"Cannot extract SAR components from {type(model).__name__}. "
        "Supported: spreg.ML_Lag, or dict with keys "
        "{'W', 'X', 'y', 'rho_hat'}. "
        "Or use rho_test(W, X, y, rho_hat) directly."
    )


def _print_corrected_table(results, model, method):
    """Print a publication-ready corrected regression table."""
    r0 = results[0]
    has_sani = r0.se_sani is not None

    print("")
    print("=" * 70)
    print("  SAR Regression with Rho-Uncertainty Correction")
    print("=" * 70)
    print(f"  Dependent variable: {_get_depvar_name(model)}")
    print(f"  n = {r0.n},  p = {r0.p},  rho_hat = {r0.rho_hat:.4f}")
    print(f"  Method: {method.upper()}")
    print("")

    # Header
    if has_sani:
        print(f"  {'Variable':>12s} {'Coef':>9s} {'SE(naive)':>10s}"
              f" {'SE(corr)':>10s} {'t(corr)':>9s} {'p-value':>9s}"
              f" {'':>4s}")
    else:
        print(f"  {'Variable':>12s} {'Coef':>9s} {'SE(naive)':>10s}"
              f" {'SE(VIF)':>10s} {'t(VIF)':>9s} {'p-value':>9s}"
              f" {'':>4s}")
    print("  " + "-" * 65)

    for r in results:
        t_corr = r.t_sani if has_sani else r.t_vif
        se_corr = r.se_sani if has_sani else r.se_vif
        p_corr = _pval(t_corr, r.nu)
        sig = "***" if p_corr < 0.01 else (
              "**" if p_corr < 0.05 else (
              "*" if p_corr < 0.10 else ""))

        name = getattr(r, 'var_name', f'x{r.test_idx}')
        print(f"  {name:>12s} {r.beta_hat:9.4f} {r.se_naive:10.4f}"
              f" {se_corr:10.4f} {t_corr:9.4f} {p_corr:9.4f} {sig:>4s}")

        # Flag if significance flipped
        p_naive = _pval(r.t_naive, r.nu)
        if p_naive < 0.05 and p_corr >= 0.05:
            print(f"  {'':>12s} ** significance at 5% lost after correction **")

    print("  " + "-" * 65)

    # SE inflation summary
    factors = [r.sani_factor if has_sani else r.vif_factor for r in results]
    avg_pct = np.mean([(f**2 - 1) * 100 for f in factors])
    print(f"  Average variance inflation: +{avg_pct:.2f}%")
    if not has_sani:
        cap = results[0].diagnostics.get('capture_est', 50)
        print(f"  VIF captures ~{cap:.0f}% of true correction"
              " (use method='sani' for full correction)")
    print("")
    print(f"  rho_hat = {r0.rho_hat:.4f}")
    print("=" * 70)


def _get_depvar_name(model):
    """Try to get dependent variable name from model."""
    for attr in ('name_y', 'name_ds'):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, str):
                return val
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return str(val[0])
    return "y"


# ======================================================================
#  CORE API: rho_test() and rho_test_all()
# ======================================================================

def rho_test(W, X, y, rho_hat=None, method='both', test_idx=1,
             sani_draws=300, sani_seeds=3, verbose=True):
    """
    Test and correct for rho-uncertainty using raw matrices.

    Parameters
    ----------
    W : array (n, n)
        Row-normalized spatial weights matrix.
    X : array (n, p)
        Design matrix (should include intercept).
    y : array (n,)
        Dependent variable.
    rho_hat : float or None
        Estimated rho. If None, estimated by profile MLE.
    method : str
        'vif' (instant), 'sani' (~10s), or 'both' (default).
    test_idx : int
        Column index of beta to test (default 1 = first non-intercept).
    sani_draws : int
        Spherical draws for SANI (default 300).
    sani_seeds : int
        Independent SANI runs to average (default 3).
    verbose : bool
        Print summary (default True).

    Returns
    -------
    RhoTestResult with .se_naive, .se_vif, .se_sani, .t_naive, etc.
    """
    # Input validation
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = W.shape[0]
    p = X.shape[1]
    nu = n - p

    if W.shape != (n, n):
        raise ValueError(f"W must be square, got {W.shape}")
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows, W has {n}")
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} elements, W has {n}")
    if test_idx < 0 or test_idx >= p:
        raise ValueError(f"test_idx={test_idx} out of range [0, {p-1}]")
    if nu <= 2:
        raise ValueError(f"Too few degrees of freedom: nu={nu}")
    if method not in ('vif', 'sani', 'both'):
        raise ValueError(f"method must be 'vif', 'sani', or 'both'")

    # Estimate rho if needed
    engine = _SAREngine(W, X, rho_hat if rho_hat else 0.0)
    if rho_hat is None:
        rho_hat = engine.find_mle(y)
        engine = _SAREngine(W, X, rho_hat)

    # Plug-in inference
    A_hat = np.eye(n) - rho_hat * W
    Ay = A_hat @ y
    beta_hat = engine.XtX_inv @ (X.T @ Ay)
    e = Ay - X @ beta_hat
    sigma2 = (e @ e) / nu
    c_vec = np.zeros(p); c_vec[test_idx] = 1.0
    d = c_vec @ engine.XtX_inv @ c_vec
    se_naive = np.sqrt(sigma2 * d)
    t_naive = beta_hat[test_idx] / se_naive

    V0 = nu / (nu - 2)

    # VIF
    vif_val, qG0q, diagnostics = _compute_vif(engine, rho_hat, test_idx)
    vif_factor = np.sqrt(max(vif_val / V0, 1.0))
    se_vif = se_naive * vif_factor
    t_vif = beta_hat[test_idx] / se_vif

    # SANI
    se_sani, t_sani, sani_factor, sani_val = None, None, None, None
    if method in ('sani', 'both'):
        sani_val = _compute_sani(engine, y, rho_hat, test_idx,
                                  ni=sani_draws, n_avg=sani_seeds)
        sani_factor = np.sqrt(max(sani_val / V0, 1.0))
        se_sani = se_naive * sani_factor
        t_sani = beta_hat[test_idx] / se_sani

    # Update capture estimate
    if sani_val is not None and sani_val > V0 and vif_val > V0:
        actual = (vif_val - V0) / (sani_val - V0) * 100
        diagnostics['capture_est'] = min(max(actual, 10), 99)

    result = RhoTestResult(
        beta_hat=beta_hat[test_idx],
        se_naive=se_naive, se_vif=se_vif, se_sani=se_sani,
        t_naive=t_naive, t_vif=t_vif, t_sani=t_sani,
        vif_factor=vif_factor, sani_factor=sani_factor,
        rho_hat=rho_hat, V0=V0, V_vif=vif_val, V_sani=sani_val,
        qG0q=qG0q, n=n, p=p, nu=nu, test_idx=test_idx,
        diagnostics=diagnostics,
    )

    if verbose:
        print(result)

    return result


def rho_test_all(W, X, y, rho_hat=None, method='both',
                 sani_draws=300, sani_seeds=3, verbose=True):
    """
    Correct all non-intercept coefficients at once.

    Returns list of RhoTestResult, one per coefficient (excl. intercept).
    """
    results = []
    p = X.shape[1]
    for j in range(1, p):
        r = rho_test(W, X, y, rho_hat=rho_hat, method=method,
                     test_idx=j, sani_draws=sani_draws,
                     sani_seeds=sani_seeds, verbose=False)
        results.append(r)

    if verbose:
        _print_multi(results)

    return results


def vif_correction(W, X, rho_hat, test_idx=1):
    """
    Quick VIF multiplier for standard errors.

    Usage:
        factor = vif_correction(W, X, rho_hat)
        se_corrected = se_naive * factor
    """
    engine = _SAREngine(W, X, rho_hat)
    vif_val, _, _ = _compute_vif(engine, rho_hat, test_idx)
    V0 = engine.nu / (engine.nu - 2)
    return np.sqrt(max(vif_val / V0, 1.0))


# ======================================================================
#  RESULT CLASS
# ======================================================================

class RhoTestResult:
    """Container for rho-uncertainty test results."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        lines = [
            "",
            "=" * 64,
            "  Rho-Uncertainty Correction for SAR Model",
            "=" * 64,
            f"  n = {self.n},  p = {self.p},  nu = {self.nu}",
            f"  rho_hat = {self.rho_hat:.4f}",
            f"  Testing: beta[{self.test_idx}] = 0",
            "",
        ]
        has_sani = self.se_sani is not None
        hdr = f"  {'':20s} {'Naive':>10s} {'VIF':>10s}"
        if has_sani: hdr += f" {'SANI':>10s}"
        lines.append(hdr)
        div = f"  {'-'*20} {'-'*10} {'-'*10}"
        if has_sani: div += f" {'-'*10}"
        lines.append(div)

        lines.append(f"  {'beta_hat':20s} {self.beta_hat:10.4f}")

        se_line = f"  {'Std. Error':20s} {self.se_naive:10.4f} {self.se_vif:10.4f}"
        if has_sani: se_line += f" {self.se_sani:10.4f}"
        lines.append(se_line)

        t_line = f"  {'t-statistic':20s} {self.t_naive:10.4f} {self.t_vif:10.4f}"
        if has_sani: t_line += f" {self.t_sani:10.4f}"
        lines.append(t_line)

        p_line = (f"  {'p-value':20s} {_pval(self.t_naive, self.nu):10.4f}"
                  f" {_pval(self.t_vif, self.nu):10.4f}")
        if has_sani:
            p_line += f" {_pval(self.t_sani, self.nu):10.4f}"
        lines.append(p_line)

        lines.append("")
        lines.append(f"  SE multiplier (VIF):  {self.vif_factor:.4f}"
                     f"  (variance +{(self.vif_factor**2-1)*100:.2f}%)")
        if has_sani:
            lines.append(f"  SE multiplier (SANI): {self.sani_factor:.4f}"
                         f"  (variance +{(self.sani_factor**2-1)*100:.2f}%)")

        cap = self.diagnostics.get('capture_est', 50)
        lines.append("")
        lines.append(f"  VIF captures ~{cap:.0f}% of true correction"
                     " (typical range: 40-67%)")

        # Significance flip detection
        t_best = self.t_sani if has_sani else self.t_vif
        for level, name in [(1.96, '5%'), (2.576, '1%'), (1.645, '10%')]:
            if abs(self.t_naive) > level and abs(t_best) <= level:
                lines.append("")
                lines.append(f"  *** WARNING: Significance at {name} level"
                             " may be spurious after correction ***")
                break

        lines.append("")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_latex(self):
        """LaTeX row for regression tables."""
        t_corr = self.t_sani if self.t_sani is not None else self.t_vif
        se_corr = self.se_sani if self.se_sani is not None else self.se_vif
        p_corr = _pval(t_corr, self.nu)
        sig = "^{***}" if p_corr < 0.01 else (
              "^{**}" if p_corr < 0.05 else (
              "^{*}" if p_corr < 0.10 else ""))
        name = getattr(self, 'var_name', f'x_{self.test_idx}')
        return (f"{name} & {self.beta_hat:.4f}{sig} & "
                f"({self.se_naive:.4f}) & ({se_corr:.4f}) \\\\")

    def summary_dict(self):
        """Return results as a flat dict (easy pandas conversion)."""
        return {
            'test_idx': self.test_idx,
            'var_name': getattr(self, 'var_name', None),
            'beta_hat': self.beta_hat,
            'se_naive': self.se_naive,
            'se_vif': self.se_vif,
            'se_sani': self.se_sani,
            't_naive': self.t_naive,
            't_vif': self.t_vif,
            't_sani': self.t_sani,
            'p_naive': _pval(self.t_naive, self.nu),
            'p_vif': _pval(self.t_vif, self.nu),
            'p_sani': _pval(self.t_sani, self.nu) if self.t_sani else None,
            'rho_hat': self.rho_hat,
            'vif_factor': self.vif_factor,
            'sani_factor': self.sani_factor,
        }


# ======================================================================
#  PRINT HELPERS
# ======================================================================

def _print_multi(results):
    """Pretty-print multi-coefficient results."""
    r0 = results[0]
    has_sani = r0.se_sani is not None
    print("")
    print("=" * 72)
    print("  Rho-Uncertainty Correction: All Coefficients")
    print("=" * 72)
    print(f"  n = {r0.n},  p = {r0.p},  rho_hat = {r0.rho_hat:.4f}")
    print("")

    hdr = f"  {'Coef':>5s} {'beta':>9s} {'SE(naive)':>9s} {'SE(corr)':>9s}"
    hdr += f" {'t(naive)':>9s} {'t(corr)':>9s} {'p(corr)':>8s} {'':>4s}"
    print(hdr)
    print("  " + "-" * 62)

    for r in results:
        t_corr = r.t_sani if has_sani else r.t_vif
        se_corr = r.se_sani if has_sani else r.se_vif
        p_corr = _pval(t_corr, r.nu)
        sig = "***" if p_corr < 0.01 else (
              "**" if p_corr < 0.05 else (
              "*" if p_corr < 0.10 else ""))
        name = getattr(r, 'var_name', str(r.test_idx))
        print(f"  {name:>5s} {r.beta_hat:9.4f} {r.se_naive:9.4f}"
              f" {se_corr:9.4f} {r.t_naive:9.4f} {t_corr:9.4f}"
              f" {p_corr:8.4f} {sig:>4s}")

    print("")
    print("=" * 72)


# ======================================================================
#  INTERNAL ENGINE
# ======================================================================

class _SAREngine:
    """Lightweight SAR computation engine."""

    def __init__(self, W, X, rho0):
        self.W, self.X, self.rho0 = W, X, rho0
        self.n, self.p = X.shape
        self.nu = self.n - self.p
        self.XtX_inv = linalg.inv(X.T @ X)
        PX = X @ self.XtX_inv @ X.T
        self.MX = np.eye(self.n) - PX
        self.w_eigs = linalg.eigvals(W)
        re = np.real(self.w_eigs)
        pos = re[re > 1e-10]; neg = re[re < -1e-10]
        self.rho_max = 1.0 / max(pos) if len(pos) > 0 else 5.0
        self.rho_min = 1.0 / min(neg) if len(neg) > 0 else -5.0

    def log_det_A(self, rho):
        return np.sum(np.log(np.abs(1.0 - rho * self.w_eigs))).real

    def profile_loglik(self, rho, y):
        Ay = y - rho * (self.W @ y)
        e = self.MX @ Ay; sse = e @ e
        if sse <= 0: return -1e20
        return self.log_det_A(rho) - 0.5 * self.n * np.log(sse)

    def find_mle(self, y):
        lo, hi = self.rho_min + 0.01, self.rho_max - 0.01
        return optimize.minimize_scalar(
            lambda r: -self.profile_loglik(r, y),
            bounds=(lo, hi), method='bounded',
            options={'xatol': 1e-10}).x

    def z_stat_idx(self, rho, y, idx):
        Ay = y - rho * (self.W @ y)
        b = self.XtX_inv @ (self.X.T @ Ay)
        e = Ay - self.X @ b
        s2 = (e @ e) / self.nu
        if s2 <= 0: return 0.0
        return b[idx] / np.sqrt(s2 * self.XtX_inv[idx, idx])

    def t_squared(self, rho, y, idx):
        return self.z_stat_idx(rho, y, idx) ** 2


# ======================================================================
#  VIF COMPUTATION
# ======================================================================

def _compute_vif(engine, rho_hat, test_idx):
    """Compute VIF using trace formulas."""
    n, nu = engine.n, engine.nu
    W, X, MX = engine.W, engine.X, engine.MX
    V0 = nu / (nu - 2)

    A0inv = linalg.inv(np.eye(n) - rho_hat * W)
    G0 = W @ A0inv

    c_vec = np.zeros(engine.p); c_vec[test_idx] = 1.0
    d = c_vec @ engine.XtX_inv @ c_vec
    q = X @ (engine.XtX_inv @ c_vec) / np.sqrt(d)

    Am = MX @ G0 @ MX
    qGq = q @ G0 @ q

    EA2 = (nu / (nu - 2)) * (q @ G0 @ G0.T @ q) \
        - (2 / (nu - 2)) * (q @ G0 @ MX @ G0.T @ q)
    trMG = np.trace(MX @ G0)
    E2AB = -2.0 * qGq * trMG / (nu - 2)
    trAm = np.trace(Am)
    trAAt = np.trace(Am @ Am.T)
    trA2 = np.trace(Am @ Am)
    EB2 = (trAm**2 + trAAt + trA2) / ((nu - 2) * (nu + 2))
    EZp2 = EA2 + E2AB + EB2

    trG2 = np.trace(G0 @ G0)
    trMGGt = np.trace(MX @ G0 @ G0.T)
    IP = trG2 + (n / nu) * trMGGt \
        - 2 * n / (nu * (nu + 2)) * (trAm**2 + trAAt + trA2)

    V_vif = V0 + EZp2 / IP

    diagnostics = {
        'EZp2': EZp2, 'IP': IP, 'qG0q': qGq,
        'EA2': EA2, 'E2AB': E2AB, 'EB2': EB2,
        'trMG': trMG, 'V_vif': V_vif, 'V0': V0,
        'capture_est': 50.0,
    }

    return V_vif, qGq, diagnostics


# ======================================================================
#  SANI COMPUTATION
# ======================================================================

def _robust_s_quad(nu, n_nodes):
    """Quadrature for s^2 ~ chi^2_nu / nu."""
    if nu < 150:
        alpha = nu / 2.0 - 1.0
        try:
            u, w = roots_genlaguerre(n_nodes, alpha)
            s2 = 2.0 * u / nu
            sw = w / gamma_fn(alpha + 1.0)
            if not (np.any(np.isnan(sw)) or np.any(np.isinf(sw))):
                return np.sqrt(np.maximum(s2, 0)), sw
        except Exception:
            pass
    xh, wh = roots_hermite(n_nodes)
    s2 = 1.0 + np.sqrt(2.0 / nu) * np.sqrt(2.0) * xh
    s2 = np.maximum(s2, 1e-12)
    return np.sqrt(s2), wh / np.sqrt(np.pi)


def _compute_sani(engine, y, rho_hat, test_idx, nt=10, ns=10,
                   ni=300, n_avg=3):
    """SANI numerical integration for V(X)."""
    n, nu = engine.n, engine.nu
    W, X, MX = engine.W, engine.X, engine.MX

    A0inv = linalg.inv(np.eye(n) - rho_hat * W)

    # Build orthonormal bases
    c_vec = np.zeros(engine.p); c_vec[test_idx] = 1.0
    d = c_vec @ engine.XtX_inv @ c_vec
    q = X @ (engine.XtX_inv @ c_vec) / np.sqrt(d)

    Q_X, _ = np.linalg.qr(X, mode='reduced')
    c_v = Q_X.T @ q
    Proj = np.eye(engine.p) - np.outer(c_v, c_v)
    U, S, _ = np.linalg.svd(Proj, full_matrices=False)
    mask = S > 1e-10
    Q_xi = Q_X @ U[:, mask]
    dim_xi = Q_xi.shape[1]

    eig_M, V_M = np.linalg.eigh(MX)
    Q_theta = V_M[:, eig_M > 0.5]

    xh, wh = roots_hermite(nt)
    t_nodes = np.sqrt(2.0) * xh
    t_weights = wh / np.sqrt(np.pi)
    s_nodes, s_weights = _robust_s_quad(nu, ns)

    def _reconstruct(t, xi, s, theta):
        eps = t * q
        if dim_xi > 0 and xi is not None:
            eps = eps + Q_xi @ xi
        eps = eps + s * np.sqrt(nu) * (Q_theta @ theta)
        return eps

    vals = []
    for seed_offset in range(n_avg):
        rng = np.random.default_rng(2025 + seed_offset)
        V = 0.0
        for it in range(nt):
            for js in range(ns):
                t, wt = t_nodes[it], t_weights[it]
                s, ws = s_nodes[js], s_weights[js]
                z2_sum = 0.0
                for _ in range(ni):
                    xi = rng.standard_normal(dim_xi) if dim_xi > 0 else None
                    z = rng.standard_normal(nu)
                    th = z / np.linalg.norm(z)
                    eps = _reconstruct(t, xi, s, th)
                    yy = A0inv @ eps
                    rhat = engine.find_mle(yy)
                    z2f = engine.t_squared(rhat, yy, test_idx)
                    xi_a = -xi if xi is not None else None
                    eps_a = _reconstruct(t, xi_a, s, -th)
                    yy_a = A0inv @ eps_a
                    rhat_a = engine.find_mle(yy_a)
                    z2a = engine.t_squared(rhat_a, yy_a, test_idx)
                    z2_sum += 0.5 * (z2f + z2a)
                V += wt * ws * (z2_sum / ni)
        vals.append(V)

    return np.mean(vals)


# ======================================================================
#  UTILITIES
# ======================================================================

def _pval(t_stat, nu):
    """Two-sided p-value from t-distribution."""
    from scipy.stats import t as t_dist
    return float(2 * t_dist.sf(abs(t_stat), df=nu))


def make_rook_W(m):
    """Row-normalized Rook contiguity on m x m grid."""
    n = m * m; W = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            idx = i * m + j; nbs = []
            if i > 0: nbs.append((i-1)*m+j)
            if i < m-1: nbs.append((i+1)*m+j)
            if j > 0: nbs.append(i*m+(j-1))
            if j < m-1: nbs.append(i*m+(j+1))
            for nb in nbs: W[idx, nb] = 1.0 / len(nbs)
    return W


# ======================================================================
#  DEMO
# ======================================================================

def demo():
    """Quick demo with simulated data."""
    print("=" * 70)
    print("  rho_uncertainty v{} — Demo".format(__version__))
    print("=" * 70)

    np.random.seed(42)
    m = 7; n = m * m
    W = make_rook_W(m)
    rho0 = 0.4
    X = np.column_stack([np.ones(n), np.random.randn(n),
                         np.random.randn(n)])
    beta_true = np.array([1.0, 0.5, -0.3])
    eps = np.random.randn(n)
    A0inv = np.linalg.inv(np.eye(n) - rho0 * W)
    y = A0inv @ (X @ beta_true + eps)

    print(f"\n  Simulated SAR: n={n}, p={X.shape[1]}, rho_true={rho0}")
    print(f"  beta_true = {beta_true}\n")

    # Demo 1: As if user has raw matrices
    print("-" * 70)
    print("  Demo 1: Using rho_test() with raw matrices")
    print("-" * 70)
    r1 = rho_test(W, X, y, method='both', test_idx=1, sani_draws=200)

    # Demo 2: As if user has a spreg-like result (simulate it)
    print("\n" + "-" * 70)
    print("  Demo 2: Using correct() with a model-like object")
    print("-" * 70)

    class FakeSpregResult:
        """Mimics spreg.ML_Lag output for testing."""
        def __init__(self):
            self.rho = r1.rho_hat
            self.x = X
            self.y = y.reshape(-1, 1)
            self.name_x = ['CONSTANT', 'income', 'hoval']
            self.name_y = 'crime'
            # Wrap W in a fake weights object
            self.w = type('W', (), {'full': lambda self: (W, None)})()

    fake_model = FakeSpregResult()
    results = correct(fake_model, method='vif')

    # Demo 3: Marginal case
    print("\n" + "-" * 70)
    print("  Demo 3: Marginal significance (correction may flip result)")
    print("-" * 70)
    np.random.seed(2026)
    X2 = np.column_stack([np.ones(n), np.random.randn(n)])
    beta2 = np.array([0.0, 0.28])
    eps2 = np.random.randn(n)
    y2 = A0inv @ (X2 @ beta2 + eps2)
    r3 = rho_test(W, X2, y2, method='both', test_idx=1, sani_draws=200)

    # Demo 4: LaTeX output
    print("\n" + "-" * 70)
    print("  Demo 4: LaTeX table row")
    print("-" * 70)
    r1.var_name = 'income'
    print(f"  {r1.to_latex()}")

    print("\n" + "=" * 70)
    print("  Demo complete. See README for full documentation.")
    print("=" * 70)

    return r1, results, r3


if __name__ == "__main__":
    demo()