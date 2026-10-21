"""
ChiralMeta GP Model — ExactGPLayer (RBF / Matern / Linear kernels).
Refactored from Prototypical_networks/model.py with no hardcoded paths.
gpytorch is optional; falls back gracefully if not installed.
"""

import logging
import numpy as np

log = logging.getLogger("ChiralMeta.GPModel")

try:
    import torch
    import gpytorch

    class ExactGPLayer(gpytorch.models.ExactGP):
        """
        Gaussian Process regression / classification layer.
        Supports 'linear', 'rbf', 'RBF', 'matern' kernels.
        """

        def __init__(self, train_x, train_y, likelihood, kernel, ard_num_dims=None):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            if kernel == "linear":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.LinearKernel()
                )
            elif kernel in ("rbf", "RBF"):
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
                )
            elif kernel == "matern":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims)
                )
            else:
                raise ValueError(f"Unsupported kernel: {kernel!r}")

        def forward(self, x):
            mean_x  = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    HAS_GP = True

except ImportError:
    log.warning("gpytorch not installed — GP model disabled. Install with: pip install gpytorch")
    HAS_GP = False
    ExactGPLayer = None  # type: ignore


# ----------------------------------------------------------------
# Simple sklearn-based GP wrapper (fallback when gpytorch missing)
# ----------------------------------------------------------------

class SklearnGPBaseline:
    """
    Gaussian Process baseline using sklearn's GaussianProcessClassifier.
    Used as fallback when gpytorch is not installed.
    """

    def __init__(self, kernel: str = "rbf"):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern
        k = Matern() if kernel == "matern" else RBF()
        self.model = GaussianProcessClassifier(kernel=k, n_restarts_optimizer=2)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            self.model.fit(X, y)
            self.is_fitted = True
        except Exception as e:
            log.warning(f"GP fit failed: {e}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full((len(X), 2), 0.5)
        try:
            return self.model.predict_proba(X)
        except Exception:
            return np.full((len(X), 2), 0.5)
