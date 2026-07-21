"""Optimized collapsed Gibbs sampler for Dirichlet process Gaussian mixtures.

This module is a refactor of an earlier implementation that relied heavily on
``numpy.matrix`` objects and repeatedly recomputed expensive quantities during
sampling.  The classes below keep only the sufficient statistics needed for a
Normal-Inverse-Wishart posterior and use numerically stable log-probabilities
whenever possible.  These changes drastically reduce Python level overhead and
allocate far fewer temporary arrays while running the sampler.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import math
import numpy as np
import scipy.special as ssp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal


def _ensure_array(x: np.ndarray | Iterable[float], *, ndim: int = 1) -> np.ndarray:
    """Convert *x* to a ``np.ndarray`` with the desired dimensionality."""

    arr = np.asarray(x, dtype=np.float64)
    if ndim == 1:
        return arr.reshape(-1)
    return np.atleast_2d(arr)


class Gaussian:
    """Gaussian component with Normal-Inverse-Wishart prior.

    The implementation maintains only the sufficient statistics required for
    collapsed Gibbs sampling, avoiding the need to store the raw data assigned
    to each component.  Predictive probabilities are evaluated using the
    multivariate Student-t distribution in log space, which keeps the sampler
    numerically stable and removes repeated determinant/inverse computations.
    """

    _mu_0: np.ndarray
    _kappa_0: float
    _nu_0: float
    _Psi_0: np.ndarray
    dim: int
    nk: int
    _sum: np.ndarray
    _square_sum: np.ndarray

    def __init__(self, X: np.ndarray | None = None, hyperparams: Dict[str, np.ndarray] | None = None):
        if X is None:
            X = np.zeros((0, 0), dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        if hyperparams is None:
            if X.size == 0:
                raise ValueError("Cannot infer dimensionality without data or hyperparameters.")
            dim = X.shape[1]
            mu_0 = np.zeros(dim, dtype=np.float64)
            kappa_0 = 1.0
            nu_0 = float(dim + 1)
            Psi_0 = np.eye(dim, dtype=np.float64)
        else:
            mu_0 = _ensure_array(hyperparams["mu_0"], ndim=1)
            dim = mu_0.size
            kappa_0 = float(hyperparams["kappa_0"])
            nu_0 = float(hyperparams["nu_0"])
            Psi_0 = np.asarray(hyperparams["Psi_0"], dtype=np.float64)

        if X.size and X.shape[1] != dim:
            raise ValueError("Data dimensionality does not match the provided hyperparameters.")

        self.dim = dim
        self._mu_0 = mu_0
        self._kappa_0 = kappa_0
        self._nu_0 = max(nu_0, self.dim + 1.0)
        self._Psi_0 = Psi_0

        self.nk = X.shape[0]
        if self.nk > 0:
            self._sum = X.sum(axis=0)
            self._square_sum = X.T @ X
        else:
            self._sum = np.zeros(self.dim, dtype=np.float64)
            self._square_sum = np.zeros((self.dim, self.dim), dtype=np.float64)

        self._recompute_cache()

    # Cached quantities -------------------------------------------------

    def _recompute_cache(self) -> None:
        if self.nk == 0:
            self._kappa_n = self._kappa_0
            self._nu_n = self._nu_0
            self._Psi_n = np.array(self._Psi_0, copy=True)
            self.mean = np.array(self._mu_0, copy=True)
        else:
            mean_data = self._sum / self.nk
            diff = mean_data - self._mu_0
            centered = self._square_sum - self.nk * np.outer(mean_data, mean_data)

            self._kappa_n = self._kappa_0 + self.nk
            self._nu_n = self._nu_0 + self.nk
            self._Psi_n = self._Psi_0 + centered + (
                (self._kappa_0 * self.nk) / self._kappa_n
            ) * np.outer(diff, diff)
            self.mean = (self._kappa_0 * self._mu_0 + self.nk * mean_data) / self._kappa_n

        df = self._nu_n - self.dim + 1.0
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive for predictive distribution.")

        scale = ((self._kappa_n + 1.0) / (self._kappa_n * df)) * self._Psi_n
        self.covar = (scale + scale.T) / 2.0  # ensure symmetry

        self._chol_factor = cho_factor(self.covar, lower=True, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(self._chol_factor[0])))
        self._df = df
        self._log_norm = (
            gammaln((df + self.dim) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * (self.dim * np.log(df * np.pi) + log_det)
        )

    # Updates -----------------------------------------------------------

    def add_point(self, x: np.ndarray) -> None:
        x = _ensure_array(x, ndim=1)
        if x.size != self.dim:
            raise ValueError("Point dimensionality does not match component.")

        self.nk += 1
        self._sum += x
        self._square_sum += np.outer(x, x)
        self._recompute_cache()

    def rm_point(self, x: np.ndarray) -> None:
        if self.nk <= 0:
            raise ValueError("Cannot remove a point from an empty component.")

        x = _ensure_array(x, ndim=1)
        self.nk -= 1
        self._sum -= x
        self._square_sum -= np.outer(x, x)

        if self.nk == 0:
            self._sum.fill(0.0)
            self._square_sum.fill(0.0)

        self._recompute_cache()

    # Predictive probabilities -----------------------------------------

    def log_predictive(self, x: np.ndarray) -> float:
        # x = _ensure_array(x, ndim=1)
        # diff = x - self.mean
        # mahal = diff @ cho_solve(self._chol_factor, diff, check_finite=False)
        # return self._log_norm - 0.5 * (self._df + self.dim) * math.log1p(mahal / self._df)
        d = self.dim
        self.nu = self._nu_0 + self.nk
        df = self.nu - d + 1
        assert d == self.mean.shape[1]
        assert (d, d) == self.covar.shape
        x_mu = x - self.mean
        sign, res = np.linalg.slogdet(self.covar)
        lognum = gammaln((d+df)/2.)
        logdenom = gammaln(df/2.) + d/2.*np.log(df*np.pi) + 1/2.*res + (d+df)/2.*np.log(1+1./df * x_mu * self.covar.I * x_mu.transpose())
        return lognum-logdenom

    def pdf(self, x: np.ndarray) -> float:
        return float(np.exp(self.log_predictive(x)))


class DpMixture:
    """Collapsed Gibbs sampler for a Dirichlet process Gaussian mixture."""

    def __init__(
        self,
        data: np.ndarray,
        hyparams: Dict[str, np.ndarray],
        alpha: float,
        sample_alpha: bool = False,
        alpha_a: float = 100.0,
        alpha_b: float = 100.0,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.hyparams = hyparams
        self.alpha = alpha
        self.sample_alpha = sample_alpha
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b

        self.assigns = np.zeros(self.data.shape[0], dtype=int)
        self.params: Dict[int, Gaussian] = {0: Gaussian(self.data, self.hyparams)}
        self._K = 1
        self.n_samples = np.array([self.data.shape[0]], dtype=int)
        self.marginallikes: list[float] = []
        self._prior_component = Gaussian(np.zeros((0, self.data.shape[1])), self.hyparams)

    # ------------------------------------------------------------------

    def _sample_z(self, i: int) -> None:
        x = self.data[i]
        old_k = self.assigns[i]
        self.params[old_k].rm_point(x)

        keys = list(self.params.keys())
        log_probs = np.empty(len(keys) + 1, dtype=np.float64)

        for idx, k in enumerate(keys):
            nk = self.params[k].nk
            if nk == 0:
                log_probs[idx] = -np.inf
            else:
                log_probs[idx] = math.log(nk) + self.params[k].log_predictive(x)

        log_probs[-1] = math.log(self.alpha) + self._prior_component.log_predictive(x)

        probs = np.exp(log_probs - logsumexp(log_probs))
        new_comp_index = np.searchsorted(np.cumsum(probs), np.random.random())

        if new_comp_index == len(keys):
            new_k = self._K
            self.params[new_k] = Gaussian(np.zeros((0, self.data.shape[1])), self.hyparams)
            self._K += 1
        else:
            new_k = keys[new_comp_index]

        self.assigns[i] = new_k
        self.params[new_k].add_point(x)

    # ------------------------------------------------------------------

    def gibbs(self, iterations: int = 1, snapshot_interval: int = 100) -> None:
        for it in range(iterations):
            print('iteration:', it)
            if it % snapshot_interval == 0 or it == iterations - 1:
                print(f"Gibbs sampling iteration: {it}")
                print(f"Alpha: {self.alpha}")
                print(f"Number of components: {self._K}")

            if self.sample_alpha:
                self.alpha = sample_concentration(self.n_samples, self.alpha_a, self.alpha_b)

            for idx in np.random.permutation(len(self.data)):
                self._sample_z(idx)

            self.n_samples = np.zeros(self._K, dtype=int)
            for comp_id, comp in self.params.items():
                if comp_id >= self.n_samples.size:
                    self.n_samples = np.pad(self.n_samples, (0, comp_id - self.n_samples.size + 1))
                self.n_samples[comp_id] = comp.nk

            self.compact_params()
            self.marginallikes.append(self.get_logpdf(self.data))

    # ------------------------------------------------------------------

    def compact_params(self) -> None:
        unused = np.nonzero(self.n_samples == 0)[0]
        used = np.nonzero(self.n_samples != 0)[0]

        if unused.size == 0:
            return

        self._K = used.size
        self.n_samples = self.n_samples[used]

        new_params: Dict[int, Gaussian] = {}
        for new_idx, old_idx in enumerate(used):
            new_params[new_idx] = self.params.pop(old_idx)
        self.params = new_params

        for new_idx in range(self._K):
            mask = self.assigns == used[new_idx]
            self.assigns[mask] = new_idx

    # ------------------------------------------------------------------

    def get_logpdf(self, data: np.ndarray | None = None) -> float:
        if data is None:
            data = self.data

        weights, dists = dict2mix(self.params)
        loglik = sum(all_loglike(X, weights, dists) for X in data)

        if self.sample_alpha:
            loglik += (
                (self.alpha_a - 1.0) * math.log(self.alpha)
                - self.alpha / self.alpha_b
                - self.alpha_a * math.log(self.alpha_b)
                - ssp.gammaln(self.alpha_a)
            )
        return loglik


def dict2mix(dic: Dict[int, Gaussian]) -> Tuple[np.ndarray, Dict[int, multivariate_normal]]:
    weights = np.zeros(len(dic), dtype=np.float64)
    dists: Dict[int, multivariate_normal] = {}
    idx = 0

    for key, component in dic.items():
        if component.nk > 0:
            weights[idx] = float(component.nk)
            dists[idx] = multivariate_normal(mean=component.mean, cov=component.covar)
            idx += 1

    weights = weights[:idx]
    if weights.size:
        weights /= weights.sum()
    return weights, dists


def mixture_logpdf(x: np.ndarray, weights: np.ndarray, dists: Dict[int, multivariate_normal]) -> float:
    if not dists:
        raise ValueError("Mixture has no active components.")

    loglikelihoods = np.empty(len(dists), dtype=np.float64)
    for key in dists:
        loglikelihoods[key] = math.log(weights[key]) + dists[key].logpdf(x)
    return float(logsumexp(loglikelihoods))


def all_loglike(X: Iterable[np.ndarray], weights: np.ndarray, dists: Dict[int, multivariate_normal]) -> float:
    return sum(mixture_logpdf(np.asarray(x, dtype=np.float64), weights, dists) for x in X)


def sample_concentration(
    nk: np.ndarray, alpha_a: float = 100.0, alpha_b: float = 100.0, max_iter_alpha: int = 20
) -> float:
    K = len(nk)
    total = float(np.sum(nk))
    alpha = np.random.gamma(alpha_a, alpha_b)

    for _ in range(max_iter_alpha):
        w_alpha = np.random.beta(alpha + 1.0, total)
        rate = alpha_b - math.log(w_alpha)
        pi = alpha_a + K - 1.0
        s_alpha = np.random.binomial(1, (rate * total) / (pi + rate * total))
        alpha = np.random.gamma(alpha_a + K - s_alpha, 1.0 / rate)
    return float(alpha)