"""Gaussian hidden Markov model with scaled forward-backward and Baum-Welch EM.

Everything in this module is deterministic: initialization is pinned
(quantile-based means, 0.9 self-transition prior, sample variance) and no
random number generator is used anywhere.

Scaled forward-backward (derivation)
------------------------------------
Let ``b_t(i) = p(x_t | s_t = i)`` be the Gaussian emission density,
``A`` the transition matrix and ``pi`` the initial distribution.  The
unscaled forward variable ``alpha_t(i) = p(x_1..x_t, s_t = i)`` underflows
for long series, so we normalize each step (Rabiner scaling).  To guard the
emission densities themselves against under/overflow we additionally factor
out the per-step maximum log-emission ``m_t = max_i log b_t(i)`` and work
with ``bt_t(i) = exp(log b_t(i) - m_t) <= 1``:

    a~_1(i) = pi_i * bt_1(i)                     d_1 = sum_i a~_1(i)
    a^_1(i) = a~_1(i) / d_1
    a~_t(j) = [sum_i a^_{t-1}(i) A_ij] * bt_t(j) d_t = sum_j a~_t(j)
    a^_t(j) = a~_t(j) / d_t

The true per-step normalizer is ``c_t = d_t * exp(m_t)`` and the
log-likelihood is the sum of the log normalizers:

    log p(x_1..x_T) = sum_t log c_t = sum_t (log d_t + m_t).

The backward pass uses the *same* normalizers:

    b^_T(i) = 1
    b^_t(i) = [sum_j A_ij bt_{t+1}(j) b^_{t+1}(j)] / d_{t+1}

With this scaling the posteriors need no further normalization:

    gamma_t(i)  = P(s_t = i | x_1..T) = a^_t(i) * b^_t(i)
    xi_t(i, j)  = P(s_t = i, s_{t+1} = j | x_1..T)
                = a^_t(i) A_ij bt_{t+1}(j) b^_{t+1}(j) / d_{t+1}

Baum-Welch M-step (diagonal Gaussian)
-------------------------------------
    pi_i     <- gamma_1(i)
    A_ij     <- sum_{t<T} xi_t(i,j) / sum_{t<T} gamma_t(i)
    mu_id    <- sum_t gamma_t(i) x_td / sum_t gamma_t(i)
    var_id   <- sum_t gamma_t(i) (x_td - mu_id)^2 / sum_t gamma_t(i)

Variances are floored at ``var_floor`` after every update.  Each EM
iteration provably does not decrease the log-likelihood; the tests assert
this on the bundled data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class HMMParams:
    """Parameter set of a diagonal-covariance Gaussian HMM.

    Attributes:
        startprob: Initial state distribution, shape ``(K,)``.
        transmat: Row-stochastic transition matrix, shape ``(K, K)``.
        means: State means, shape ``(K, D)``.
        variances: Per-dimension state variances, shape ``(K, D)``.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def copy(self) -> "HMMParams":
        """Return a deep copy of the parameter set."""
        return HMMParams(
            self.startprob.copy(),
            self.transmat.copy(),
            self.means.copy(),
            self.variances.copy(),
        )


@dataclass
class HMMFitResult:
    """Outcome of a Baum-Welch fit.

    Attributes:
        log_likelihood: Log-likelihood of the returned parameters.
        n_iter: Number of E-steps performed.
        converged: True iff the log-likelihood improvement fell below
            ``tol`` before ``max_iter`` was reached.  Non-convergence is
            reported through this flag, never as an exception.
        loglik_history: Log-likelihood at each E-step (non-decreasing).
    """

    log_likelihood: float
    n_iter: int
    converged: bool
    loglik_history: list = field(default_factory=list)


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Coerce observations to shape ``(T, D)`` and validate them.

    Args:
        X: Array of shape ``(T,)`` or ``(T, D)`` with ``D`` in {1, 2}.

    Returns:
        Validated float array of shape ``(T, D)``.

    Raises:
        ValueError: If the array is empty, has unsupported dimensionality,
            or contains NaN/inf.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"observations must be 1-D or 2-D, got ndim={X.ndim}")
    if X.shape[0] == 0:
        raise ValueError("observation sequence is empty")
    if X.shape[1] not in (1, 2):
        raise ValueError(f"only univariate or 2-D observations supported, got D={X.shape[1]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("observations contain NaN or inf")
    return X


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM with deterministic Baum-Welch EM.

    Args:
        n_states: Number of hidden states K.  Must be >= 2; K = 1 is a
            degenerate model and raises ``ValueError``.
        tol: Convergence tolerance on the log-likelihood improvement.
        max_iter: Maximum number of EM iterations.
        var_floor: Lower bound applied to every variance after each update
            (guards zero-variance states when K exceeds the data support).
    """

    def __init__(
        self,
        n_states: int,
        tol: float = 1e-8,
        max_iter: int = 500,
        var_floor: float = 1e-8,
    ) -> None:
        if int(n_states) != n_states or n_states < 2:
            raise ValueError(f"n_states must be an integer >= 2 (K=1 is degenerate), got {n_states}")
        if tol <= 0 or max_iter < 1 or var_floor <= 0:
            raise ValueError("tol and var_floor must be > 0 and max_iter >= 1")
        self.n_states = int(n_states)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.var_floor = float(var_floor)
        self.params: Optional[HMMParams] = None
        self.fit_result: Optional[HMMFitResult] = None

    # ------------------------------------------------------------------ #
    # Initialization (pinned, RNG-free)
    # ------------------------------------------------------------------ #
    def pinned_init(self, X: np.ndarray) -> HMMParams:
        """Deterministic EM starting point.

        Pinned rule: state ``k`` gets per-dimension means at the empirical
        quantile ``q_k = (2k + 1) / (2K)`` (linear interpolation), every
        state gets the full-sample variance (``ddof=0``, floored), the
        transition matrix has 0.9 self-transitions with the remaining mass
        spread uniformly, and the initial distribution is uniform.

        Args:
            X: Observations, shape ``(T,)`` or ``(T, D)``.

        Returns:
            The pinned initial :class:`HMMParams`.
        """
        X = _as_2d(X)
        K, D = self.n_states, X.shape[1]
        qs = (2.0 * np.arange(K) + 1.0) / (2.0 * K)
        means = np.quantile(X, qs, axis=0)  # (K, D), linear interpolation
        var = np.maximum(np.var(X, axis=0, ddof=0), self.var_floor)
        variances = np.tile(var, (K, 1))
        transmat = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(transmat, 0.9)
        startprob = np.full(K, 1.0 / K)
        return HMMParams(startprob, transmat, means, variances)

    # ------------------------------------------------------------------ #
    # Emission densities
    # ------------------------------------------------------------------ #
    def _log_emissions(self, X: np.ndarray, params: HMMParams) -> np.ndarray:
        """Log emission densities, shape ``(T, K)``.

        ``log b_t(i) = sum_d log N(x_td; mu_id, var_id)``.
        """
        diff = X[:, None, :] - params.means[None, :, :]  # (T, K, D)
        v = params.variances[None, :, :]
        return -0.5 * np.sum(_LOG_2PI + np.log(v) + diff * diff / v, axis=2)

    # ------------------------------------------------------------------ #
    # Scaled forward / backward
    # ------------------------------------------------------------------ #
    def _forward(self, logb: np.ndarray, params: HMMParams):
        """Scaled forward pass (see module docstring).

        Returns:
            Tuple ``(alpha_hat, d, m)`` with the scaled forward variables
            ``(T, K)``, per-step normalizers ``d`` and log-emission shifts
            ``m``; the log-likelihood is ``sum(log d + m)``.
        """
        T, K = logb.shape
        m = logb.max(axis=1)  # per-step shift guards emission underflow
        bt = np.exp(logb - m[:, None])
        alpha = np.empty((T, K))
        d = np.empty(T)
        a = params.startprob * bt[0]
        d[0] = a.sum()
        alpha[0] = a / d[0]
        A = params.transmat
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * bt[t]
            d[t] = a.sum()
            alpha[t] = a / d[t]
        return alpha, d, m

    def _backward(self, logb: np.ndarray, params: HMMParams, d: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Scaled backward pass sharing the forward normalizers ``d``."""
        T, K = logb.shape
        bt = np.exp(logb - m[:, None])
        beta = np.empty((T, K))
        beta[T - 1] = 1.0
        A = params.transmat
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (bt[t + 1] * beta[t + 1])) / d[t + 1]
        return beta

    # ------------------------------------------------------------------ #
    # Public inference API
    # ------------------------------------------------------------------ #
    def _require_fitted(self) -> HMMParams:
        if self.params is None:
            raise ValueError("model is not fitted; call fit() first")
        return self.params

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of ``X`` under the fitted parameters."""
        params = self._require_fitted()
        X = _as_2d(X)
        logb = self._log_emissions(X, params)
        _, d, m = self._forward(logb, params)
        return float(np.sum(np.log(d) + m))

    def filtered_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Filtered posteriors ``P(s_t = i | x_1..t)``, shape ``(T, K)``."""
        params = self._require_fitted()
        X = _as_2d(X)
        logb = self._log_emissions(X, params)
        alpha, _, _ = self._forward(logb, params)
        return alpha

    def smoothed_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posteriors ``P(s_t = i | x_1..T)``, shape ``(T, K)``."""
        params = self._require_fitted()
        X = _as_2d(X)
        logb = self._log_emissions(X, params)
        alpha, d, m = self._forward(logb, params)
        beta = self._backward(logb, params, d, m)
        return alpha * beta

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path (log-space Viterbi), shape ``(T,)``.

        Ties are broken toward the lower state index (argmax convention).
        """
        params = self._require_fitted()
        X = _as_2d(X)
        logb = self._log_emissions(X, params)
        T, K = logb.shape
        with np.errstate(divide="ignore"):
            logA = np.log(params.transmat)
            logpi = np.log(params.startprob)
        delta = logpi + logb[0]
        psi = np.empty((T, K), dtype=int)
        for t in range(1, T):
            cand = delta[:, None] + logA  # (from, to)
            psi[t] = np.argmax(cand, axis=0)
            delta = cand[psi[t], np.arange(K)] + logb[t]
        states = np.empty(T, dtype=int)
        states[T - 1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1][states[t + 1]]
        return states

    def stationary_distribution(self, tol: float = 1e-13, max_iter: int = 10000) -> np.ndarray:
        """Stationary distribution of the fitted transition matrix.

        Pinned power method: start from the uniform vector and iterate
        ``pi <- pi @ A`` with L1 renormalization until
        ``max|pi_new - pi| < tol`` (or ``max_iter``).

        Returns:
            Probability vector ``pi`` with ``pi @ A = pi``.
        """
        params = self._require_fitted()
        A = params.transmat
        pi = np.full(self.n_states, 1.0 / self.n_states)
        for _ in range(max_iter):
            nxt = pi @ A
            nxt = nxt / nxt.sum()
            if np.max(np.abs(nxt - pi)) < tol:
                return nxt
            pi = nxt
        return pi

    def n_parameters(self, n_dims: int) -> int:
        """Free parameter count: K*D means + K*D variances + K(K-1) transition + (K-1) initial."""
        K = self.n_states
        return 2 * K * n_dims + K * (K - 1) + (K - 1)

    def aic(self, X: np.ndarray) -> float:
        """Akaike information criterion ``-2 LL + 2 p`` (lower is better)."""
        X = _as_2d(X)
        return -2.0 * self.score(X) + 2.0 * self.n_parameters(X.shape[1])

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion ``-2 LL + p ln T`` (lower is better)."""
        X = _as_2d(X)
        return -2.0 * self.score(X) + self.n_parameters(X.shape[1]) * np.log(X.shape[0])

    # ------------------------------------------------------------------ #
    # Baum-Welch EM
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, init: Optional[HMMParams] = None) -> HMMFitResult:
        """Fit by Baum-Welch EM from a deterministic starting point.

        Args:
            X: Observations, shape ``(T,)`` or ``(T, D)``, ``T > K``.
            init: Optional warm-start parameters (used by the rolling
                refit schedule); defaults to :meth:`pinned_init`.

        Returns:
            :class:`HMMFitResult`; the fitted parameters (states relabeled
            so means[:, 0] are ascending) land in ``self.params``.

        Raises:
            ValueError: On invalid observations or ``T <= n_states``.
        """
        X = _as_2d(X)
        T = X.shape[0]
        if T <= self.n_states:
            raise ValueError(f"need more observations than states: T={T}, K={self.n_states}")
        params = init.copy() if init is not None else self.pinned_init(X)

        history: list = []
        ll_prev = -np.inf
        converged = False
        n_iter = 0
        for _ in range(self.max_iter):
            # E-step: scores the CURRENT parameters, so on convergence the
            # recorded log-likelihood matches the returned parameter set.
            logb = self._log_emissions(X, params)
            alpha, d, m = self._forward(logb, params)
            ll = float(np.sum(np.log(d) + m))
            history.append(ll)
            n_iter += 1
            if n_iter > 1 and ll - ll_prev < self.tol:
                converged = True
                break
            ll_prev = ll
            beta = self._backward(logb, params, d, m)
            gamma = alpha * beta  # (T, K), rows already sum to 1

            # xi summed over t, vectorized: sum_t outer(alpha_t, bt_{t+1} *
            # beta_{t+1} / d_{t+1}) elementwise-multiplied by A.
            bt = np.exp(logb - logb.max(axis=1)[:, None])
            w = bt[1:] * beta[1:] / d[1:, None]  # (T-1, K)
            xi_sum = params.transmat * (alpha[:-1].T @ w)  # (K, K)

            # M-step
            gsum = gamma.sum(axis=0)  # (K,)
            gsum_trans = gamma[:-1].sum(axis=0)
            params.startprob = gamma[0].copy()
            params.transmat = xi_sum / gsum_trans[:, None]
            params.means = (gamma.T @ X) / gsum[:, None]
            diff = X[:, None, :] - params.means[None, :, :]
            params.variances = np.maximum(
                np.einsum("tk,tkd->kd", gamma, diff * diff) / gsum[:, None],
                self.var_floor,
            )
        else:
            # max_iter exhausted: params had one more M-step than the last
            # recorded E-step, so score them once for a consistent report.
            logb = self._log_emissions(X, params)
            _, d, m = self._forward(logb, params)
            ll = float(np.sum(np.log(d) + m))

        # Pinned label convention: sort states by means[:, 0] ascending
        # (state 0 = lowest mean).  Applied once, after EM finishes.
        order = np.argsort(params.means[:, 0], kind="stable")
        params = HMMParams(
            params.startprob[order],
            params.transmat[np.ix_(order, order)],
            params.means[order],
            params.variances[order],
        )
        self.params = params
        self.fit_result = HMMFitResult(ll, n_iter, converged, history)
        return self.fit_result


def forward_step(
    params: HMMParams,
    alpha_prev: np.ndarray,
    x_t: np.ndarray,
) -> np.ndarray:
    """One incremental scaled-forward update for online filtering.

    Given the filtered posterior ``alpha_prev = P(s_{t-1} | x_1..t-1)`` and
    a new observation, return ``P(s_t | x_1..t)``.  Identical math to one
    step of :meth:`GaussianHMM._forward`; used by the regime gate between
    refits so the filter never re-reads the past.

    Args:
        params: Current HMM parameters.
        alpha_prev: Filtered probabilities at t-1, shape ``(K,)``.
        x_t: New observation, shape ``(D,)`` or scalar.

    Returns:
        Filtered probabilities at t, shape ``(K,)``.
    """
    x = np.atleast_1d(np.asarray(x_t, dtype=float))[None, :]
    diff = x[:, None, :] - params.means[None, :, :]
    v = params.variances[None, :, :]
    logb = -0.5 * np.sum(_LOG_2PI + np.log(v) + diff * diff / v, axis=2)[0]
    bt = np.exp(logb - logb.max())
    a = (alpha_prev @ params.transmat) * bt
    return a / a.sum()
