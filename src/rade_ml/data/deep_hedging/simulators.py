"""
Monte Carlo market simulators for deep hedging data generation.

Provides price path simulators under different dynamics:
    - GBMSimulator:    Geometric Brownian Motion (constant volatility)
    - HestonSimulator: Heston stochastic volatility model

Both simulators also compute the analytical Black-Scholes delta at each
rebalancing date for benchmark comparison against the learned hedge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Output of a market simulation.

    Attributes
    ----------
    spot_paths : np.ndarray
        Simulated spot prices [num_paths, num_steps + 1].
    vol_paths : np.ndarray or None
        Simulated instantaneous vol (Heston only) [num_paths, num_steps + 1].
    dt : float
        Time increment per step.
    times : np.ndarray
        Time grid [num_steps + 1].
    bs_deltas : np.ndarray or None
        Black-Scholes delta at each timestep [num_paths, num_steps + 1].
    payoffs : np.ndarray or None
        Option payoffs at maturity [num_paths].
    """

    spot_paths: np.ndarray = None
    vol_paths: Optional[np.ndarray] = None
    dt: float = 0.0
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    bs_deltas: Optional[np.ndarray] = None
    payoffs: Optional[np.ndarray] = None


class GBMSimulator:
    """
    Geometric Brownian Motion path simulator.

    dS = (r - q) * S * dt + sigma * S * dW

    Parameters
    ----------
    spot_0 : float
        Initial spot price.
    risk_free_rate : float
        Annualised risk-free rate.
    dividend_yield : float
        Continuous dividend yield.
    volatility : float
        Annualised volatility.
    """

    def __init__(
        self,
        spot_0: float = 100.0,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        volatility: float = 0.2,
    ) -> None:
        self.S0 = spot_0
        self.r = risk_free_rate
        self.q = dividend_yield
        self.sigma = volatility

    def simulate(
        self,
        maturity: float,
        num_steps: int,
        num_paths: int,
        seed: Optional[int] = None,
        strike: Optional[float] = None,
        option_type: str = "call",
    ) -> SimulationResult:
        """
        Generate GBM price paths and optionally compute BS deltas / payoffs.

        :param maturity: time to expiry in years
        :param num_steps: number of rebalancing intervals
        :param num_paths: number of Monte Carlo paths
        :param seed: random seed for reproducibility
        :param strike: option strike (if provided, computes BS delta and payoffs)
        :param option_type: "call" or "put"
        :return: SimulationResult
        """
        rng = np.random.default_rng(seed)
        dt = maturity / num_steps
        times = np.linspace(0.0, maturity, num_steps + 1)

        Z = rng.standard_normal((num_paths, num_steps))
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        log_returns = drift + diffusion
        log_paths = np.zeros((num_paths, num_steps + 1))
        log_paths[:, 0] = np.log(self.S0)
        log_paths[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)
        spot_paths = np.exp(log_paths)

        bs_deltas = None
        payoffs = None
        if strike is not None:
            bs_deltas = self._compute_bs_deltas(
                spot_paths, strike, maturity, times, option_type
            )
            if option_type.lower() == "call":
                payoffs = np.maximum(spot_paths[:, -1] - strike, 0.0)
            else:
                payoffs = np.maximum(strike - spot_paths[:, -1], 0.0)

        return SimulationResult(
            spot_paths=spot_paths.astype(np.float32),
            dt=dt,
            times=times.astype(np.float32),
            bs_deltas=bs_deltas.astype(np.float32) if bs_deltas is not None else None,
            payoffs=payoffs.astype(np.float32) if payoffs is not None else None,
        )

    def _compute_bs_deltas(
        self,
        spot_paths: np.ndarray,
        strike: float,
        maturity: float,
        times: np.ndarray,
        option_type: str,
    ) -> np.ndarray:
        """Compute BS delta at each point on the path grid."""
        num_paths, num_steps_plus_1 = spot_paths.shape
        deltas = np.zeros_like(spot_paths)

        for t_idx in range(num_steps_plus_1 - 1):
            tau = maturity - times[t_idx]
            if tau < 1e-10:
                if option_type.lower() == "call":
                    deltas[:, t_idx] = (spot_paths[:, t_idx] > strike).astype(float)
                else:
                    deltas[:, t_idx] = -(spot_paths[:, t_idx] < strike).astype(float)
                continue

            S = spot_paths[:, t_idx]
            d1 = (np.log(S / strike) + (self.r - self.q + 0.5 * self.sigma ** 2) * tau) / (
                self.sigma * np.sqrt(tau)
            )
            if option_type.lower() == "call":
                deltas[:, t_idx] = np.exp(-self.q * tau) * norm.cdf(d1)
            else:
                deltas[:, t_idx] = np.exp(-self.q * tau) * (norm.cdf(d1) - 1.0)

        # at maturity: delta is 1 (ITM) or 0 (OTM) for call
        if option_type.lower() == "call":
            deltas[:, -1] = (spot_paths[:, -1] > strike).astype(float)
        else:
            deltas[:, -1] = -(spot_paths[:, -1] < strike).astype(float)

        return deltas


class HestonSimulator:
    """
    Heston stochastic volatility model simulator.

    dS = (r - q) * S * dt + sqrt(v) * S * dW_1
    dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_2
    corr(dW_1, dW_2) = rho

    Uses the QE (Quadratic Exponential) scheme for the variance process
    to ensure non-negativity.

    Parameters
    ----------
    spot_0 : float
        Initial spot price.
    v0 : float
        Initial variance.
    risk_free_rate : float
        Annualised risk-free rate.
    dividend_yield : float
        Continuous dividend yield.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol-of-vol.
    rho : float
        Correlation between spot and variance Brownian motions.
    """

    def __init__(
        self,
        spot_0: float = 100.0,
        v0: float = 0.04,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        kappa: float = 1.5,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
    ) -> None:
        self.S0 = spot_0
        self.v0 = v0
        self.r = risk_free_rate
        self.q = dividend_yield
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def simulate(
        self,
        maturity: float,
        num_steps: int,
        num_paths: int,
        seed: Optional[int] = None,
        strike: Optional[float] = None,
        option_type: str = "call",
    ) -> SimulationResult:
        """
        Generate Heston price + variance paths.

        Uses Euler-Maruyama discretisation with full truncation for the
        variance process to maintain positivity.

        :param maturity: time to expiry in years
        :param num_steps: number of rebalancing intervals
        :param num_paths: number of Monte Carlo paths
        :param seed: random seed for reproducibility
        :param strike: option strike (if provided, computes payoffs)
        :param option_type: "call" or "put"
        :return: SimulationResult
        """
        rng = np.random.default_rng(seed)
        dt = maturity / num_steps
        times = np.linspace(0.0, maturity, num_steps + 1)

        # correlated Brownian increments
        Z1 = rng.standard_normal((num_paths, num_steps))
        Z2 = rng.standard_normal((num_paths, num_steps))
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1.0 - self.rho ** 2) * Z2

        spot_paths = np.zeros((num_paths, num_steps + 1))
        vol_paths = np.zeros((num_paths, num_steps + 1))
        spot_paths[:, 0] = self.S0
        vol_paths[:, 0] = self.v0

        for t in range(num_steps):
            v_t = np.maximum(vol_paths[:, t], 0.0)
            sqrt_v = np.sqrt(v_t)

            # variance process (full truncation)
            dv = self.kappa * (self.theta - v_t) * dt + self.xi * sqrt_v * np.sqrt(dt) * W2[:, t]
            vol_paths[:, t + 1] = np.maximum(v_t + dv, 0.0)

            # spot process
            drift = (self.r - self.q - 0.5 * v_t) * dt
            diffusion = sqrt_v * np.sqrt(dt) * W1[:, t]
            spot_paths[:, t + 1] = spot_paths[:, t] * np.exp(drift + diffusion)

        payoffs = None
        if strike is not None:
            if option_type.lower() == "call":
                payoffs = np.maximum(spot_paths[:, -1] - strike, 0.0)
            else:
                payoffs = np.maximum(strike - spot_paths[:, -1], 0.0)

        return SimulationResult(
            spot_paths=spot_paths.astype(np.float32),
            vol_paths=vol_paths.astype(np.float32),
            dt=dt,
            times=times.astype(np.float32),
            payoffs=payoffs.astype(np.float32) if payoffs is not None else None,
        )
