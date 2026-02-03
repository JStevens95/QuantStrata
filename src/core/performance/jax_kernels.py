"""
JAX-based MC kernels (optional): GBM path generation and vanilla/digital payoff.

Use when JAX is installed for CPU/GPU acceleration. All functions raise
if JAX is not available. Device is determined by JAX default backend.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from src.core.performance.backend import jax_available

OptionTypeStr = Literal["call", "put"]


def _require_jax() -> None:
    if not jax_available():
        raise RuntimeError(
            "JAX is not installed. Install with: pip install jax jaxlib"
        )


def _jax_scan(step_fn: Any, carry: Any, n: int) -> tuple[Any, Any]:
    """Run step_fn n times; returns (final_carry, stacked_ys)."""
    import jax.lax as lax
    def body(c, _):
        (c_new, y) = step_fn(c, None)
        return c_new, y
    return lax.scan(body, carry, None, length=n)


def gbm_paths_jax(
    spot0: float,
    drift: float,
    vol: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    key: Any,
    *,
    device: Optional[Any] = None,
) -> Any:
    """
    Generate GBM paths using JAX (exact discretization).

    S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z).

    Parameters
    ----------
    spot0 : float
        Initial spot.
    drift : float
        Drift mu.
    vol : float
        Volatility sigma.
    n_paths : int
        Number of paths.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    key : jax.random.PRNGKey
        JAX random key (e.g. jax.random.PRNGKey(seed)).
    device : optional
        JAX device; None = default backend (CPU/GPU).

    Returns
    -------
    jax.Array
        Terminal spots, shape (n_paths,) or paths (n_paths, n_steps+1) if requested.
    """
    _require_jax()
    import jax.numpy as jnp
    import jax.random as jr

    sqrt_dt = jnp.sqrt(dt)
    log_drift = (drift - 0.5 * vol * vol) * dt
    vol_sqrt_dt = vol * sqrt_dt

    def step(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Any]:
        spots, k = carry
        k, k_next = jr.split(k)
        z = jr.normal(k, shape=spots.shape)
        next_spot = spots * jnp.exp(log_drift + vol_sqrt_dt * z)
        return (next_spot, k_next), next_spot

    k, k0 = jr.split(key)
    z0 = jr.normal(k0, shape=(n_paths,))
    spots = jnp.full((n_paths,), float(spot0)) * jnp.exp(log_drift + vol_sqrt_dt * z0)

    carry_final, _ = _jax_scan(step, (spots, k), n_steps - 1)
    final_spots = carry_final[0]
    return final_spots


def gbm_terminal_spots_jax(
    spot0: float,
    drift: float,
    vol: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    key: Any,
    *,
    device: Optional[Any] = None,
) -> Any:
    """
    Generate GBM terminal spot values only (no path storage) using JAX.

    Single-step exact: S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z).
    """
    _require_jax()
    import jax.numpy as jnp
    import jax.random as jr

    T = n_steps * dt
    sqrt_T = jnp.sqrt(T)
    log_drift = (drift - 0.5 * vol * vol) * T
    z = jr.normal(key, shape=(n_paths,))
    terminal = jnp.full((n_paths,), float(spot0)) * jnp.exp(log_drift + vol * sqrt_T * z)
    return terminal


def vanilla_payoff_jax(
    spots: Any,
    strike: float,
    option_type: OptionTypeStr,
) -> Any:
    """
    Vanilla option payoff in JAX: max(S-K, 0) for call, max(K-S, 0) for put.

    Parameters
    ----------
    spots : jax.Array
        Terminal spots, shape (n_paths,).
    strike : float
        Strike K.
    option_type : "call" | "put"

    Returns
    -------
    jax.Array
        Payoffs, shape (n_paths,).
    """
    _require_jax()
    import jax.numpy as jnp

    if option_type == "call":
        return jnp.maximum(spots - strike, 0.0)
    return jnp.maximum(strike - spots, 0.0)


def digital_payoff_jax(
    spots: Any,
    strike: float,
    option_type: OptionTypeStr,
    payout: float = 1.0,
) -> Any:
    """Digital (cash-or-nothing) payoff in JAX."""
    _require_jax()
    import jax.numpy as jnp

    if option_type == "call":
        return jnp.where(spots > strike, payout, 0.0)
    return jnp.where(spots < strike, payout, 0.0)
