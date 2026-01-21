from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel

from src.marketdata.curves.factories import ZeroRateCurveFactory

from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import CurveBootstrapSpec, CurveZeroSpec

from src.marketdata.providers.synthetic.context import SyntheticGenerationState
from src.marketdata.providers.synthetic.engine import rng_for_market_id
from src.marketdata.providers.synthetic.registry import SyntheticRegistry


def register_ir_generators(*, registry: SyntheticRegistry, base_seed: int, config: SyntheticProviderConfig) -> None:
    """
    Register all IR generators (curves, later: fixings, inflation, credit curves, etc.).

    Why this function exists
    ------------------------
    - Keeps the registration surface consistent across asset classes.
    - Allows the SyntheticProvider to remain stable while we expand Vn coverage.

    What gets registered today
    --------------------------
    - IR / CURVE : produces stored curve params and wires a ZeroRateCurveFactory.

    Parameters
    ----------
    registry:
        Central synthetic registry used by SyntheticMarketEngine.
    base_seed:
        Provider-level seed. Each MarketId gets its own deterministic sub-stream.
    config:
        SyntheticProviderConfig which provides CurveZeroSpec / CurveBootstrapSpec.
    """
    ir = _IrGenerators(base_seed=int(base_seed), config=config)

    # Register IR curves. MarketId specifics (ccy, curve type, index) live in mid.name/qualifiers.
    registry.register(asset_class="IR", mkt_type="CURVE", generator=ir.generate_curve)


@dataclass(frozen=True, slots=True)
class _IrGenerators:
    """
    Implementation container for IR synthetic generation.

    Notes
    -----
    We keep generation logic inside an object so:
    - access to seed + config is centralized
    - helper methods stay close to the asset class implementation
    - registration stays clean and consistent
    """
    base_seed: int
    config: SyntheticProviderConfig

    # ---------------------------------------------------------------------
    # CURVE
    # ---------------------------------------------------------------------

    def generate_curve(self, market_id: MarketId, state: SyntheticGenerationState) -> None:
        """
        Generate an IR curve panel and corresponding curve factory.

        Storage contract (canonical)
        ----------------------------
        curve_param_panels[mid] = Panel(data=[T,S,K,2], axis_names=("time","scenario","tenor","cols"))
            - cols: [tenor, zero_rate]
        curve_factories[mid] = ZeroRateCurveFactory(...)

        Curve method routing
        --------------------
        - If curve_method == "zeros": generate a smooth zero curve directly.
        - If curve_method == "bootstrap": (Vn-ready) we currently fall back to zeros
          using the CurveZeroSpec, but preserve the switch and spec plumbing.

        This keeps the provider "desk stable" while bootstrap engines evolve.
        """
        # Create a deterministic RNG stream for this exact MarketId key.
        rng = rng_for_market_id(base_seed=int(self.base_seed), market_id=market_id)

        # Decide which curve generation method is configured for this MarketId.
        method = str(self.config.curve_method_for(market_id)).strip().lower()

        if method == "zeros":
            # Use the configured zero-curve spec (supports per-id overrides).
            spec = self.config.curve_zero_spec(market_id)

            # Generate the canonical [T,S,K,2] params panel.
            params = _generate_zero_curve_param_panel(
                rng=rng,
                n_time=int(state.n_time),
                n_scenarios=int(state.n_scenarios),
                spec=spec,
            )

            # Store params panel into the shared state.
            state.curve_param_panels[market_id] = Panel(
                data=params,
                axis_names=("time", "scenario", "tenor", "cols"),
            )

            # Store the factory that will reconstruct a Curve object at snapshot time.
            state.curve_factories[market_id] = ZeroRateCurveFactory(
                extrapolation=str(spec.extrapolation),
            )
            return

        if method == "bootstrap":
            # Pull the bootstrap spec (so plumbing is correct and tests can verify routing).
            # NOTE: We are not using it yet; we will later route this into your bootstrapper
            #       (native/QuantLib) and still store the final tenor/zero grid.
            _bootstrap_spec: CurveBootstrapSpec = self.config.curve_bootstrap_spec(market_id)

            # For now, fall back to the zeros path to keep examples runnable.
            # This is intentionally conservative, not silent: behaviour is still deterministic.
            spec = self.config.curve_zero_spec(market_id)

            params = _generate_zero_curve_param_panel(
                rng=rng,
                n_time=int(state.n_time),
                n_scenarios=int(state.n_scenarios),
                spec=spec,
            )

            state.curve_param_panels[market_id] = Panel(
                data=params,
                axis_names=("time", "scenario", "tenor", "cols"),
            )
            state.curve_factories[market_id] = ZeroRateCurveFactory(
                extrapolation=str(spec.extrapolation),
            )
            return

        # Defensive guard: config should prevent this, but we keep this to be desk-safe.
        raise ValueError(f"Unsupported curve generation method={method!r} for MarketId={market_id.key()}.")


# -------------------------------------------------------------------------
# Numeric helpers (IR curve param generation)
# -------------------------------------------------------------------------

def _generate_zero_curve_param_panel(
    *,
    rng: np.random.Generator,
    n_time: int,
    n_scenarios: int,
    spec: CurveZeroSpec,
) -> np.ndarray:
    """
    Generate a canonical IR zero curve parameter panel.

    Output shape
    ------------
    [T, S, K, 2] where:
      - params[..., 0] = tenor grid (years)
      - params[..., 1] = zero rate at that tenor (continuous comp proxy)

    Curve shape model (stable, deterministic, market-like)
    ------------------------------------------------------
    r(tau) = base_rate + slope*tau + curvature*exp(-tau) + noise

    Notes
    -----
    - This is not a full curve model; it is a stable synthetic approximation.
    - The output is designed to reconstruct ZeroRateCurve objects consistently.
    """
    # Ensure integers for array shapes (prevents accidental float shapes).
    t_count = int(n_time)
    s_count = int(n_scenarios)

    # Normalize tenors to a clean 1D float array.
    tenors = np.asarray(spec.tenors, dtype=float).reshape(-1)
    if tenors.size == 0:
        raise ValueError("CurveZeroSpec.tenors must be non-empty.")

    # Ensure tenors are finite and strictly increasing (desk-grade guardrails).
    if not np.all(np.isfinite(tenors)):
        raise ValueError("CurveZeroSpec.tenors must be finite.")
    if np.any(np.diff(tenors) <= 0.0):
        raise ValueError("CurveZeroSpec.tenors must be strictly increasing.")

    # K is the number of tenor knot points.
    k_count = int(tenors.size)

    # Allocate output panel [T,S,K,2].
    out = np.empty((t_count, s_count, k_count, 2), dtype=float)

    # Column 0 is the tenor grid repeated across time/scenarios.
    out[..., 0] = tenors[None, None, :, None].squeeze(-1)

    # Build the deterministic base curve shape on the tenor grid.
    base_curve = (
        float(spec.base_rate)
        + float(spec.slope) * tenors
        + float(spec.curvature) * np.exp(-tenors)
    )

    # Create optional noise per (time,scenario,tenor).
    noise_scale = float(spec.noise_scale)
    if noise_scale > 0.0:
        noise = rng.normal(loc=0.0, scale=noise_scale, size=(t_count, s_count, k_count))
    else:
        noise = 0.0

    # Column 1 is the zero rate values.
    out[..., 1] = base_curve[None, None, :] + noise

    return out