from __future__ import annotations

# NumPy is used purely for robust shape / ndim validation (ndarray coercion).
import numpy as np

# Typing imports for clarity and static checking.
from typing import Any, Mapping, Optional

# Core MarketDataset + MarketId + Panel data structures.
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel

# Factories are used to infer what storage layout we should validate against.
from src.marketdata.curves.factory import ZeroRateCurveFactory
from src.marketdata.surfaces.factory import GridVolFactory


def validate_dataset_layout(ds: MarketDataset, *, strict: bool = True) -> None:
    """
    Opinionated validation for MarketDataset storage contracts.

    This is deliberately *stricter* than MarketDataset.__post_init__ because we
    want predictable, desk-grade storage semantics.

    Parameters
    ----------
    strict:
        If True, enforce canonical storage:
          - quotes: [T] or [T,S]
          - curves (ZeroRateCurveFactory): [T,S,K,2] with axis_names ("time","scenario","tenor","cols")
          - vols   (GridVolFactory):       [T,S,n_exp,n_k] with axis_names ("time","scenario","expiry","strike")

        If False, allow legacy encodings (not recommended for production):
          - vols may also be stored as [T,S,P] with P = n_exp*n_k and axis_names ("time","scenario","params")
    """
    # Number of time points in the dataset (T).
    t_count = len(ds.dates)
    # Number of scenarios in the dataset (S).
    s_count = int(ds.n_scenarios)

    # ---------------------------------------------------------------------
    # Quotes
    # ---------------------------------------------------------------------
    # Validate each quote panel in ds.panels.
    for mid, panel in ds.panels.items():
        # Ensure the panel declares an axis_names tuple and that the first axis is "time".
        if not panel.axis_names or panel.axis_names[0] != "time":
            raise ValueError(
                f"Quote panel first axis must be 'time' for {mid.key()}, got axis_names={panel.axis_names}."
            )

        # Coerce panel.data to an ndarray so we can reliably check ndim/shape.
        x = np.asarray(panel.data)

        # 1D: [T] quotes with no scenario axis.
        if x.ndim == 1:
            # Check length matches number of dates in the dataset.
            if x.shape[0] != t_count:
                raise ValueError(
                    f"Quote panel [T] mismatch for {mid.key()}: got {x.shape[0]} expected T={t_count}."
                )
            # Continue to next quote panel once validated.
            continue

        # 2D: [T,S] quotes with scenario axis.
        if x.ndim == 2:
            # Ensure the second axis name exists and is declared as "scenario".
            if len(panel.axis_names) < 2 or panel.axis_names[1] != "scenario":
                raise ValueError(
                    f"Quote panel [T,S] must declare axis_names[1]=='scenario' for {mid.key()}, "
                    f"got axis_names={panel.axis_names}."
                )
            # Validate exact shape matches (T,S).
            if x.shape != (t_count, s_count):
                raise ValueError(
                    f"Quote panel [T,S] shape mismatch for {mid.key()}: got {x.shape} expected ({t_count},{s_count})."
                )
            # Continue to next quote panel once validated.
            continue

        # Anything else is rejected: quotes are only allowed as [T] or [T,S].
        raise ValueError(f"Quote panel ndim must be 1 or 2 for {mid.key()}, got ndim={x.ndim}.")

    # ---------------------------------------------------------------------
    # Curves (ZeroRateCurveFactory)
    # ---------------------------------------------------------------------
    # Validate each curve parameter panel in ds.curve_params.
    for mid, panel in ds.curve_params.items():
        # Ensure the curve param panel declares first axis as "time".
        if not panel.axis_names or panel.axis_names[0] != "time":
            raise ValueError(
                f"Curve param panel first axis must be 'time' for {mid.key()}, got axis_names={panel.axis_names}."
            )

        # Look up the factory that will reconstruct the curve object at snapshot time.
        factory = ds.curve_factories.get(mid)
        # If curve_params exists but factory is missing, the dataset is inconsistent.
        if factory is None:
            raise ValueError(f"Missing curve factory for {mid.key()} (curve_params exists but curve_factories absent).")

        # Coerce data to ndarray for robust shape/ndim checks.
        x = np.asarray(panel.data)

        # Only validate structure for ZeroRateCurveFactory here (other curve factories can be added later).
        if isinstance(factory, ZeroRateCurveFactory):
            # Canonical storage is [T,S,K,2].
            if x.ndim != 4:
                raise ValueError(
                    f"ZeroRateCurveFactory expects curve panel ndim=4 for {mid.key()}, got ndim={x.ndim}."
                )

            # axis_names must contain 4 entries for the 4D tensor.
            if len(panel.axis_names) < 4:
                raise ValueError(f"Curve panel axis_names too short for {mid.key()}: {panel.axis_names}")

            # Second axis must be "scenario".
            if panel.axis_names[1] != "scenario":
                raise ValueError(f"Curve panel must have axis_names[1]=='scenario' for {mid.key()}.")

            # Validate the [T,S,...] prefix matches dataset sizes.
            if x.shape[0] != t_count or x.shape[1] != s_count:
                raise ValueError(
                    f"Curve panel [T,S,...] mismatch for {mid.key()}: got {x.shape[:2]} expected ({t_count},{s_count})."
                )

            # Last dim must be 2: [tenor, zero_rate].
            if x.shape[-1] != 2:
                raise ValueError(
                    f"Zero curve params must have last dim size 2 (tenor, zero_rate) for {mid.key()}, got {x.shape[-1]}."
                )

    # ---------------------------------------------------------------------
    # Vols (GridVolFactory)
    # ---------------------------------------------------------------------
    # Validate each vol parameter panel in ds.vol_params.
    for mid, panel in ds.vol_params.items():
        # Ensure the vol param panel declares first axis as "time".
        if not panel.axis_names or panel.axis_names[0] != "time":
            raise ValueError(
                f"Vol param panel first axis must be 'time' for {mid.key()}, got axis_names={panel.axis_names}."
            )

        # Look up the factory that will reconstruct the vol surface object at snapshot time.
        factory = ds.vol_factories.get(mid)
        # If vol_params exists but factory is missing, the dataset is inconsistent.
        if factory is None:
            raise ValueError(f"Missing vol factory for {mid.key()} (vol_params exists but vol_factories absent).")

        # Coerce data to ndarray for robust shape/ndim checks.
        x = np.asarray(panel.data)

        # Only validate structure for GridVolFactory here (other surface factories can be added later).
        if isinstance(factory, GridVolFactory):
            # Factory defines the expiries / strikes grid sizes.
            expiries = np.asarray(factory.expiries, dtype=float).reshape(-1)
            strikes = np.asarray(factory.strikes, dtype=float).reshape(-1)

            # Compute expected grid dimensions.
            n_exp = int(expiries.size)
            n_k = int(strikes.size)

            # Require the scenario axis to be declared at dim=1.
            if len(panel.axis_names) < 2 or panel.axis_names[1] != "scenario":
                raise ValueError(
                    f"Vol panel must have axis_names[1]=='scenario' for {mid.key()}, got axis_names={panel.axis_names}."
                )

            # Validate [T,S,...] prefix matches dataset sizes.
            if x.shape[0] != t_count or x.shape[1] != s_count:
                raise ValueError(
                    f"Vol panel [T,S,...] mismatch for {mid.key()}: got {x.shape[:2]} expected ({t_count},{s_count})."
                )

            # ------------------------------------------------------------------
            # Canonical (strict) storage: [T,S,n_exp,n_k] with ("expiry","strike")
            # ------------------------------------------------------------------
            if x.ndim == 4:
                # axis_names should have 4 entries for the 4D tensor.
                if len(panel.axis_names) < 4:
                    raise ValueError(f"Vol panel axis_names too short for {mid.key()}: {panel.axis_names}")

                # In strict mode, enforce semantic axis naming for expiry/strike dims.
                if strict:
                    if panel.axis_names[2] not in {"expiry", "expiries"} or panel.axis_names[3] not in {"strike", "strikes"}:
                        raise ValueError(
                            f"Strict mode: vol grid must declare axes ('expiry','strike') at dims (2,3) for {mid.key()}, "
                            f"got axis_names={panel.axis_names}."
                        )

                # Validate the grid dimensions match factory configuration.
                if x.shape[2] != n_exp or x.shape[3] != n_k:
                    raise ValueError(
                        f"Vol grid shape mismatch for {mid.key()}: got (n_exp,n_k)=({x.shape[2]},{x.shape[3]}) "
                        f"expected ({n_exp},{n_k})."
                    )

                # Successfully validated this vol panel.
                continue

            # ------------------------------------------------------------------
            # Legacy encoding (non-strict only): [T,S,P] with P=n_exp*n_k
            # ------------------------------------------------------------------
            if (not strict) and x.ndim == 3:
                # axis_names should have 3 entries for the 3D tensor.
                if len(panel.axis_names) < 3:
                    raise ValueError(f"Vol panel axis_names too short for {mid.key()}: {panel.axis_names}")

                # Third axis must be named "params" for the flattened representation.
                if panel.axis_names[2] != "params":
                    raise ValueError(
                        f"Legacy vol params must declare axis_names[2]=='params' for {mid.key()}, got axis_names={panel.axis_names}."
                    )

                # Validate flattened size equals n_exp * n_k.
                expected = n_exp * n_k
                if x.shape[2] != expected:
                    raise ValueError(
                        f"Legacy flattened vol size mismatch for {mid.key()}: got P={x.shape[2]} expected P={expected}."
                    )

                # Successfully validated this vol panel in legacy mode.
                continue

            # If strict mode is enabled, any non-4D grid storage is rejected.
            if strict:
                raise ValueError(
                    f"Strict mode: GridVolFactory vol panel must be ndim=4 for {mid.key()}, got ndim={x.ndim}."
                )

            # Non-strict mode: we still only allow ndim=4 or ndim=3 legacy; otherwise reject.
            raise ValueError(
                f"GridVolFactory vol panel must be ndim=4 (grid) or ndim=3 (legacy params) for {mid.key()}, got ndim={x.ndim}."
            )


def build_marketdataset(
    *,
    dates: list[str],
    n_scenarios: int,
    quote_panels: Mapping[MarketId, Panel],
    curve_param_panels: Mapping[MarketId, Panel],
    curve_factories: Mapping[MarketId, Any],
    vol_param_panels: Mapping[MarketId, Panel],
    vol_factories: Mapping[MarketId, Any],
    meta: Optional[Mapping[str, Any]] = None,
    validate: bool = True,
    strict: bool = True,
) -> MarketDataset:
    """
    Thin builder/wirer for MarketDataset.

    This exists to keep examples clean and enforce consistent construction.

    Parameters
    ----------
    validate:
        If True, runs validate_dataset_layout(...) in addition to MarketDataset.__post_init__.
    strict:
        Passed to validate_dataset_layout(...). Use strict=True for production-grade datasets.
    """
    # Construct the MarketDataset (this will also run MarketDataset.__post_init__ invariants).
    ds = MarketDataset(
        dates=dates,                              # Time grid (list of ISO date strings).
        n_scenarios=int(n_scenarios),             # Scenario count (S).
        panels=quote_panels,                      # Scalar quote panels keyed by MarketId.
        curve_params=curve_param_panels,          # Curve parameter panels keyed by MarketId.
        curve_factories=curve_factories,          # Curve factories keyed by MarketId (build at snapshot time).
        vol_params=vol_param_panels,              # Vol parameter panels keyed by MarketId.
        vol_factories=vol_factories,              # Vol factories keyed by MarketId (build at snapshot time).
        meta=meta,                                # Optional metadata (provider name, seed, etc.).
    )

    # Optionally run our stricter, opinionated validation checks.
    if validate:
        validate_dataset_layout(ds, strict=strict)

    # Return the dataset to callers (examples/providers/builders).
    return ds