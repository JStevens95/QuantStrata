from __future__ import annotations

import numpy as np
from typing import Any, Mapping, Optional

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel

from src.marketdata.curves.factories import ZeroRateCurveFactory
from src.marketdata.surfaces.factories import GridVolFactory


def validate_dataset_layout(ds: MarketDataset) -> None:
    """
    Extra (optional) validation beyond MarketDataset.__post_init__.

    This is intentionally opinionated for V1:
    - quote panels: [T] or [T,S] with scenario axis declared when 2D
    - zero curve panels: when using ZeroRateCurveFactory -> expect [T,S,K,2]
    - grid vol panels: when using GridVolFactory -> expect [T,S,n_exp,n_k] with matching sizes
    """
    T = len(ds.dates)
    S = int(ds.n_scenarios)

    # --- Quotes ---
    for mid, p in ds.panels.items():
        if p.axis_names[0] != "time":
            raise ValueError(f"Quote panel first axis must be 'time' for {mid.key()}, got {p.axis_names}.")
        x = np.asarray(p.data)
        if x.ndim == 1:
            if x.shape[0] != T:
                raise ValueError(f"Quote panel [T] mismatch for {mid.key()}: {x.shape[0]} vs T={T}.")
        elif x.ndim == 2:
            if len(p.axis_names) < 2 or p.axis_names[1] != "scenario":
                raise ValueError(
                    f"Quote panel [T,S] must declare axis_names[1]=='scenario' for {mid.key()}."
                )
            if x.shape != (T, S):
                raise ValueError(f"Quote panel shape mismatch for {mid.key()}: {x.shape} vs ({T},{S}).")
        else:
            raise ValueError(f"Quote panel ndim must be 1 or 2 for {mid.key()}, got {x.ndim}.")

    # --- Curves ---
    for mid, p in ds.curve_params.items():
        if p.axis_names[0] != "time":
            raise ValueError(f"Curve panel first axis must be 'time' for {mid.key()}, got {p.axis_names}.")
        factory = ds.curve_factories[mid]
        x = np.asarray(p.data)

        if isinstance(factory, ZeroRateCurveFactory):
            if x.ndim != 4:
                raise ValueError(f"ZeroRateCurveFactory expects curve panel ndim=4 for {mid.key()}, got {x.ndim}.")
            if len(p.axis_names) < 4:
                raise ValueError(f"Curve panel axis_names too short for {mid.key()}: {p.axis_names}")
            if p.axis_names[1] != "scenario":
                raise ValueError(f"Curve panel must have scenario axis at dim=1 for {mid.key()}.")
            if x.shape[0] != T or x.shape[1] != S:
                raise ValueError(f"Curve panel [T,S,...] mismatch for {mid.key()}: {x.shape[:2]} vs ({T},{S}).")
            if x.shape[-1] != 2:
                raise ValueError(f"Zero curve params require last dim size 2 (tenor,zero) for {mid.key()}.")

    # --- Vols ---
    for mid, p in ds.vol_params.items():
        if p.axis_names[0] != "time":
            raise ValueError(f"Vol panel first axis must be 'time' for {mid.key()}, got {p.axis_names}.")
        factory = ds.vol_factories[mid]
        x = np.asarray(p.data)

        if isinstance(factory, GridVolFactory):
            exp = np.asarray(factory.expiries, dtype=float).reshape(-1)
            k = np.asarray(factory.strikes, dtype=float).reshape(-1)

            if x.ndim != 4:
                raise ValueError(f"GridVolFactory expects vol panel ndim=4 for {mid.key()}, got {x.ndim}.")
            if len(p.axis_names) < 4:
                raise ValueError(f"Vol panel axis_names too short for {mid.key()}: {p.axis_names}")
            if p.axis_names[1] != "scenario":
                raise ValueError(f"Vol panel must have scenario axis at dim=1 for {mid.key()}.")
            if x.shape[0] != T or x.shape[1] != S:
                raise ValueError(f"Vol panel [T,S,...] mismatch for {mid.key()}: {x.shape[:2]} vs ({T},{S}).")
            if x.shape[2] != exp.size or x.shape[3] != k.size:
                raise ValueError(
                    f"Vol grid shape mismatch for {mid.key()}: got (n_exp,n_k)=({x.shape[2]},{x.shape[3]}) "
                    f"expected ({exp.size},{k.size})."
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
) -> MarketDataset:
    """
    Thin builder/wirer for MarketDataset. Keeps examples clean and ensures a consistent
    construction path across the library.

    Parameters
    ----------
    validate:
        If True, runs MarketDataset.__post_init__ (always) plus validate_dataset_layout(ds).
    """
    ds = MarketDataset(
        dates=dates,
        n_scenarios=int(n_scenarios),
        panels=quote_panels,
        curve_params=curve_param_panels,
        curve_factories=curve_factories,
        vol_params=vol_param_panels,
        vol_factories=vol_factories,
        meta=meta,
    )

    if validate:
        validate_dataset_layout(ds)

    return ds