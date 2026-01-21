from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from src.marketdata.core.ids import MarketId
from src.marketdata.core.types import PanelKind
from src.marketdata.core.panel import Panel



@dataclass(frozen=True, slots=True)
class PanelSchema:
    """
    Declarative schema describing how a MarketId's data is stored in Panels.

    This is used to:
      1) validate generated arrays before storing them
      2) document expected shapes/axis_names in a single place
      3) keep generator implementations consistent across asset classes

    Notes
    -----
    - `axis_names` must match what your MarketDataset slicing/snapshot expects.
    - `shape_fn` validates shape in a flexible way (depends on config/grid sizes).
    """

    kind: PanelKind
    axis_names: Tuple[str, ...]
    shape_fn: Callable[[np.ndarray, "SyntheticSchemaRuntime"], None]


@dataclass(frozen=True, slots=True)
class SyntheticSchemaRuntime:
    """
    Runtime values available to schema validators.

    We pass these so that schemas can validate variable grid sizes, like:
      - curve tenor count
      - vol expiry/strike grid sizes
      - quote dimensions
    """
    n_time: int
    n_scenarios: int


@dataclass(frozen=True, slots=True)
class MarketSchema:
    """
    Schema for a MarketId kind produced by the synthetic engine.

    Fields
    ------
    schema_id:
        A stable identifier for logging and documentation.
    dependencies:
        MarketIds that must exist before generating this MarketId.
        (The engine will ensure topological order.)
    panel_schema:
        Declarative storage schema for validation.
    """

    schema_id: str
    dependencies: Tuple[MarketId, ...]
    panel_schema: PanelSchema


# -----------------------------------------------------------------------------
# Common schema helpers
# -----------------------------------------------------------------------------

def quote_ts_schema(*, schema_id: str) -> MarketSchema:
    """
    Standard quote schema: Panel shaped [T,S] with axis_names ("time","scenario").
    """
    def _shape(arr: np.ndarray, rt: SyntheticSchemaRuntime) -> None:
        if arr.shape != (rt.n_time, rt.n_scenarios):
            raise ValueError(f"Quote panel must be shape {(rt.n_time, rt.n_scenarios)}, got {arr.shape}.")

    return MarketSchema(
        schema_id=schema_id,
        dependencies=tuple(),
        panel_schema=PanelSchema(kind="quote", axis_names=("time", "scenario"), shape_fn=_shape),
    )


def curve_params_schema(*, schema_id: str, dependencies: Sequence[MarketId] = ()) -> MarketSchema:
    """
    Standard curve params schema: Panel shaped [T,S,K,2] with axis_names ("time","scenario","tenor","cols").

    K is variable, but cols must always be 2: [tenor, zero_rate].
    """
    def _shape(arr: np.ndarray, rt: SyntheticSchemaRuntime) -> None:
        if arr.ndim != 4:
            raise ValueError(f"Curve params must be 4D [T,S,K,2], got ndim={arr.ndim} shape={arr.shape}.")
        if arr.shape[0] != rt.n_time or arr.shape[1] != rt.n_scenarios:
            raise ValueError(f"Curve params must start with [T,S]=[{rt.n_time},{rt.n_scenarios}], got {arr.shape[:2]}.")
        if arr.shape[3] != 2:
            raise ValueError("Curve params last dim must be 2: [tenor, zero_rate].")

    return MarketSchema(
        schema_id=schema_id,
        dependencies=tuple(dependencies),
        panel_schema=PanelSchema(kind="curve_params", axis_names=("time", "scenario", "tenor", "cols"), shape_fn=_shape),
    )


def vol_params_schema_flattened(
    *,
    schema_id: str,
    dependencies: Sequence[MarketId] = (),
) -> MarketSchema:
    """
    Standard V1 grid vol params schema: Panel shaped [T,S,P] with axis_names ("time","scenario","params").

    P is variable (e.g. n_exp*n_strikes).

    Notes
    -----
    This is compatible with your existing GridVolFactory which accepts flattened params.
    FX desk-quote surfaces (ATM/RR/BF) will use a different schema in Phase 1.
    """
    def _shape(arr: np.ndarray, rt: SyntheticSchemaRuntime) -> None:
        if arr.ndim != 3:
            raise ValueError(f"Vol params must be 3D [T,S,P], got ndim={arr.ndim} shape={arr.shape}.")
        if arr.shape[0] != rt.n_time or arr.shape[1] != rt.n_scenarios:
            raise ValueError(f"Vol params must start with [T,S]=[{rt.n_time},{rt.n_scenarios}], got {arr.shape[:2]}.")
        if arr.shape[2] <= 0:
            raise ValueError("Vol params P dimension must be > 0.")

    return MarketSchema(
        schema_id=schema_id,
        dependencies=tuple(dependencies),
        panel_schema=PanelSchema(kind="vol_params", axis_names=("time", "scenario", "params"), shape_fn=_shape),
    )