"""Scenario generation: preset stress packs and historical-based shocks."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.interfaces import ScenarioPack, ScenarioShock
from src.marketdata.scenarios.shocks import CompositeShock, ParallelRateShock, SpotShock, VolShock


def preset_stress_pack(
    name: str,
    *,
    spot_id: Optional[MarketId] = None,
    vol_id: Optional[MarketId] = None,
    domestic_curve_id: Optional[MarketId] = None,
    foreign_curve_id: Optional[MarketId] = None,
) -> ScenarioPack:
    """
    Return a ScenarioPack of predefined stress scenarios.

    Preset names define shock sizes; market IDs are passed so the same preset
    can be used for different books.

    Supported presets
    -----------------
    - "spot_down_10": spot -10% (requires spot_id).
    - "spot_up_10": spot +10%.
    - "vol_up_5": vol +5% (requires vol_id).
    - "vol_up_25": vol +25%.
    - "rates_up_25bp": domestic curve +25bp (requires domestic_curve_id).
    - "rates_down_50bp": domestic curve -50bp.
    - "crisis_style": composite: spot -15%, vol +30%, domestic rates -50bp
      (requires spot_id, vol_id, domestic_curve_id).

    Parameters
    ----------
    name : str
        Preset name (e.g. "spot_down_10", "crisis_style").
    spot_id : MarketId, optional
        Required for spot presets.
    vol_id : MarketId, optional
        Required for vol presets.
    domestic_curve_id : MarketId, optional
        Required for rate / crisis presets.
    foreign_curve_id : MarketId, optional
        Optional for rate presets (e.g. "rates_up_25bp_foreign").

    Returns
    -------
    ScenarioPack
    """
    name_lower = name.strip().lower()
    scenarios: Dict[str, ScenarioShock] = {}

    if name_lower == "spot_down_10":
        if spot_id is None:
            raise ValueError("preset 'spot_down_10' requires spot_id.")
        scenarios["spot_down_10"] = SpotShock(
            name="spot_down_10",
            spot_id=spot_id,
            bump=-0.10,
            bump_mode="relative",
        )
    elif name_lower == "spot_up_10":
        if spot_id is None:
            raise ValueError("preset 'spot_up_10' requires spot_id.")
        scenarios["spot_up_10"] = SpotShock(
            name="spot_up_10",
            spot_id=spot_id,
            bump=0.10,
            bump_mode="relative",
        )
    elif name_lower == "vol_up_5":
        if vol_id is None:
            raise ValueError("preset 'vol_up_5' requires vol_id.")
        scenarios["vol_up_5"] = VolShock(
            name="vol_up_5",
            vol_id=vol_id,
            bump=0.05,
            bump_mode="relative",
        )
    elif name_lower == "vol_up_25":
        if vol_id is None:
            raise ValueError("preset 'vol_up_25' requires vol_id.")
        scenarios["vol_up_25"] = VolShock(
            name="vol_up_25",
            vol_id=vol_id,
            bump=0.25,
            bump_mode="relative",
        )
    elif name_lower == "rates_up_25bp":
        if domestic_curve_id is None:
            raise ValueError("preset 'rates_up_25bp' requires domestic_curve_id.")
        scenarios["rates_up_25bp"] = ParallelRateShock(
            name="rates_up_25bp",
            curve_id=domestic_curve_id,
            rate_shift=0.0025,
        )
    elif name_lower == "rates_down_50bp":
        if domestic_curve_id is None:
            raise ValueError("preset 'rates_down_50bp' requires domestic_curve_id.")
        scenarios["rates_down_50bp"] = ParallelRateShock(
            name="rates_down_50bp",
            curve_id=domestic_curve_id,
            rate_shift=-0.005,
        )
    elif name_lower == "crisis_style":
        if spot_id is None or vol_id is None or domestic_curve_id is None:
            raise ValueError(
                "preset 'crisis_style' requires spot_id, vol_id, and domestic_curve_id."
            )
        scenarios["crisis_style"] = CompositeShock(
            name="crisis_style",
            shocks=[
                SpotShock(
                    name="spot_down_15",
                    spot_id=spot_id,
                    bump=-0.15,
                    bump_mode="relative",
                ),
                VolShock(
                    name="vol_up_30",
                    vol_id=vol_id,
                    bump=0.30,
                    bump_mode="relative",
                ),
                ParallelRateShock(
                    name="rates_down_50bp",
                    curve_id=domestic_curve_id,
                    rate_shift=-0.005,
                ),
            ],
        )
    else:
        raise ValueError(
            f"Unknown preset name: {name!r}. "
            "Supported: spot_down_10, spot_up_10, vol_up_5, vol_up_25, "
            "rates_up_25bp, rates_down_50bp, crisis_style."
        )

    return ScenarioPack(scenarios=scenarios)


def shocks_from_historical_series(
    series_by_id: Dict[MarketId, np.ndarray],
    *,
    percentile: float = 5.0,
    use_relative: bool = True,
    horizon: int = 1,
) -> List[ScenarioShock]:
    """
    Build scenario shocks from historical series (e.g. worst 1-day move).

    For each series, computes the given percentile of period moves (e.g. 5th
    percentile = bad move) and returns a list of SpotShock, VolShock, or
    ParallelRateShock depending on MarketId type (SPOT, VOL, CURVE).

    Parameters
    ----------
    series_by_id : dict
        MarketId -> 1d array of levels (e.g. spot, vol, or rate).
    percentile : float
        Percentile of move to use (e.g. 5.0 = 5th percentile = adverse move).
    use_relative : bool
        If True, shock is relative (for spot/vol); if False, absolute (for rates).
    horizon : int
        Period length for move (e.g. 1 = 1-day move).

    Returns
    -------
    list of ScenarioShock
        One shock per MarketId. Caller can wrap in CompositeShock for one
        multi-factor "historical worst" scenario.
    """
    shocks: List[ScenarioShock] = []
    for mid, arr in series_by_id.items():
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size < horizon + 1:
            continue
        # Period change: arr[t] - arr[t-horizon] or (arr[t] - arr[t-h]) / arr[t-h]
        changes = arr[horizon:] - arr[:-horizon]
        if use_relative:
            base = arr[:-horizon]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_changes = np.where(base != 0, changes / base, 0.0)
            bump = float(np.percentile(rel_changes, percentile))
        else:
            bump = float(np.percentile(changes, percentile))

        mkt_type = getattr(mid, "mkt_type", "").upper()
        name_suffix = str(mid).replace(".", "_")[:32]
        if mkt_type == "SPOT":
            shocks.append(
                SpotShock(
                    name=f"hist_spot_{name_suffix}",
                    spot_id=mid,
                    bump=bump,
                    bump_mode="relative",
                )
            )
        elif mkt_type == "VOL":
            shocks.append(
                VolShock(
                    name=f"hist_vol_{name_suffix}",
                    vol_id=mid,
                    bump=bump,
                    bump_mode="relative",
                )
            )
        elif mkt_type == "CURVE":
            shocks.append(
                ParallelRateShock(
                    name=f"hist_rate_{name_suffix}",
                    curve_id=mid,
                    rate_shift=bump,
                )
            )
    return shocks


def composite_from_preset(
    name: str,
    *,
    spot_id: Optional[MarketId] = None,
    vol_id: Optional[MarketId] = None,
    domestic_curve_id: Optional[MarketId] = None,
    foreign_curve_id: Optional[MarketId] = None,
) -> CompositeShock:
    """
    Return a single CompositeShock for a preset (e.g. crisis_style).

    For presets that are a single composite (e.g. "crisis_style"), returns
    that CompositeShock. For single-shock presets, returns a CompositeShock
    of one element.

    Parameters
    ----------
    name : str
        Same as preset_stress_pack.
    spot_id, vol_id, domestic_curve_id, foreign_curve_id
        Same as preset_stress_pack.

    Returns
    -------
    CompositeShock
    """
    pack = preset_stress_pack(
        name,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=domestic_curve_id,
        foreign_curve_id=foreign_curve_id,
    )
    shocks_list = list(pack.scenarios.values())
    if len(shocks_list) == 1:
        single = shocks_list[0]
        if isinstance(single, CompositeShock):
            return single
        return CompositeShock(name=name, shocks=[single])
    return CompositeShock(name=name, shocks=shocks_list)
