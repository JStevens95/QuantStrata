from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaticProviderConfig:
    """
    Configuration for StaticProvider (dataset replay provider).

    Policies
    --------
    strict_freq:
        If True, get_timeseries(...) requires request.freq to match dataset.meta["freq"]
        if present (case-insensitive).

    strict_date_coverage:
        If True, requested start/end must be fully covered by dataset.dates.
        If False, return the intersection (clipped range) as long as it's non-empty.

    strict_scenario_coverage:
        If True, request.scenarios must be <= dataset.n_scenarios.
        If False, scenarios are clipped to dataset.n_scenarios.

    include_only_requested_ids:
        If True, returned dataset is filtered down to IDs in request.universe only.
        If False, returned dataset includes everything stored in the underlying dataset.
    """
    strict_freq: bool = True
    strict_date_coverage: bool = True
    strict_scenario_coverage: bool = True
    include_only_requested_ids: bool = True