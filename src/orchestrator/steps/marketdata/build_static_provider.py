"""
Step: BuildStaticProviderStep

Builds a StaticProvider from ctx.state["dataset"] and attaches it to ctx.provider.

This step is written defensively to handle slight differences in StaticProvider
constructor signatures across refactors (e.g., name vs _name parameter).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import ConfigError
from src.orchestrator.core.step import Step

from src.marketdata.providers.static.config import StaticProviderConfig
from src.marketdata.providers.static.provider import StaticProvider


@dataclass(slots=True)
class BuildStaticProviderStep(Step):
    """
    Build StaticProvider for a loaded MarketDataset.

    Inputs
    ------
    ctx.state["dataset"] : MarketDataset

    Outputs
    -------
    ctx.provider : StaticProvider
    ctx.state["provider"] : StaticProvider
    """

    def run(self, ctx: Context) -> Context:
        # Retrieve dataset produced by LoadDatasetStep.
        dataset = ctx.get("dataset")
        if dataset is None:
            raise ConfigError("BuildStaticProviderStep requires ctx.state['dataset'] to be set.")

        # Read step params.
        params = dict(getattr(ctx.cfg, "params", {}) or {})

        # Optional provider display name.
        provider_name = str(params.get("provider_name", "StaticProvider")).strip() or "StaticProvider"

        # Optional StaticProviderConfig overrides.
        cfg_raw = dict(params.get("static_config") or {})
        static_cfg = StaticProviderConfig(
            strict_freq=bool(cfg_raw.get("strict_freq", True)),
            strict_date_coverage=bool(cfg_raw.get("strict_date_coverage", True)),
            strict_scenario_coverage=bool(cfg_raw.get("strict_scenario_coverage", True)),
            include_only_requested_ids=bool(cfg_raw.get("include_only_requested_ids", True)),
        )

        # Build provider with best-effort compatibility across signature variants.
        provider = _build_static_provider(dataset=dataset, config=static_cfg, name=provider_name)

        # Attach provider onto Context for downstream steps/pipelines.
        ctx.provider = provider
        ctx.put("provider", provider)

        # Log the provider name in a safe way.
        prov_name = getattr(provider, "name", None)
        ctx.logger.info("Built StaticProvider | name=%s", prov_name if prov_name is not None else "<unknown>")

        return ctx


def _build_static_provider(*, dataset: object, config: StaticProviderConfig, name: str) -> StaticProvider:
    """
    Construct StaticProvider while tolerating minor signature changes.

    We try common variants:
    - StaticProvider(dataset=..., config=..., name=...)
    - StaticProvider(dataset=..., config=..., _name=...)
    - StaticProvider(dataset=..., config=...)
    """
    # Try "name" keyword first (cleaner public API).
    try:
        return StaticProvider(dataset=dataset, config=config, name=name)  # type: ignore[arg-type]
    except TypeError:
        pass

    # Try legacy/private "_name" keyword.
    try:
        return StaticProvider(dataset=dataset, config=config, _name=name)  # type: ignore[arg-type]
    except TypeError:
        pass

    # Fall back to no name parameter at all.
    return StaticProvider(dataset=dataset, config=config)  # type: ignore[arg-type]