from __future__ import annotations

from src.marketdata.providers.static.config import StaticProviderConfig


def test_static_provider_config_defaults() -> None:
    cfg = StaticProviderConfig()
    assert cfg.strict_freq is True
    assert cfg.strict_date_coverage is True
    assert cfg.strict_scenario_coverage is True
    assert cfg.include_only_requested_ids is True