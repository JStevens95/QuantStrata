from __future__ import annotations

import pytest

from src.orchestrator.core.registry import PipelineRegistry
from src.orchestrator.core.pipeline import Pipeline


def test_registry_register_and_get() -> None:
    reg = PipelineRegistry()

    def builder(cfg: object) -> Pipeline:
        return Pipeline(name="x", steps=[])

    reg.register("p", builder)
    assert reg.get("p") is builder
    assert reg.names() == ("p",)


def test_registry_rejects_duplicate_names() -> None:
    reg = PipelineRegistry()

    def builder(cfg: object) -> Pipeline:
        return Pipeline(name="x", steps=[])

    reg.register("p", builder)
    with pytest.raises(Exception):
        reg.register("p", builder)