"""
PipelineRegistry.

We keep a registry so:
- user code can refer to pipelines by name (string)
- discovery can register builtins in one place
- no import side effects are required to "find" pipelines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from src.orchestrator.core.errors import ConfigError
from src.orchestrator.core.pipeline import Pipeline

# Pipeline builder signature:
# - input: RunConfig (typed object, but kept as `object` here to avoid circular imports)
# - output: Pipeline (steps composed)
PipelineBuilder = Callable[[object], Pipeline]


@dataclass(slots=True)
class PipelineRegistry:
    """
    In-memory mapping from pipeline_name -> pipeline_builder.
    """
    _builders: Dict[str, PipelineBuilder] = field(default_factory=dict)

    def register(self, name: str, builder: PipelineBuilder) -> None:
        """
        Register a new pipeline builder.

        Raises
        ------
        ConfigError
            If the name is empty or already registered.
        """
        pipeline_name = str(name).strip()
        if not pipeline_name:
            raise ConfigError("Pipeline name must be a non-empty string.")
        if pipeline_name in self._builders:
            raise ConfigError(f"Pipeline already registered: {pipeline_name}")
        self._builders[pipeline_name] = builder

    def get(self, name: str) -> PipelineBuilder:
        """
        Retrieve a builder by name.

        Raises
        ------
        ConfigError
            If the name is unknown.
        """
        pipeline_name = str(name).strip()
        if pipeline_name not in self._builders:
            raise ConfigError(f"Unknown pipeline: {pipeline_name}")
        return self._builders[pipeline_name]

    def names(self) -> Tuple[str, ...]:
        """Return registered pipeline names in stable sorted order."""
        return tuple(sorted(self._builders.keys()))