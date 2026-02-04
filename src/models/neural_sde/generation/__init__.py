"""
Generative components for Neural SDEs.

Provides:
- Conditional path generation
- Scenario generation
- Data augmentation

Example:
    from src.models.neural_sde.generation import PathGenerator
    
    generator = PathGenerator(sde_model)
    
    # Generate scenarios
    scenarios = generator.generate_scenarios(
        S0=100.0, T=1.0, n_scenarios=1000
    )
"""

from src.models.neural_sde.generation.generator import (
    PathGenerator,
    ScenarioGenerator,
    DataAugmenter,
)

__all__ = [
    "PathGenerator",
    "ScenarioGenerator",
    "DataAugmenter",
]
