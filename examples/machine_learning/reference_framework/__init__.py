"""
Reference ML Framework — production-grade template for hedge fund ML infrastructure.

Standalone implementation of features, validation, monitoring, and ensemble
orchestration. Does not modify the QuantStrata library.

Modules:
    features:   Central feature definitions, transforms, and GNN-specific builders
    validation: Input validation and deployment gates
    monitoring: Drift baselines, drift detection, prediction logging
    ensemble:   Multi-model registry, routers, and orchestration
"""

__version__ = "1.0.0"
