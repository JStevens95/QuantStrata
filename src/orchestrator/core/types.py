"""
Lightweight orchestrator type aliases.

We keep these intentionally simple to avoid coupling the orchestrator core
to specific domain objects (MarketDataset, models, etc.). That keeps Vn clean.
"""

from __future__ import annotations

from typing import Dict, Tuple

# A stable identifier for a single orchestrator run.
RunId = str

# A stable identifier for a single step within a pipeline.
StepName = str

# Optional tags that can be attached to a step for grouping/filtering later.
Tags = Tuple[str, ...]

# Mutable state shared across steps (escape hatch for Vn extension).
State = Dict[str, object]