"""
Portfolio → GNN inputs for the generic ML pipeline.

Re-exports build_gnn_dataset_from_portfolio and gnn_inputs_to_tf_dataset
from gnn_rnn_hybrid/dataset_utils so roadmap-style imports work:

    from src.machine_learning.data.portfolio import (
        build_gnn_dataset_from_portfolio,
        gnn_inputs_to_tf_dataset,
    )
"""

from __future__ import annotations

from src.machine_learning.data.gnn_rnn_hybrid.dataset_utils import (
    build_gnn_dataset_from_portfolio,
    gnn_inputs_to_tf_dataset,
)

__all__ = [
    "build_gnn_dataset_from_portfolio",
    "gnn_inputs_to_tf_dataset",
]
