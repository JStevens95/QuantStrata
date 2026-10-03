"""Shared fixtures for the layer tests: a small graph and matching tensors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

#: Instruments in the toy universe. Small enough to reason about by hand,
#: large enough that a head reshape or a neighbour gather has somewhere to
#: go wrong.
N_NODES = 6

#: How many of those are targets. They are the last rows, which is the
#: ordering the data module guarantees and the attention layer relies on.
N_TARGETS = 2

#: Samples per batch.
BATCH = 4

#: Width of the encoded attribute vector.
N_ATTRIBUTES = 7

#: Width of the selected elementary basis.
N_ELEMENTARY = 3

#: Window length of the P&L history.
SEQUENCE = 5

#: Layer width used throughout. Divisible by two so a multi-head test has a
#: valid head count available.
UNITS = 8


@pytest.fixture
def target_indices() -> torch.Tensor:
    """Which rows of the node table are targets."""
    return torch.arange(N_NODES - N_TARGETS, N_NODES)


@pytest.fixture
def adjacency() -> torch.Tensor:
    """
    Return a sparse ring graph with self-loops, row-normalised.

    A ring rather than a random graph because every node then has exactly
    the same degree, so a test that fails has failed for a reason other
    than one node happening to be unusual. Row-normalised because that is
    what the graph builder produces and what the layers assume.
    """
    rows, columns = [], []
    for node in range(N_NODES):
        for neighbour in (node, (node - 1) % N_NODES, (node + 1) % N_NODES):
            rows.append(node)
            columns.append(neighbour)
    indices = torch.tensor([rows, columns], dtype=torch.long)
    values = torch.full((len(rows),), 1.0 / 3.0, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N_NODES, N_NODES)).coalesce()


@pytest.fixture
def dense_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Return the same graph densified, for the dense-path comparisons."""
    return adjacency.to_dense()


@pytest.fixture
def node_features() -> torch.Tensor:
    """Return encoded attributes, one row per instrument."""
    generator = np.random.default_rng(0)
    return torch.tensor(generator.normal(size=(N_NODES, N_ATTRIBUTES)), dtype=torch.float32)


@pytest.fixture
def pnl_history() -> torch.Tensor:
    """Return a batch of elementary P&L windows."""
    generator = np.random.default_rng(1)
    return torch.tensor(generator.normal(size=(BATCH, SEQUENCE, N_ELEMENTARY)), dtype=torch.float32)
