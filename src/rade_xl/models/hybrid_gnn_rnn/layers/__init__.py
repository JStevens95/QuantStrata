"""
The architectural blocks of the hybrid graph-temporal network.

One block per file, each independently testable in isolation: given an input
of a known shape, assert the output shape, the gradient flow and the
invariances the block is supposed to have. Testing an architecture only
end-to-end makes every shape bug a bisection exercise.

Modules
-------
``gnn.py``
    Propagates each instrument's attributes along the similarity graph, so
    its representation reflects its neighbourhood and not only itself.
``rnn.py``
    Reads the recent window of elementary P&L into one fixed-width summary
    of the current regime.
``fusion.py``
    Crosses the two streams: graph-masked attention plus a learned gate,
    producing one vector per instrument per sample.
``attention.py``
    Lets the target instruments attend to each other over the target
    sub-graph, so correlated targets are predicted consistently.
``projection.py``
    Turns each target's representation into a P&L number: a per-target
    fitted baseline plus a shared residual correction, with unfitted
    targets borrowing a baseline from their nearest neighbours.
"""

from .attention import TargetAttentionLayer
from .fusion import FusionLayer
from .gnn import GnnBlock, GraphSage, MixedGraphSage, activation_function
from .projection import ProjectionLayer
from .rnn import RnnBlock

__all__ = [
    "FusionLayer",
    "GnnBlock",
    "GraphSage",
    "MixedGraphSage",
    "ProjectionLayer",
    "RnnBlock",
    "TargetAttentionLayer",
    "activation_function",
]
