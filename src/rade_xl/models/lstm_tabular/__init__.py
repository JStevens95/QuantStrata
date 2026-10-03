"""
A recurrent network with no graph: the flagship's control group.

Tier
----
1 -- a custom architecture, but the framework's own data module and no
custom fitted state. A custom ``nn.Module`` does not by itself move a model
up a tier; needing your own *data build* does.

Why this is a measurement rather than a model
-----------------------------------------------
Nobody would deploy this. It is the most informative of the three baselines
anyway, because it is the only one that isolates a single variable.

The flagship fuses two streams: a graph that encodes *what each instrument
is*, and a recurrence that encodes *what the book just did*. This package is
the second stream alone, on the same data, through the same engine, with the
same loss. The gap between them is therefore an estimate of what the graph
contributes, and it is a number that has never been measured -- it is
assumed every time the graph is maintained, debugged, or justified to
somebody.

The ridge and tree baselines cannot answer that question, because they
differ from the flagship in several ways at once. A graph model that beats a
tree has shown only that *something* about it helps. A graph model that
beats this has shown that the graph does.

Why it is not simply the flagship with the graph switched off
---------------------------------------------------------------
A flag on ``HybridGnnRnn`` would be cheaper to write and worse in two ways.
It would put a branch in the flagship's forward pass that exists only for an
experiment, and every reader of that file afterwards would have to hold two
architectures in mind. And a disabled graph still leaves the fusion layer,
the target attention and their parameters in place, so the comparison would
measure "the graph's inputs" rather than "the graph".

A separate model keeps the flagship honest and makes the comparison mean
what it says.

Importing this package registers the model under ``"lstm_tabular"``.
"""

from .model import LstmTabular
from .register import LstmTabularModel
from .spec import LstmTabularSpec

__all__ = ["LstmTabular", "LstmTabularModel", "LstmTabularSpec"]
