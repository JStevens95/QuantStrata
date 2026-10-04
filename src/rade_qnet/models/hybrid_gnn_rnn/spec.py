"""
Specifications for the hybrid GNN-RNN model and its data build.

Every number the flagship needs that is not a weight lives here, validated at
load time rather than discovered at layer-construction time. The original
carried roughly sixty configuration fields across six nested dataclasses with
no validation; a typo in a layer's ``units`` surfaced as a shape error
somewhere inside a forward pass, several minutes into a run.

What is *not* here
------------------
Sequence length, split fractions, batch size, batch shuffling, scaling
method and whether the reduction basis may see held-out rows are all
framework settings, carried on the ``SourceSpec`` this module is given.
They are deliberately not repeated here.

An earlier draft did repeat them, and that was a mistake worth naming: two
fields meaning the same thing is one field and a bug waiting for someone to
set the wrong one. It also meant re-implementing machinery the framework
already provides and tests -- an explicit-split strategy for parity replay,
a leakage annotation written into the run lineage, and the structural
separation of batch order from scenario split that fixes defect 3.

The one compatibility flag that *is* model-specific is
``AttributeEncoderSpec.numeric_precision``, with its twin
``GraphSpec.precision``. Both default to the correct behaviour, so
forgetting to set one fails safe.

Defect 9 -- the reduction basis being selected over the full scaled history
-- is reproduced through the framework's ``transforms.reduction.fit_on``,
which already defaults to ``"train"`` and already writes a leakage warning
into the lineage when set to ``"all"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ...core.spec.base import Spec

#: Activations the layers accept, matching the set the original supported.
#: A literal rather than a free string, so a typo is a specification error
#: rather than a crash several minutes into a run.
ActivationName = Literal["relu", "leaky_relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"]

__all__ = [
    "ActivationName",
    "AttributeEncoderSpec",
    "GraphSpec",
    "HybridDataSpec",
    "HybridModelSpec",
]


class AttributeEncoderSpec(Spec):
    """
    How instrument attributes become a feature matrix.

    Fitted along the **entity axis** over the full universe, which is
    legitimate rather than leakage: which instruments exist, and what their
    attributes are, is known before any P&L is observed. Restricting the
    encoder to training-period instruments would discard information a
    trader has rather than information they lack.

    Parameters
    ----------
    numeric_keys
        Attributes scaled to zero mean and unit variance.
    categorical_keys
        Attributes one-hot encoded.
    multi_label_keys
        Attributes holding a list per instrument, binarised and then
        row-normalised so an instrument with three risk factors is not
        three times as far from the origin as one with a single factor.
    maturity_key
        Which numeric attribute the decay features are derived from.
    numeric_precision
        **Compatibility flag.** Which precision the numeric attributes are
        rounded to and scaled in. The original rounded its attributes to
        ``float32``, computed the mean and standard deviation in ``float64``
        over those rounded values, then downcast both statistics back to
        ``float32`` to centre and scale -- which loses roughly six decimal
        digits for no benefit on a matrix of a few hundred rows.
        ``"float64"`` is correct and is the default; ``"float32"`` reproduces
        the original exactly and is used by the parity tests only.
    n_decay_terms
        How many exponential-decay features to derive from maturity. Each
        is ``exp(-lambda * tau)`` for a lambda on a linear grid from 10.0
        down to 0.1, which gives the model a short, a medium and a long view
        of time to maturity rather than one raw number whose relationship to
        value is strongly non-linear.
    """

    numeric_keys: tuple[str, ...] = ("moneyness", "yrs_to_maturity", "delta", "vega")
    categorical_keys: tuple[str, ...] = ("product_type", "product_subtype", "trade_type")
    multi_label_keys: tuple[str, ...] = ("underlying_risk_factors",)
    maturity_key: str = "yrs_to_maturity"
    numeric_precision: Literal["float64", "float32"] = "float64"
    n_decay_terms: int = Field(default=3, ge=0, le=16)

    @model_validator(mode="after")
    def _check_maturity_key_is_numeric(self) -> AttributeEncoderSpec:
        """
        Reject a maturity key that is not among the numeric attributes.

        Returns
        -------
        AttributeEncoderSpec
            The validated spec.

        Raises
        ------
        ValueError
            If decay terms are requested from an attribute that is not
            numeric. Unchecked, the decay features would be computed from
            whatever that attribute held and the failure would surface as an
            unhelpful cast error inside the encoder.
        """
        if self.n_decay_terms and self.maturity_key not in self.numeric_keys:
            raise ValueError(
                f"maturity_key={self.maturity_key!r} must be one of numeric_keys "
                f"{self.numeric_keys} for its {self.n_decay_terms} decay term(s) to "
                f"be computable"
            )
        return self


class GraphSpec(Spec):
    """
    How the instrument graph is built from encoded attributes.

    Also fitted along the entity axis over the full universe, for the same
    reason the encoder is.

    The ``alpha_*`` weights scale each attribute group before distances are
    measured. They are what stop a one-hot product type with eight levels
    from dominating a single scaled delta purely by occupying more columns.

    Parameters
    ----------
    n_neighbours
        How many neighbours each instrument connects to. Clamped down at
        build time when the universe is smaller than this.
    distance_metric
        How distance between two instruments' attribute vectors is measured.
    include_quota
        Whether a target instrument's neighbourhood is required to contain a
        minimum number of elementary instruments. Without it, targets in a
        dense cluster connect only to each other and the graph carries no
        path from a target to anything that could replicate it.
    min_elementary_neighbours
        The quota's elementary minimum, when enforced.
    min_target_neighbours
        The quota's target minimum, when enforced.
    alpha_moneyness, alpha_maturity, alpha_delta, alpha_vega
        Weights on the numeric attribute groups.
    alpha_product_type, alpha_product_subtype
        Weights on the categorical groups.
    alpha_underlying, alpha_underlying_risk_factors
        Weights on the underlying and its risk-factor set.
    precision
        **Compatibility flag**, and the twin of
        :attr:`AttributeEncoderSpec.numeric_precision`. The original summed
        each row's edge weights and took their reciprocal in ``float32``,
        which costs a last bit on every edge. ``"float64"`` is correct and
        is the default; ``"float32"`` reproduces the original. The parity
        tests set this and the encoder's together, because they are one
        decision -- reproduce the baseline's feature arithmetic -- expressed
        in the two places it applies.
    """

    n_neighbours: int = Field(default=5, ge=1)
    precision: Literal["float64", "float32"] = "float64"
    distance_metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"
    include_quota: bool = False
    min_elementary_neighbours: int = Field(default=2, ge=0)
    min_target_neighbours: int = Field(default=1, ge=0)

    alpha_moneyness: float = Field(default=1.0, ge=0.0)
    alpha_maturity: float = Field(default=1.0, ge=0.0)
    alpha_delta: float = Field(default=1.0, ge=0.0)
    alpha_vega: float = Field(default=1.0, ge=0.0)
    alpha_product_type: float = Field(default=1.0, ge=0.0)
    alpha_product_subtype: float = Field(default=0.5, ge=0.0)
    alpha_underlying: float = Field(default=1.0, ge=0.0)
    alpha_underlying_risk_factors: float = Field(default=0.5, ge=0.0)


class HybridDataSpec(Spec):
    """
    The model-specific half of the flagship's data build.

    Carried in ``ModelSourceSpec.params`` and validated here. Everything the
    framework already owns -- splitting, sequence length, scaling, batching,
    caching -- stays on the source spec and is not repeated.

    Parameters
    ----------
    directory
        Folder holding this cluster's P&L, attributes and universe. Carried
        here rather than on the source spec because ``ModelSourceSpec`` has
        no path: a model's raw input is not always one file, and for this
        model it is five.
    variance_threshold
        How much of each group's variance the selected basis must span.
        Higher keeps more instruments. Not expressible as the framework's
        ``reduction.n_components``, because the count is derived per group
        rather than fixed across the book.
    weight_tail
        Emphasis placed on tail scenarios when choosing the basis. One
        weights every scenario equally. Above one, a basis is chosen to span
        the days the book actually moved rather than the quiet majority.
    scale_targets
        Whether target P&L is standardised. Recorded on the fitted state so
        that inverting a prediction cannot guess wrong.
    encoder
        Attribute encoding settings.
    graph
        Graph construction settings.
    """

    directory: Path | None = None
    variance_threshold: float = Field(default=0.99, gt=0.0, le=1.0)
    weight_tail: float = Field(default=1.0, ge=1.0)
    scale_targets: bool = True
    encoder: AttributeEncoderSpec = Field(default_factory=AttributeEncoderSpec)
    graph: GraphSpec = Field(default_factory=GraphSpec)


class HybridModelSpec(Spec):
    """
    The network's shape.

    Deliberately flat. The original nested six layer configurations two
    levels deep, each with a ``general`` and a ``parameters`` sub-dictionary,
    which made the common case -- "make every layer wider" -- a six-place
    edit. Here the widths that are almost always equal share one field and
    the per-layer overrides are optional.

    Parameters
    ----------
    units
        The width used by every block that does not override it. One number,
        because tuning these independently is rare and tuning them together
        is constant.
    gnn_units, rnn_units, fusion_units, attention_units, projection_units
        Per-block overrides. ``None`` means ``units``.
    gnn_layers
        How many message-passing rounds. Each round widens an instrument's
        receptive field by one hop, so this is the radius of the
        neighbourhood the model can see.
    gnn_type
        Which message-passing rule. ``mixed_graph_sage`` concatenates an
        instrument with the mean *and* the maximum over its neighbours, so
        the layer sees both the typical neighbour and the extreme one --
        which for a risk graph is the one that matters.
        ``graph_sage`` uses a single aggregate.
    gnn_aggregation
        Which aggregate ``graph_sage`` uses. Ignored by
        ``mixed_graph_sage``, which uses both.
    gnn_activation
        Non-linearity between message-passing rounds and after the residual
        add. The sublayers themselves are linear, so this is applied by the
        block rather than inside each round.
    gnn_normalise
        Whether to normalise each round's output. Message passing repeatedly
        averages, which shrinks activations towards the graph mean; without
        normalisation a deep stack converges to a constant and the
        instruments become indistinguishable.
    rnn_type
        Which recurrence encodes the P&L history. ``bilstm`` doubles the
        output width, which the fusion block accounts for.
    rnn_layers
        Stacked recurrent layers.
    fusion_mode
        How the attended graph stream and the recurrent stream are
        combined. ``gate`` learns a per-feature blend; ``add`` sums them.
        A gate is the default because the right blend is not the same for
        every instrument -- a liquid instrument's own history is worth more
        than its neighbours', and a thinly traded one's is worth less.
    fusion_heads, attention_heads
        How many attention heads each block uses. Defaulted to the values
        the original ran in production rather than to a textbook four: a
        refactor that silently changes a hyperparameter is no longer
        comparable with the model it replaced, and that is a modelling
        decision to take deliberately rather than inherit from a default.
    attention_activation
        Non-linearity inside the target block's feed-forward network.
    attention_normalise
        Whether the target block normalises after each sublayer. A
        transformer block without it is unstable at any depth.
    neighbour_cap
        Most neighbours any node attends over. Bounds attention at
        ``O(n·k)`` rather than ``O(n²)``, which is what keeps a large
        universe tractable.
        Heads in the target-attention block.
    projection_activation
        Non-linearity inside the output head's residual network.
    baseline_weight_norm
        Whether each target's baseline kernel is split into a unit-norm
        direction and a positive magnitude. Decoupling the two makes the
        gradient on direction independent of how large the kernel has
        grown, which keeps a target whose baseline has learnt a big
        amplitude from also learning slowly.
    new_target_blend
        How an instrument with no fitted baseline borrows one from the
        instruments that have: by softmaxed cosine similarity in attribute
        space, or by inverse distance.
    new_target_neighbours
        How many fitted instruments a new one borrows from.
    new_target_temperature
        Sharpness of the cosine blend. Larger concentrates the borrow on
        the single closest instrument.
    new_target_distance_power
        Exponent on the inverse-distance blend.
    new_target_residual_damping
        Scales the residual correction for new instruments. Below one
        because the residual network never saw them, so its correction is
        extrapolation rather than fit.
    dropout
        Applied in the GNN, attention and projection blocks.
    use_residual
        Whether message-passing rounds carry a residual connection. With
        several rounds and no residual, early-layer signal is lost.
    """

    units: int = Field(default=16, ge=1)
    gnn_units: int | None = Field(default=None, ge=1)
    rnn_units: int | None = Field(default=None, ge=1)
    fusion_units: int | None = Field(default=None, ge=1)
    attention_units: int | None = Field(default=None, ge=1)
    projection_units: int | None = Field(default=None, ge=1)

    gnn_layers: int = Field(default=2, ge=1)
    gnn_type: Literal["graph_sage", "mixed_graph_sage"] = "mixed_graph_sage"
    gnn_aggregation: Literal["mean", "max"] = "mean"
    gnn_activation: ActivationName = "relu"
    gnn_normalise: bool = True
    rnn_type: Literal["lstm", "gru", "bilstm"] = "lstm"
    rnn_layers: int = Field(default=2, ge=1)
    fusion_mode: Literal["gate", "add"] = "gate"
    fusion_heads: int = Field(default=1, ge=1)
    neighbour_cap: int = Field(default=50, ge=1)
    attention_heads: int = Field(default=1, ge=1)
    attention_activation: ActivationName = "tanh"
    attention_normalise: bool = True
    projection_activation: ActivationName = "gelu"
    baseline_weight_norm: bool = False
    new_target_blend: Literal["cosine_softmax", "inverse_distance"] = "cosine_softmax"
    new_target_neighbours: int = Field(default=5, ge=1)
    new_target_temperature: float = Field(default=5.0, gt=0.0)
    new_target_distance_power: float = Field(default=2.0, gt=0.0)
    new_target_residual_damping: float = Field(default=1.0, ge=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_residual: bool = True

    def width(self, block: str) -> int:
        """
        Return the width for one block, falling back to the shared default.

        Parameters
        ----------
        block
            Block name: ``gnn``, ``rnn``, ``fusion``, ``attention`` or
            ``projection``.

        Returns
        -------
        int
            The resolved width.

        Raises
        ------
        ValueError
            If the block name is not one of the five. Raised rather than
            silently returning the default, because a typo would otherwise
            produce a model that is quietly the wrong shape.
        """
        override = getattr(self, f"{block}_units", _MISSING)
        if override is _MISSING:
            raise ValueError(
                f"unknown block {block!r}; expected one of gnn, rnn, fusion, attention, projection"
            )
        return self.units if override is None else int(override)


#: Sentinel distinguishing "no override set" from "block does not exist".
#: `None` cannot serve, because `None` is the legitimate value meaning
#: "fall back to the shared width".
_MISSING = object()
