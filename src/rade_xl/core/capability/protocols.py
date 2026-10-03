"""
Opt-in capabilities a model may implement.

Every protocol here is ``runtime_checkable``, and the framework tests for them
with ``isinstance``. That choice is deliberate and has three consequences
worth being explicit about.

**A model never pays for a feature it does not use.** A simple regressor does
not implement :class:`StaticInputs`, so the engine never asks about device
placement for static tensors. There is no base-class method to stub out and no
``None`` to return.

**Adding a capability cannot break an existing model.** A new protocol is new
behaviour for models that implement it and a no-op for every model that does
not. Compare the alternative -- a growing abstract base -- where each addition
is a breaking change to every subclass in existence.

**The capability is visible in the type, not in a flag.** The framework never
reads ``model.has_static_inputs``. It asks whether the model satisfies the
protocol, which cannot disagree with what the model actually implements.

A caveat on runtime checking
----------------------------
``isinstance`` against a runtime-checkable protocol verifies *method presence*,
not signatures. A model with a ``static_inputs`` method of the wrong shape
passes the check and then fails when called. This is why
``rade_xl.testkit.conformance`` exists: it calls each capability it detects and
checks what comes back. The ``isinstance`` test routes; the conformance suite
verifies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from ..contract.data import Batch, TensorLike

__all__ = [
    "CustomStep",
    "Inductive",
    "Precomputable",
    "Routable",
    "StaticInputs",
]


@runtime_checkable
class StaticInputs(Protocol):
    """
    The model consumes inputs that are constant across every batch.

    An adjacency matrix, an entity feature table, an index array. Implementing
    this lets the engine move them to the device once, at the start of
    training, rather than treating them as part of each sample.

    This protocol is what retires a specific inefficiency in the
    implementation being replaced. There, static tensors were merged into
    every sample and then *compared across the batch* during collation, with
    the first returned once they were confirmed equal -- so an adjacency
    matrix was compared against itself once per sample, per batch, per epoch,
    to establish something true by construction. Declaring the inputs static
    deletes both the copying and the comparison.
    """

    def static_inputs(self) -> Mapping[str, TensorLike]:
        """
        Return the inputs that do not vary between batches.

        Called once per fit, before the first batch, and the result is held
        for the duration of the run. Implementations must therefore not
        return something that depends on batch state.

        Returns
        -------
        Mapping
            Static inputs, keyed as declared in the model's signature.
        """
        ...


@runtime_checkable
class Precomputable(Protocol):
    """
    An expensive encoding of the static inputs can be computed once and reused.

    The motivating case is a graph encoder. If the graph does not change
    between batches, neither do its node embeddings, so recomputing them per
    batch repeats identical work -- which during evaluation can dominate the
    runtime entirely.

    The framework calls :meth:`precompute` once at the start of an evaluation
    or inference pass and passes the result to each forward call. It does
    *not* do this during training, where the encoder's parameters are being
    updated and a cached embedding would be stale after the first step.
    """

    def precompute(self, static: Mapping[str, TensorLike]) -> Mapping[str, TensorLike]:
        """
        Compute the reusable encoding of the static inputs.

        Parameters
        ----------
        static
            The model's static inputs, already on the correct device.

        Returns
        -------
        Mapping
            Named intermediate tensors to be reused across the pass.
        """
        ...

    def forward_with_precomputed(
        self,
        batch: Batch,
        precomputed: Mapping[str, TensorLike],
    ) -> TensorLike:
        """
        Run a forward pass using a previously computed encoding.

        Must return the same values the ordinary forward pass would, up to
        floating-point reassociation. The conformance suite checks this
        against the model's standard forward path, because a divergence here
        means evaluation and training silently measure different models.

        Parameters
        ----------
        batch
            One batch of dynamic inputs.
        precomputed
            The output of :meth:`precompute`.

        Returns
        -------
        TensorLike
            Predictions for the batch.
        """
        ...


@runtime_checkable
class CustomStep(Protocol):
    """
    The model owns its loss computation.

    The default training step computes one loss from one output and one
    target, which covers most models and should be left alone when it does.
    Implementing this protocol takes over that computation, which is required
    when the loss is not a function of a single output: multi-objective
    training, auxiliary losses, a regulariser over the model's own
    parameters, or a loss that needs the static inputs.

    What the framework still owns
    -----------------------------
    The step returns a loss; it does not perform the update. Backward passes,
    gradient clipping, optimiser stepping, scheduler stepping, mixed-precision
    scaling and distributed synchronisation all remain the engine's
    responsibility. That boundary is what keeps a custom loss compatible with
    every execution mode -- a model that called ``backward()`` itself would
    quietly break under gradient accumulation and under distributed training.
    """

    def training_step(self, batch: Batch, static: Mapping[str, TensorLike]) -> TensorLike:
        """
        Compute the scalar loss for one batch.

        Parameters
        ----------
        batch
            One batch, carrying the dynamic inputs and the target.
        static
            The model's static inputs, already on the correct device.

        Returns
        -------
        TensorLike
            A scalar loss tensor with a gradient path back to the model's
            parameters. Returning a detached tensor produces a run that
            trains for its full epoch budget and learns nothing, so the
            conformance suite checks that gradients reach the parameters.
        """
        ...


@runtime_checkable
class Routable(Protocol):
    """
    The model can serve as one member of a job set.

    A job set runs the same model across many jobs -- one per instrument,
    region or cluster. The framework needs to know which targets a given
    member is actually responsible for, so that predictions from several
    members can be assembled into one portfolio-level result without
    overlapping or leaving gaps.

    The declaration is required rather than inferred because the honest answer
    is often narrower than the job definition. A member nominally assigned
    twelve instruments may have trained on nine, the other three having been
    dropped for insufficient history. Inferring coverage from the job
    definition would then attribute three instruments' predictions to a model
    that never saw them.
    """

    def covered_targets(self) -> Sequence[str]:
        """
        Return the targets this member actually trained on.

        Returns
        -------
        Sequence of str
            Target identifiers, which must be a subset of those the job
            assigned. The framework checks this and reports any target that
            was assigned to no member.
        """
        ...


@runtime_checkable
class Inductive(Protocol):
    """
    The model can predict for entities absent during training.

    A model that learned a per-entity embedding table cannot do this: a new
    instrument has no row. A model that learned a function of entity
    *features* can, provided those features are available.

    The distinction is declared rather than attempted because the failure mode
    of getting it wrong is quiet. Asking a transductive model about an unseen
    entity tends to yield a default embedding and a confident, meaningless
    prediction -- rather than an error. Declaring the capability lets the
    inference pipeline reject the request instead.
    """

    def supports_unseen_entities(self) -> bool:
        """
        Return whether unseen entities can be predicted for.

        A method rather than a class-level constant because the answer can
        depend on how the instance was configured -- the same class may use
        an embedding table or a feature encoder according to its spec.

        Returns
        -------
        bool
            True if the model generalises to entities it has not seen.
        """
        ...

    # No companion method that *performs* the prediction, deliberately.
    #
    # An earlier draft of this protocol carried a `resolve_unseen` hook so
    # the pipeline would have something to call. It was removed before it
    # shipped, because designing it honestly needs a concrete scenario this
    # framework does not yet have: an entity absent from training is also
    # absent from the fitted state a reloaded model applies, so it has no
    # node, no index and no encoded attributes, and a hook handed only
    # identifiers and the existing static tensors has nothing to build from.
    #
    # The useful signature is therefore not knowable yet, and a protocol
    # method nothing calls and nothing implements is worse than its absence:
    # it reads as a supported path. The declaration alone already pays for
    # itself -- it is what lets the inference pipeline refuse a transductive
    # model rather than return a default embedding -- and the mechanism
    # belongs with the universe handling that can supply the missing rows.
