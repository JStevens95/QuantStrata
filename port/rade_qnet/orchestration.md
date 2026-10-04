# `src/rade_qnet/orchestration`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 30 | 1227 | `41743506e557df79` |
| 2 | `serving.py` | 448 | 17345 | `c2ee35c80b1a71f6` |

---

## 1. `src/rade_qnet/orchestration/__init__.py`

1227 bytes · SHA-256 `41743506e557df79`

```python
"""
Who coordinates the work.

``orchestration`` is the conductor.  It holds the four pipelines that define the
model lifecycle, the fan-out that runs one model across a list of jobs, and the
placement layer that decides which process or GPU each job lands on.

The separation inside this package is between *sequence* and *placement*:

``pipelines``
    The ordered stages of a single run.  Deterministic and placement-agnostic.
``jobs``
    One model, many jobs -- shared defaults, per-job overrides (including
    different architecture complexity per job), and the aggregated summary.
``compute``
    Where a unit of work executes: in-process, across processes, across GPUs,
    or on a cluster.

Keeping these apart is what makes a job set reproducible: running sequentially
and running across eight processes must produce byte-identical artifacts, which
is only achievable if the pipeline cannot observe its own placement.

Dependency rule
---------------
May import: ``core``, ``sources``, ``engines``, ``storage``, ``analysis``.
May not import: ``models``.  A pipeline resolves a model through the registry,
never by importing it, so the framework never depends on the model library.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/orchestration/serving.py`

17345 bytes · SHA-256 `c2ee35c80b1a71f6`

```python
"""
Holding a saved model open, for a caller that will use it more than once.

Why the pipelines are not enough
---------------------------------
:func:`rade_qnet.api.infer` reads the bundle from disk, verifies it against
its manifest, resolves the model definition, rebuilds the architecture, loads
the weights and places the result on a device -- and then does all of that
again on the next call.  For scheduled batch scoring that is not waste, it is
the guarantee: the artefact on disk is provably the artefact that produced the
numbers, every time, with no possibility of a process having drifted.

For a service answering requests it is unusable.  So this module holds the
work open, and the split between what is cached and what is not is the whole
of its design.

What is cached, and what deliberately is not
----------------------------------------------
Cached for the life of the handle, because it depends on the bundle alone:
the opened :class:`~rade_qnet.orchestration.stages.reload.LoadedBundle` --
the disk read, the manifest verification, the definition lookup, the fitted
state -- and the rebuilt, weight-loaded model.  That is the expensive part,
and nothing about a later request can change it.

*Not* cached, unless the model has no static inputs: the prepared handle.
Static inputs are things like a graph adjacency or an entity-attribute table,
computed from the data build rather than from the bundle, and a model that
has them can see them change between requests -- a new instrument alters the
graph.  Reusing a handle prepared against last request's adjacency would
produce a confident number computed against the wrong neighbourhood, which is
the failure mode this framework spends most of its refusals avoiding.

Most models have no static inputs at all, so most get the full saving.  The
rest pay a device placement per request and keep the correctness, and the
test suite pins both halves of that behaviour.

What this does not promise
---------------------------
**Thread safety.**  A handle holds one model object and the engines mutate it
during a forward pass -- autocast state, cached encodings for a
:class:`~rade_qnet.core.authoring.capabilities.Precomputable` model.  Two
threads calling :meth:`Predictor.predict` on one instance is undefined.  The
supported pattern is one handle per worker, which is also what lets a process
pool scale without the handles contending.

**Freshness.**  A handle is a snapshot.  Promoting a new run to
``production`` does not move an already-open handle onto it, because silently
swapping the model under a running service is worse than requiring a restart:
the numbers would change with no event in the log that explains why.  Ask the
registry again and open a new handle.

The interactive counterpart
----------------------------
:class:`Agent` is the same idea for a policy, answering ``act`` rather than
``predict``.  It is a separate type rather than a flag, because a predictor is handed inputs and
returns values with provenance while an agent is handed one observation and
returns one action, with no dataset, no split and nothing to attribute a
number to. One class covering both would mean a method that sometimes took a
source and sometimes an observation, and a caller finding out which at run
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.lifecycle.components import ComponentError, get_learner
from ..core.lifecycle.errors import ContractError
from ..core.provenance.logging import get_logger
from .pipelines.infer import DEFAULT_SPLIT, InferPipeline
from .stages.reload import load_bundle, load_policy
from .stages.resolve import pipeline_for

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..core.contract.result import Predictions
    from ..core.contract.signature import PolicySignature
    from ..core.lifecycle.context import RunContext
    from ..core.spec.data import SourceSpec
    from .stages.reload import LoadedBundle, LoadedPolicy

#: The method a learner declares when it can act without exploring.
#:
#: Looked up by name rather than declared on ``PolicyLearner``, so that
#: serving does not add a contract method nothing yet honours -- the mistake
#: ``EngineCapabilities`` made for four phases.
#:
#: Looked up on the learner *type* rather than on an instance, and that is
#: not a convenience. Constructing a learner requires an optimiser built over
#: the policy's parameters, a resolved hardware block and a gradient-norm
#: tracker, none of which a served policy has or should have. The constraint
#: that follows is the right one anyway: a greedy action depends on the
#: policy's output and the action space, never on the exploration schedule or
#: the state of the optimiser, so a correct implementation never needed an
#: instance. A learner declares this as a ``staticmethod`` taking the policy
#: and one observation.
GREEDY_METHOD = "act_greedily"

__all__ = ["Agent", "Predictor"]

_LOGGER = get_logger(__name__)


class Predictor:
    """
    A saved supervised model, held open across many predictions.

    Construct one through :func:`rade_qnet.api.load` rather than directly;
    the public function resolves a tag or an alias to a directory and builds
    the run context this needs.

    Parameters
    ----------
    directory
        The bundle directory.
    context
        The run context predictions are made under. Supplies the output
        root and the identifiers that appear in the log.
    verify
        Whether to re-hash the bundle's files against its manifest. Done
        once, at open, rather than per prediction -- which is the point of
        holding the bundle rather than reopening it.

    Attributes
    ----------
    directory
        Where the bundle was loaded from.

    Examples
    --------
    >>> predictor = api.load("hybrid_gnn_rnn@production")  # doctest: +SKIP
    >>> for batch in requests:  # doctest: +SKIP
    ...     predictions = predictor.predict(source=batch)  # doctest: +SKIP
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        context: RunContext,
        verify: bool = True,
    ) -> None:
        """
        Open the bundle once and keep what a prediction needs.

        Parameters
        ----------
        directory
            The bundle directory.
        context
            The run context predictions are made under.
        verify
            Whether to verify the bundle against its manifest.
        """
        self.directory = Path(directory)
        self._context = context
        self._loaded: LoadedBundle = load_bundle(self.directory, verify=verify)
        # Populated after the first prediction, and only for a model whose
        # static inputs are empty -- see the module docstring for why a
        # model with static inputs may not reuse one.
        self._prepared: object | None = None
        _LOGGER.info("opened predictor: %s", self._loaded.describe())

    @property
    def model_name(self) -> str:
        """
        Return the name the model is registered under.

        Returns
        -------
        str
            From the manifest, so it is what was recorded rather than what
            the resolved class happens to be called now.
        """
        return self._loaded.model_name

    def describe(self) -> str:
        """
        Return a one-line description, for a log line or a health endpoint.

        Returns
        -------
        str
            Model name, bundle version and the splits it was trained on.
        """
        return self._loaded.describe()

    def predict(
        self,
        *,
        source: SourceSpec | None = None,
        split: str = DEFAULT_SPLIT,
        entities: Iterable[str] | None = None,
    ) -> Predictions:
        """
        Predict, reusing everything that does not depend on this request.

        The returned predictions carry the same provenance a one-shot
        :func:`rade_qnet.api.infer` produces -- bundle version, spec digest,
        both source fingerprints and a timestamp. A held model must not be
        cheaper to reconcile than a cold one, because reconciling after the
        fact is the ordinary case and the held path is the one that will run
        a thousand times a day.

        Parameters
        ----------
        source
            A source specification to predict over, or ``None`` for the
            bundle's own.
        split
            Which split of the source to predict over.
        entities
            Identifiers to predict for, or ``None`` for whatever the source
            holds. Naming entities the model never saw requires it to
            declare :class:`~rade_qnet.core.authoring.capabilities.Inductive`;
            otherwise the request is refused rather than answered with a
            default embedding.

        Returns
        -------
        Predictions
            Values in the target's original units, with provenance.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        pipeline_type = pipeline_for(self._loaded.definition, "infer", InferPipeline)
        with self._context.activate():
            pipeline = pipeline_type(
                context=self._context,
                directory=self.directory,
                source=source,
                split=split,
                entities=None if entities is None else tuple(entities),
                verify=False,
                preloaded=self._loaded,
                prepared=self._prepared,
            )
            predictions = pipeline.run()

        self._remember(pipeline)
        return predictions

    def _remember(self, pipeline: InferPipeline) -> None:
        """
        Keep the prepared model, but only when doing so is sound.

        A handle is reusable exactly when the static inputs it was prepared
        against cannot differ next time, and the only case where that is
        certain is when there are none. Anything cleverer -- comparing
        adjacency matrices, hashing the static block -- would be a
        correctness argument resting on a comparison, where this rests on
        there being nothing to compare.

        Parameters
        ----------
        pipeline
            The pipeline that just ran, holding the handle it restored.
        """
        if self._prepared is not None or pipeline.handle is None:
            return
        if pipeline.handle.static:
            _LOGGER.debug("%s has static inputs; preparing per prediction", self._loaded.model_name)
            return
        self._prepared = pipeline.handle


class Agent:
    """
    A saved policy, held open so it can be asked for actions.

    Construct one through :func:`rade_qnet.api.act` or
    :func:`rade_qnet.api.load`, not directly.

    A separate type from :class:`Predictor` rather than a flag on it. A
    predictor is handed a dataset and returns values with provenance; an
    agent is handed one observation and returns one action, with no split,
    no identifiers and nothing to attribute a number to. One class covering
    both would mean a method that sometimes took a ``source`` and sometimes
    an ``observation``, and a caller discovering which at run time.

    What is cached is simpler than for a predictor: a policy has no static
    inputs, so there is no case where reuse could go stale, and the whole
    rebuilt network is kept. The same two caveats apply -- a handle is not
    thread-safe, and it is a snapshot that does not follow an alias.

    Parameters
    ----------
    directory
        The bundle directory.
    verify
        Whether to re-hash the bundle's files against its manifest. Done
        once, at open.

    Attributes
    ----------
    directory
        Where the bundle was loaded from.

    Raises
    ------
    BundleError
        If the bundle holds a supervised model rather than a policy.
    """

    def __init__(self, directory: Path | str, *, verify: bool = True) -> None:
        """
        Open the bundle once and rebuild the policy from it.

        Parameters
        ----------
        directory
            The bundle directory.
        verify
            Whether to verify the bundle against its manifest.
        """
        self.directory = Path(directory)
        self._loaded: LoadedPolicy = load_policy(self.directory, verify=verify)
        # The type, not an instance: see GREEDY_METHOD for why one cannot be
        # built here and why that is the correct constraint rather than a
        # limitation to work around.
        self._learner_name = self._loaded.spec.training.learner
        self._learner_type = get_learner(self._learner_name)
        _LOGGER.info("opened agent: %s", self._loaded.describe())

    @property
    def model_name(self) -> str:
        """
        Return the name the policy is registered under.

        Returns
        -------
        str
            From the manifest.
        """
        return self._loaded.model_name

    @property
    def signature(self) -> PolicySignature:
        """
        Return the spaces the policy was built against.

        Returns
        -------
        PolicySignature
            Read off the bundle rather than off an environment -- which is
            the point of having saved them, since a served policy has no
            environment to read them from.
        """
        return self._loaded.signature

    def describe(self) -> str:
        """
        Return a one-line description, for a log or a health endpoint.

        Returns
        -------
        str
            Model name, version and both spaces.
        """
        return self._loaded.describe()

    def act(self, observation: object) -> object:
        """
        Return the action the policy takes for one observation.

        Greedily, with no exploration. A served agent is being asked what it
        believes, not being trained, and that is the opposite of the choice
        the ``fit_steps`` driver makes during training, where exploring is
        the entire point.

        That distinction is why this raises rather than falling back to
        :meth:`PolicyLearner.act`. ``act`` is the *exploratory* action --
        ``RandomPolicyLearner`` samples from the policy's output rather than
        taking its argmax, deliberately. Serving through it would mean a
        deployed policy quietly returning a different action each time it
        was asked the same question, with no symptom to notice and no entry
        in the log. Refusing is the only honest answer until a learner
        declares a greedy mode, and the refusal names the method to add.

        Parameters
        ----------
        observation
            One observation, unbatched, matching the signature's
            observation space.

        Returns
        -------
        object
            One action, matching the signature's action space.

        Raises
        ------
        ContractError
            If the observation does not match the declared space.
        ComponentError
            If the learner the policy was trained with declares no greedy
            mode. Expected until the first real algorithm lands; the
            message names ``act_greedily`` and the learner that lacks it.
        """
        self._check(observation)
        greedily = getattr(self._learner_type, GREEDY_METHOD, None)
        if greedily is None:
            raise ComponentError(
                f"the policy at {self.directory} was trained by the "
                f"{self._learner_name!r} learner, which declares no "
                f"{GREEDY_METHOD!r}. Serving a policy means acting without "
                f"exploration, and this learner's 'act' explores by design -- "
                f"using it here would return a different action each time the "
                f"same observation was presented, silently and with nothing "
                f"in the log to show it. Add a static "
                f"{GREEDY_METHOD}(policy, observation) to the learner"
            )
        return greedily(self._loaded.policy, observation)

    def _check(self, observation: object) -> None:
        """
        Refuse an observation the weights were not trained against.

        Checked here rather than left to the engine because the failure a
        shape mismatch produces downstream is not always an exception. A
        network whose first layer happens to accept the wrong width returns
        a number, and that number is indistinguishable from a real one.

        Parameters
        ----------
        observation
            What the caller passed.

        Raises
        ------
        ContractError
            If the observation's shape differs from the declared space's.
        """
        expected = tuple(self._loaded.signature.observation.shape)
        shape = getattr(observation, "shape", None)
        if shape is None or tuple(shape) == expected:
            return
        raise ContractError(
            f"the policy at {self.directory} was trained against observations "
            f"of shape {expected} but was given one of shape {tuple(shape)}; "
            f"the bundle's signature is what the weights expect and cannot be "
            f"adapted to"
        )
```

