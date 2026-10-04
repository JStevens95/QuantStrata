"""
Where data comes from, how it is split, and how it is batched.

The module's central decision is visible in its type list: **splitting and
batching are separate specs**.

:class:`SplitSpec` decides which scenarios are training data. :class:`LoaderSpec`
decides the order batches are visited in. They are unrelated questions, and the
implementation this framework replaces drove both from a single ``shuffle``
flag -- so ``shuffle=True``, the natural choice for training throughput, turned
a chronological split of a time series into a random one. Nothing raised, and
the validation score improved.

A second decision worth naming: :class:`ReductionSpec` carries an explicit
``fit_on``. Basis selection in the previous implementation ran over the full
scaled history, validation and test rows included, which leaks the held-out
distribution into the choice of features. The flag makes the old behaviour
reachable -- for refactor parity -- and the correct behaviour the default.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = [
    "CacheSpec",
    "ChronologicalSplitSpec",
    "ExplicitSplitSpec",
    "GroupedSplitSpec",
    "LoaderSpec",
    "ModelSourceSpec",
    "PurgedKFoldSplitSpec",
    "ReductionSpec",
    "ScalingSpec",
    "SequenceSpec",
    "SourceSpec",
    "SplitSpec",
    "TabularSourceSpec",
    "TransformsSpec",
]


class ChronologicalSplitSpec(Spec):
    """
    Split the scenario axis by time order.

    The default strategy, and the only defensible one for a time series
    without a specific reason to do otherwise.

    Parameters
    ----------
    kind
        Discriminator.
    validation_fraction, test_fraction
        Fractions of the scenario axis taken from the end, test last. Must sum
        to less than one.
    gap_scenarios
        Scenarios discarded at each boundary, in addition to the automatic gap
        a sequence length implies. Use this when a target is computed over a
        forward window, since such a target carries information from
        scenarios after its own index.
    """

    kind: Literal["chronological"] = "chronological"
    validation_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    gap_scenarios: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_fractions_leave_training_data(self) -> ChronologicalSplitSpec:
        """
        Reject fractions that leave no training scenarios.

        Returns
        -------
        ChronologicalSplitSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the fractions sum to one or more.
        """
        total = self.validation_fraction + self.test_fraction
        if total >= 1.0:
            raise SpecError(
                f"split fractions must sum to less than 1.0, received "
                f"validation_fraction={self.validation_fraction} + "
                f"test_fraction={self.test_fraction} = {total}; reduce one of them"
            )
        return self


class PurgedKFoldSplitSpec(Spec):
    """
    Cross-validation with an embargo either side of each held-out fold.

    Parameters
    ----------
    kind
        Discriminator.
    n_folds
        Number of folds.
    embargo_scenarios
        Scenarios purged either side of a held-out fold. Without an embargo,
        a fold boundary leaks in both directions, which is the specific
        failure plain k-fold has on serially correlated data.
    test_fraction
        Fraction held back from cross-validation entirely, as a final test
        set.
    """

    kind: Literal["purged_kfold"] = "purged_kfold"
    n_folds: int = Field(default=5, ge=2)
    embargo_scenarios: int = Field(default=0, ge=0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)


class GroupedSplitSpec(Spec):
    """
    Split so that every row of a group lands in exactly one split.

    Parameters
    ----------
    kind
        Discriminator.
    group_key
        Column or attribute identifying the group. Required: a grouped split
        with no group is just a random split, and silently becoming one is
        the failure this spec exists to prevent.
    validation_fraction, test_fraction
        Fractions of *groups*, not of rows.
    """

    kind: Literal["grouped"] = "grouped"
    group_key: str
    validation_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)


class ExplicitSplitSpec(Spec):
    """
    Use caller-supplied scenario indices verbatim.

    The strategy used for refactor parity: a golden fixture records the exact
    indices the previous implementation produced, and reproducing them removes
    the split from the list of things that could explain a difference.

    Parameters
    ----------
    kind
        Discriminator.
    train, validation, test
        Scenario indices. All three are required, because an explicit split
        with an inferred part is not explicit.
    """

    kind: Literal["explicit"] = "explicit"
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    @model_validator(mode="after")
    def _check_disjoint_and_non_empty(self) -> ExplicitSplitSpec:
        """
        Reject overlapping splits or an empty training set.

        Returns
        -------
        ExplicitSplitSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any index appears in more than one
            split, or train is empty.
        """
        if not self.train:
            raise SpecError("an explicit split requires at least one training index")

        splits = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        names = sorted(splits)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                shared = splits[first] & splits[second]
                if shared:
                    sample = sorted(shared)[:5]
                    raise SpecError(
                        f"explicit split indices overlap between {first} and {second}: "
                        f"{len(shared)} shared index(es), e.g. {sample}"
                    )
        return self


#: A split strategy, discriminated on ``kind``. Using a discriminated union
#: means an invalid ``kind`` reports that one field, rather than reporting four
#: failed alternatives and leaving the reader to work out which was intended.
SplitSpec = Annotated[
    ChronologicalSplitSpec | PurgedKFoldSplitSpec | GroupedSplitSpec | ExplicitSplitSpec,
    Field(discriminator="kind"),
]


class LoaderSpec(Spec):
    """
    How batches are formed and visited. Not how data is split.

    Parameters
    ----------
    batch_size
        Samples per batch.
    shuffle
        Whether to randomise batch order within the training split. This
        affects *order only*. It cannot affect which scenarios are in which
        split -- that is :class:`SplitSpec`, and a test asserts the
        independence directly.
    drop_last
        Discard a final short batch. Useful when a model requires a fixed
        batch size.
    num_workers
        Worker processes for data loading. Zero means load in the main
        process, which is the right default under a job-set process pool --
        nested pools oversubscribe the machine.
    pin_memory, persistent_workers
        Loader performance settings, meaningful only when workers are in use.
    """

    batch_size: int = Field(default=32, ge=1)
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False

    @model_validator(mode="after")
    def _check_worker_settings(self) -> LoaderSpec:
        """
        Reject worker settings that cannot take effect.

        Returns
        -------
        LoaderSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``persistent_workers`` is requested
            with no workers.
        """
        if self.persistent_workers and self.num_workers == 0:
            raise SpecError(
                "persistent_workers=True requires num_workers >= 1; "
                "with no workers there is nothing to keep alive"
            )
        return self


class ScalingSpec(Spec):
    """
    Feature and target scaling, fitted on the scenario axis.

    Fitted on training rows only. That is not configurable, because there is
    no legitimate reason to fit a scaler on held-out data.

    Parameters
    ----------
    method
        ``standard`` subtracts the mean and divides by the standard deviation.
        ``robust`` uses the median and interquartile range, which is the
        better choice for P&L series with heavy tails.
    scale_target
        Whether to scale the target as well as the features. When true, the
        fitted state must be able to invert it -- enforced by
        :class:`~rade_qnet.core.contract.state.FittedState`.
    """

    method: Literal["none", "standard", "robust"] = "standard"
    scale_target: bool = True


class ReductionSpec(Spec):
    """
    Dimensionality reduction, fitted on the scenario axis.

    Parameters
    ----------
    method
        ``basis_selection`` chooses a subset of the input series;
        ``pca`` projects onto components.
    n_components
        Target dimensionality. ``None`` lets the method choose.
    fit_on
        Which rows the reduction may observe.

        ``train``
            The default and the correct setting. Selection sees training rows
            only.
        ``all``
            Selection sees every row, including validation and test. This
            **leaks** the held-out distribution into the choice of features.
            It exists for one reason: reproducing the previous
            implementation's behaviour when verifying a refactor. It should
            never be set in a production configuration.
    """

    method: Literal["none", "basis_selection", "pca"] = "none"
    n_components: int | None = Field(default=None, ge=1)
    fit_on: Literal["train", "all"] = "train"


class SequenceSpec(Spec):
    """
    Rolling-window construction over the scenario axis.

    Parameters
    ----------
    length
        Window length. ``1`` means a non-sequential model.
    stride
        Step between consecutive windows.
    confine_to_split
        Whether a window must lie entirely within its own split.

        False -- the default -- keeps every label and relies on the
        boundary gap below to stop a window reaching into a neighbouring
        split. That is this framework's design, and
        :func:`~rade_qnet.sources.dataset.transforms.sequence.windows_stay_within`
        is the check that the gap is wide enough.

        True drops each split's first ``length - 1`` labels instead of
        requiring a gap. The two cost the same number of scenarios; the
        difference is whether they come from between the splits or from
        the front of each. Set it when the splits are given explicitly
        and inserting a gap is not an option.

    Notes
    -----
    A ``length`` above one makes the split boundary gap mandatory: a window
    starting in the training split would otherwise extend into validation, so
    the validation score would be partly a memory of training. The splitter
    derives that gap from this value, which is why the two specs are consumed
    together.
    """

    length: int = Field(default=1, ge=1)
    stride: int = Field(default=1, ge=1)
    confine_to_split: bool = False


class TransformsSpec(Spec):
    """
    The fitted transforms applied while building a dataset.

    Parameters
    ----------
    scaling, reduction, sequence
        Each transform's settings. All three default to a usable
        configuration, so a source spec need not mention them.
    """

    scaling: ScalingSpec = Field(default_factory=ScalingSpec)
    reduction: ReductionSpec = Field(default_factory=ReductionSpec)
    sequence: SequenceSpec = Field(default_factory=SequenceSpec)


class CacheSpec(Spec):
    """
    Caching of an expensive data build.

    Parameters
    ----------
    enabled
        Whether to reuse a previous build with the same fingerprint.
    directory
        Where cached builds live. ``None`` places them under the run root.
    """

    enabled: bool = False
    directory: Path | None = None


class _SourceSpecBase(Spec):
    """
    Settings shared by every source kind.

    Not part of the public union -- see :data:`SourceSpec`.
    """

    split: SplitSpec = Field(default_factory=ChronologicalSplitSpec)
    loader: LoaderSpec = Field(default_factory=LoaderSpec)
    transforms: TransformsSpec = Field(default_factory=TransformsSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)


class TabularSourceSpec(_SourceSpecBase):
    """
    A dataset read from a tabular file.

    The path a simple model takes: point at a file, name the target column,
    and the framework's standard data module does the rest.

    Parameters
    ----------
    kind
        Discriminator.
    path
        File to read. ``None`` is permitted so the spec is constructible with
        defaults for testing; a run with no path fails when the source is
        built, where the error can name the stage.
    target_column
        Column holding the target.
    feature_columns
        Columns to use as features. ``None`` means every column except the
        target, which is convenient and also a common source of surprise --
        naming them explicitly is recommended for anything that matters.
    """

    kind: Literal["tabular"] = "tabular"
    path: Path | None = None
    target_column: str = "target"
    feature_columns: tuple[str, ...] | None = None


class ModelSourceSpec(_SourceSpecBase):
    """
    A dataset built by the model's own data module.

    The path a complex model takes. The framework still owns splitting,
    transforms, loading and caching; the model owns how raw inputs become
    arrays.

    Parameters
    ----------
    kind
        Discriminator.
    params
        Settings passed to the model's data module, validated by that
        module's own spec class rather than here. This is the one place a
        specification is deliberately open-ended, and it is narrow and named.
    """

    kind: Literal["model"] = "model"
    params: Mapping[str, object] = Field(default_factory=dict)


#: A data source, discriminated on ``kind``.
SourceSpec = Annotated[TabularSourceSpec | ModelSourceSpec, Field(discriminator="kind")]
