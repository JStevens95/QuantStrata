"""
Tests for ``SupervisedModel``, the base class for supervised models.

``SupervisedModel`` exists so that a straightforward model is a short file. It
supplies ``build_data`` and ``signature``, leaving a subclass to write
``data_module`` -- one line -- and ``build_model``. Without it, every
supervised model would reimplement the same twenty lines of source-to-payload wrapping,
and each copy would be a separate opportunity to get it wrong.

The design point worth recording is that ``data_module`` is *abstract* rather
than defaulted to the built-in tabular module. It cannot be defaulted:
``core`` has an empty dependency set, so it cannot import ``sources``, and a
default would invert the dependency direction that the whole architecture
rests on. The cost is one line in every subclass; the benefit is that ``core``
stays importable without any of the layers above it.

The rest of the tests are about the errors. Every fault this base can detect
is in the *model package* -- a data module that is not one, a prepared dataset
missing a field, a source that is not a source -- so each is reported as a
component error naming the offending type. The alternative is an
``AttributeError`` several stages later, where nothing says which of the
user's methods returned the wrong thing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pytest

from src.rade_qnet.core.capability.supervised import DataModuleLike, SupervisedModel
from src.rade_qnet.core.runtime.errors import ComponentError
from src.rade_qnet.core.spec.data import TabularSourceSpec
from src.rade_qnet.core.spec.run import SupervisedRunSpec
from src.rade_qnet.sources.dataset.tabular import TabularDataModule


@pytest.fixture
def spec(tmp_path):
    """
    Build a supervised run spec over a small CSV.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    target = features.sum(axis=1)

    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "target"])
        writer.writerows(np.column_stack([features, target]).tolist())

    return SupervisedRunSpec.model_validate(
        {
            "model": "synthetic_tabular",
            "source": TabularSourceSpec(path=path),
            "reports": {"enabled": ()},
        }
    )


class CsvRegressor(SupervisedModel):
    """The whole of a supervised model's data wiring, as a subclass should write it."""

    def data_module(self, spec):
        """Return the built-in tabular module."""
        del spec
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return a stand-in model, since these tests never train."""
        del spec, signature
        return object()


class TestTheDependencyDirection:
    """Why ``data_module`` is abstract rather than defaulted."""

    def test_data_module_is_abstract(self):
        """
        Because ``core`` cannot import ``sources``.

        Defaulting it to the built-in tabular module would invert the
        dependency direction the architecture rests on -- ``core`` would then
        import a layer above it, and the one-way stack would no longer be one
        way. One line per subclass is the price.
        """
        assert "data_module" in SupervisedModel.__abstractmethods__

    def test_the_expected_interface_is_declared_structurally(self):
        """
        As a protocol, so the consumer declares what it needs.

        That is what lets a user's own data module satisfy it without
        importing anything from ``core``, and what lets ``core`` describe its
        requirement without naming the class that meets it.
        """
        assert isinstance(TabularDataModule(), DataModuleLike)

    def test_an_object_missing_a_method_does_not_satisfy_the_protocol(self):
        """
        So the check is worth making before the build runs.

        An object with ``build`` but no ``batch_sources`` would otherwise fail
        halfway through the build, after the expensive read.
        """

        class HalfModule:
            """A module that can load but not batch."""

            def build(self, spec, *, seed=0):
                """Return nothing useful."""
                del spec, seed
                return object()

        assert not isinstance(HalfModule(), DataModuleLike)


class TestBuildData:
    """The twenty lines a supervised model no longer has to write."""

    def test_one_payload_per_split(self, spec):
        """
        So a model gets the splits its configuration asked for.

        A fixed three-payload assumption would break on a run with no
        validation fraction, which is a legitimate configuration.
        """
        bundle = CsvRegressor().build_data(spec)
        assert set(bundle.splits) == {"train", "validation", "test"}

    def test_each_payload_reports_its_sample_count(self, spec):
        """
        Because it is the denominator of every averaged metric.

        A zero would make a reported mean loss infinite, and an inflated one
        would make it quietly too small.
        """
        bundle = CsvRegressor().build_data(spec)
        assert all(payload.n_samples > 0 for payload in bundle.splits.values())

    def test_the_source_is_stored_as_the_loader(self, spec):
        """
        One object rather than a payload plus a parallel source map.

        A ``DatasetSource`` satisfies both ``BatchSource`` and
        ``Iterable[Batch]``, so keeping one reference means the two cannot
        drift apart -- which they would, the first time a stage rebuilt one
        and not the other.
        """
        bundle = CsvRegressor().build_data(spec)
        loader = bundle.splits["train"].loader
        assert loader.n_samples == bundle.splits["train"].n_samples

    def test_the_signature_and_state_come_from_the_data_build(self, spec):
        """
        Not reconstructed, so the bundle describes what was actually built.

        A reconstructed signature agrees with the real one only while two
        pieces of code agree about the transforms.
        """
        bundle = CsvRegressor().build_data(spec)
        assert bundle.signature.dynamic
        assert bundle.state is not None

    def test_the_lineage_is_carried_through(self, spec):
        """
        Because it is the only record of which data this run saw.

        Lost here, a run could not be compared against another one, and
        comparison is how anyone finds out the data changed.
        """
        assert CsvRegressor().build_data(spec).lineage.source_fingerprint

    def test_the_signature_defaults_to_the_data_build_s(self, spec):
        """
        Which is right for a model that consumes everything on offer.

        A model using only some of its inputs should narrow it, so the saved
        signature describes what the model used rather than what was
        available -- but that is an override, not the default.
        """
        model = CsvRegressor()
        bundle = model.build_data(spec)
        assert model.signature(bundle) is bundle.signature


class TestComponentErrors:
    """Every detectable fault is in the model package, and is named as such."""

    def test_a_data_module_that_is_not_one_is_reported(self, spec):
        """
        Naming the type that was returned.

        The alternative is an ``AttributeError`` on ``build`` several lines
        later, which says nothing about which of the user's methods was
        wrong.
        """

        class NotAModule(CsvRegressor):
            """A model whose data module is not a data module."""

            def data_module(self, spec):
                """Return the wrong kind of thing."""
                del spec
                return object()

        with pytest.raises(ComponentError, match="data_module"):
            NotAModule().build_data(spec)

    def test_a_prepared_dataset_missing_a_field_is_reported(self, spec):
        """
        Naming the fields, because a custom data module is the likely cause.

        A user whose own module forgot to attach the lineage gets told which
        field is absent, rather than an attribute error from inside the
        bundle construction.
        """

        @dataclass
        class Incomplete:
            """A prepared dataset with no lineage."""

            signature: object = None
            state: object = None

        class BadModule:
            """A data module whose build returns the wrong shape."""

            def build(self, spec, *, seed=0):
                """Return an incomplete prepared dataset."""
                del spec, seed
                return Incomplete()

            def batch_sources(self, prepared, spec, *, seed=0):
                """Return nothing, since the build already failed."""
                del prepared, spec, seed
                return {}

        class WithBadModule(CsvRegressor):
            """A model wired to the broken module."""

            def data_module(self, spec):
                """Return the broken module."""
                del spec
                return BadModule()

        with pytest.raises(ComponentError, match="lineage"):
            WithBadModule().build_data(spec)

    def test_a_source_that_is_not_a_batch_source_is_reported(self, spec):
        """
        Naming the split and the type.

        A training loop could not consume it, and the failure without this
        check is inside the loop's first iteration -- where the message is
        about an object not being iterable.
        """

        class BadSourceModule:
            """A data module whose sources are not sources."""

            def __init__(self) -> None:
                """Hold a real module to delegate the build to."""
                self.inner = TabularDataModule()

            def build(self, spec, *, seed=0):
                """Delegate, so the prepared dataset is valid."""
                return self.inner.build(spec, seed=seed)

            def batch_sources(self, prepared, spec, *, seed=0):
                """Return something that is not a batch source."""
                del prepared, spec, seed
                return {"train": object()}

        class WithBadSources(CsvRegressor):
            """A model wired to the broken module."""

            def data_module(self, spec):
                """Return the broken module."""
                del spec
                return BadSourceModule()

        with pytest.raises(ComponentError, match="BatchSource"):
            WithBadSources().build_data(spec)
