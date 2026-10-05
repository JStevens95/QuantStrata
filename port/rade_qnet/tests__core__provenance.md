# `tests/rade_qnet/core/provenance`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 43 | `17f398962e74a13b` |
| 2 | `test_provenance_hashing.py` | 215 | 8554 | `550397124d3379c4` |
| 3 | `test_provenance_logging.py` | 215 | 8350 | `0beb7c90ec33c592` |
| 4 | `test_provenance_seeding.py` | 222 | 8306 | `164ef7aa60f3bcac` |

---

## 1. `tests/rade_qnet/core/provenance/__init__.py`

43 bytes · SHA-256 `17f398962e74a13b`

```python
"""Mirror of rade_qnet.core.provenance."""
```

---

## 2. `tests/rade_qnet/core/provenance/test_provenance_hashing.py`

8554 bytes · SHA-256 `550397124d3379c4`

```python
"""
Tests for stable hashing.

The decisive test here is :meth:`TestStability.test_digest_survives_a_fresh_interpreter`.
It is the one that rules out Python's built-in ``hash()``, whose string
randomisation differs per process. Using ``hash()`` would make every step-cache
key change between runs, so tuning would silently rebuild the same dataset on
every trial -- a defect that costs hours and announces itself only as
unexplained slowness.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.rade_qnet.core.provenance.hashing import (
    abbreviate_digest,
    canonical_json,
    digest_arrays,
    digest_file,
    digest_payload,
    digest_spec,
)
from src.rade_qnet.core.spec.run import parse_run_spec


class TestCanonicalJson:
    """Canonicalisation makes equal payloads produce equal text."""

    def test_key_order_is_irrelevant(self):
        """
        Two dictionaries written in different orders canonicalise identically.

        This is what makes a spec's digest independent of how the YAML was
        laid out, which it must be: reordering two keys in a configuration
        file is not a change to the run.
        """
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_output_is_valid_json(self):
        """Canonical output is still parseable."""
        assert json.loads(canonical_json({"b": [1, 2], "a": "x"})) == {"a": "x", "b": [1, 2]}

    def test_paths_are_rendered_portably(self):
        """
        A path is rendered in POSIX form.

        Otherwise the same spec hashes differently on Windows and Linux, and a
        cache populated on one is useless on the other.
        """
        assert "a/b/c" in canonical_json({"path": Path("a") / "b" / "c"})

    def test_sets_are_ordered(self):
        """A set has no order, so it is sorted before encoding."""
        assert canonical_json({"s": {"b", "a"}}) == canonical_json({"s": {"a", "b"}})

    def test_numpy_scalars_and_arrays_are_encoded(self):
        """Numpy values appear in specs and must not break encoding."""
        assert canonical_json({"x": np.int64(3)}) == canonical_json({"x": 3})
        assert canonical_json({"x": np.array([1.0, 2.0])}) == canonical_json({"x": [1.0, 2.0]})

    def test_an_unencodable_object_is_rejected(self):
        """
        An arbitrary object raises rather than hashing its memory address.

        Falling back on ``repr`` would embed an address, so the same payload
        would hash differently in every process -- silently.
        """
        with pytest.raises(TypeError):
            canonical_json({"x": object()})


class TestDigests:
    """Digests are stable, sensitive, and fixed-width."""

    def test_equal_payloads_digest_equally(self):
        """The property a cache key depends on."""
        assert digest_payload({"a": 1, "b": [2, 3]}) == digest_payload({"b": [2, 3], "a": 1})

    def test_different_payloads_digest_differently(self):
        """A changed value changes the key, so a stale entry is detectable."""
        assert digest_payload({"a": 1}) != digest_payload({"a": 2})

    def test_digest_is_a_sha256_hex_string(self):
        """Fixed width, which the manifest schema validates against."""
        digest = digest_payload({"a": 1})
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_equal_specs_digest_equally(self):
        """
        Two specs parsed from the same payload digest identically.

        Without this a bundle could not claim to record the configuration that
        produced it, because the same configuration would not recognise itself.
        """
        first = parse_run_spec({"model": "demo", "seed": 3}, origin="test")
        second = parse_run_spec({"seed": 3, "model": "demo"}, origin="test")
        assert digest_spec(first) == digest_spec(second)

    def test_a_changed_spec_field_changes_the_digest(self):
        """A different configuration is a different run."""
        first = parse_run_spec({"model": "demo", "seed": 3}, origin="test")
        second = parse_run_spec({"model": "demo", "seed": 4}, origin="test")
        assert digest_spec(first) != digest_spec(second)

    def test_abbreviation_prefixes_the_full_digest(self):
        """A short form for logs must still identify the same digest."""
        digest = digest_payload({"a": 1})
        assert digest.startswith(abbreviate_digest(digest))
        assert len(abbreviate_digest(digest)) == 12


class TestFileDigests:
    """File digests detect any change to contents."""

    def test_identical_contents_digest_equally(self, tmp_path):
        """Two files with the same bytes are the same file, for our purposes."""
        first, second = tmp_path / "a.bin", tmp_path / "b.bin"
        first.write_bytes(b"payload")
        second.write_bytes(b"payload")
        assert digest_file(first) == digest_file(second)

    def test_a_single_changed_byte_changes_the_digest(self, tmp_path):
        """Bundle verification depends on this sensitivity."""
        path = tmp_path / "a.bin"
        path.write_bytes(b"payload")
        before = digest_file(path)
        path.write_bytes(b"payloae")
        assert digest_file(path) != before

    def test_large_files_are_read_in_chunks(self, tmp_path):
        """
        A file larger than the chunk size digests correctly.

        Checkpoints are gigabytes, so the implementation streams. This
        confirms the streaming produces the same answer as reading at once.
        """
        path = tmp_path / "big.bin"
        payload = b"x" * (3 * (1 << 20) + 17)
        path.write_bytes(payload)
        assert digest_file(path) == hashlib.sha256(payload).hexdigest()


class TestArrayDigests:
    """Array digests account for values, shape and dtype."""

    def test_equal_arrays_digest_equally(self):
        """The same numbers in the same layout are the same data."""
        assert digest_arrays({"x": np.arange(6.0)}) == digest_arrays({"x": np.arange(6.0)})

    def test_mapping_order_is_irrelevant(self):
        """Names are folded in sorted order, so insertion order cannot matter."""
        first = digest_arrays({"a": np.zeros(2), "b": np.ones(2)})
        second = digest_arrays({"b": np.ones(2), "a": np.zeros(2)})
        assert first == second

    def test_renaming_an_array_changes_the_digest(self):
        """
        The name is part of the data.

        Two sources delivering the same numbers under different keys are not
        interchangeable, because a model looks inputs up by name.
        """
        assert digest_arrays({"a": np.zeros(2)}) != digest_arrays({"b": np.zeros(2)})

    def test_reshaping_changes_the_digest(self):
        """
        The same values in a different shape are different data.

        A digest folding only the bytes would call a feature matrix and its
        transpose identical, which would make a cache return the wrong one.
        """
        values = np.arange(6.0)
        assert digest_arrays({"x": values}) != digest_arrays({"x": values.reshape(2, 3)})

    def test_dtype_changes_the_digest(self):
        """float32 and float64 data are not interchangeable."""
        assert digest_arrays({"x": np.arange(4, dtype=np.float32)}) != digest_arrays(
            {"x": np.arange(4, dtype=np.float64)}
        )


class TestStability:
    """The property that rules out Python's salted hash()."""

    def test_digest_survives_a_fresh_interpreter(self, repository_root):
        """
        The same payload digests identically in a separate process.

        Run in a subprocess with hash randomisation explicitly enabled, which
        is what would perturb ``hash()``. If this test fails, step caches and
        bundle provenance are both unreliable across runs.
        """
        program = (
            f"import sys; sys.path.insert(0, {str(repository_root)!r});"
            "from src.rade_qnet.core.provenance.hashing import digest_payload;"
            "print(digest_payload({'model': 'demo', 'seed': 3, 'nested': {'a': [1, 2]}}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": "random", "PATH": "/usr/bin:/bin"},
        )
        in_process = digest_payload({"model": "demo", "seed": 3, "nested": {"a": [1, 2]}})
        assert completed.stdout.strip() == in_process
```

---

## 3. `tests/rade_qnet/core/provenance/test_provenance_logging.py`

8350 bytes · SHA-256 `0beb7c90ec33c592`

```python
"""
Tests for contextual logging.

Two properties are load-bearing. First, configuration never happens at import:
a library that installs a root handler when imported takes over the logging of
every application that uses it. Second, the run, job and stage identifiers are
restored on exit, so a nested stage cannot leak its name to its parent -- which
would make a job set's logs attribute work to the wrong job.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from src.rade_qnet.core.provenance.logging import (
    ROOT_LOGGER_NAME,
    apply_context_payload,
    bound_context,
    configure_logging,
    context_payload,
    current_context,
    get_logger,
)


@pytest.fixture(autouse=True)
def _clear_context():
    """
    Ensure each test starts and ends with no bound identifiers.

    The identifiers live in context variables, which are process-wide. A test
    that left one bound would silently change what a later test observes.
    """
    yield
    apply_context_payload({})


class TestLoggerNaming:
    """Logger names place every record under one hierarchy."""

    def test_loggers_live_under_the_framework_root(self):
        """One root means an application can configure the framework alone."""
        assert get_logger("rade_qnet.core.lifecycle.pipeline").name.startswith(ROOT_LOGGER_NAME)

    def test_the_src_prefix_is_stripped(self):
        """
        A module imported as ``src.rade_qnet...`` still logs as ``rade_qnet...``.

        The repository imports through ``src``, an installed distribution does
        not. Without stripping, the same module logs under two different names
        depending on how it was imported, and a log filter configured for one
        silently misses the other.
        """
        assert get_logger("src.rade_qnet.storage.bundle").name == "rade_qnet.storage.bundle"


class TestConfiguration:
    """Configuration is explicit, and does not happen on import."""

    def test_importing_installs_no_handler(self):
        """
        The framework root has no handler until asked.

        If this fails, importing ``rade_qnet`` changes the logging behaviour of
        the importing application, which is not ours to change.
        """
        fresh = logging.getLogger("rade_qnet.test_no_autoconfig")
        assert not fresh.handlers

    def test_configure_writes_to_the_given_stream(self):
        """Records reach the stream the caller nominated."""
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        get_logger("rade_qnet.test").info("hello from the test")
        assert "hello from the test" in stream.getvalue()

    def test_configure_is_idempotent_without_force(self):
        """
        Calling twice does not double every record.

        A pipeline and a job-set driver may both configure logging; duplicated
        handlers would emit each line as many times as configure was called.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        configure_logging(level=logging.INFO, stream=stream)
        get_logger("rade_qnet.test").info("once")
        assert stream.getvalue().count("once") == 1


class TestBoundContext:
    """Identifiers bind for a block and are restored afterwards."""

    def test_identifiers_are_visible_inside_the_block(self):
        """A stage's logs can be attributed while the stage runs."""
        with bound_context(run_id="r-1", job_id="EURUSD", stage="fit"):
            assert current_context() == {"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"}

    def test_identifiers_are_restored_on_exit(self):
        """Nothing leaks past the block."""
        with bound_context(run_id="r-1", stage="fit"):
            pass
        assert current_context() == {}

    def test_identifiers_are_restored_after_an_exception(self):
        """
        A failing stage does not leave its name bound.

        Otherwise every subsequent log line in the run would be attributed to
        the stage that already failed.
        """
        with pytest.raises(RuntimeError), bound_context(run_id="r-1", stage="fit"):
            raise RuntimeError("stage failed")
        assert current_context() == {}

    def test_nesting_restores_the_outer_stage(self):
        """
        An inner stage's name does not outlive it.

        This is what lets a pipeline call a sub-pipeline and still have its
        own later stages logged under its own name.
        """
        with bound_context(run_id="r-1", stage="outer"):
            with bound_context(stage="inner"):
                assert current_context()["stage"] == "inner"
            assert current_context()["stage"] == "outer"

    def test_partial_binding_leaves_other_identifiers_alone(self):
        """Binding a stage does not clear the run it belongs to."""
        with bound_context(run_id="r-1", job_id="EURUSD"), bound_context(stage="fit"):
            assert current_context() == {"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"}


class TestRecordDecoration:
    """Bound identifiers reach the emitted record."""

    def test_identifiers_appear_in_the_output(self):
        """
        A log line from arbitrary code carries the run it came from.

        This is the whole point of using context variables: a model's own
        logging is attributed without the model knowing the framework exists.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        with bound_context(run_id="r-42", job_id="USDJPY", stage="build_data"):
            get_logger("rade_qnet.test").info("building")
        output = stream.getvalue()
        assert "r-42" in output
        assert "USDJPY" in output
        assert "build_data" in output

    def test_records_without_a_context_still_emit(self):
        """
        An unbound record is not dropped and does not raise.

        Logging happens during import and during teardown, when no run is
        active. A filter that required the identifiers would lose exactly the
        records emitted when something went wrong early.
        """
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, force=True)
        get_logger("rade_qnet.test").info("no context here")
        assert "no context here" in stream.getvalue()


class TestContextPayload:
    """The payload is how identifiers cross a process boundary."""

    def test_payload_round_trips(self):
        """
        Identifiers survive export and re-application.

        A worker process inherits nothing, so the parent exports a payload and
        the worker applies it. Without this, every log line from a parallel
        job set is unattributable.
        """
        with bound_context(run_id="r-9", job_id="GBPUSD", stage="fit"):
            payload = context_payload()
        assert current_context() == {}
        apply_context_payload(payload)
        assert current_context() == {"run_id": "r-9", "job_id": "GBPUSD", "stage": "fit"}

    def test_an_empty_payload_clears_the_context(self):
        """Applying nothing is how a worker resets between jobs."""
        apply_context_payload({"run_id": "r-1"})
        apply_context_payload({})
        assert current_context() == {}

    def test_applying_replaces_rather_than_merges(self):
        """
        A reused worker does not retain a finished job's identifier.

        A process pool reuses workers. If applying a payload merged into the
        existing context, a worker that handled ``EURUSD`` and was then given
        a run with no job identifier would keep logging ``EURUSD`` --
        attributing work to a job that had already completed.
        """
        apply_context_payload({"run_id": "r-1", "job_id": "EURUSD", "stage": "fit"})
        apply_context_payload({"run_id": "r-2"})
        assert current_context() == {"run_id": "r-2"}

    def test_payload_is_json_encodable(self):
        """
        The payload is plain strings.

        It travels through whatever a process pool uses to serialise, so it
        must not contain anything exotic.
        """
        with bound_context(run_id="r-1", job_id="EURUSD"):
            assert json.loads(json.dumps(context_payload())) == context_payload()
```

---

## 4. `tests/rade_qnet/core/provenance/test_provenance_seeding.py`

8306 bytes · SHA-256 `164ef7aa60f3bcac`

```python
"""
Tests for seeding and determinism.

Two designs are under test here, and both were chosen over an obvious
alternative.

``determinism`` is three-valued (``off``, ``warn``, ``strict``) rather than a
boolean, because forcing deterministic kernels has a real performance cost and
some operations have no deterministic implementation at all. A boolean hides
that trade-off; the implementation this replaces set it globally inside a
swallowed ``try``/``except``, so a run could silently be non-deterministic
while claiming otherwise.

Per-job seeds are *derived by hashing*, not by adding an index. That gives
order independence -- a single failed job re-run alone reproduces -- and avoids
the correlation that consecutive integer seeds can produce.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.provenance.seeding import (
    derive_seed,
    register_seeder,
    registered_seeders,
    seed_everything,
    unregister_seeder,
)


@pytest.fixture(autouse=True)
def _clear_seeders():
    """
    Remove any seeder a test registered.

    The registry is module-level, so a leftover seeder would be invoked by
    every later test -- and the failure would appear in whichever test ran
    next, not the one at fault.
    """
    before = set(registered_seeders())
    yield
    for name in set(registered_seeders()) - before:
        unregister_seeder(name)


class TestSeedEverything:
    """Seeding is reproducible and reports what it applied."""

    def test_the_applied_seed_is_returned(self):
        """
        The caller learns what was actually used.

        Recorded in the training result, so a run can be repeated without
        re-deriving the value.
        """
        assert seed_everything(1234) == 1234

    def test_seeding_makes_the_standard_library_reproducible(self):
        """Two runs with one seed draw the same numbers."""
        seed_everything(7)
        first = [random.random() for _ in range(5)]
        seed_everything(7)
        assert [random.random() for _ in range(5)] == first

    def test_seeding_makes_numpy_reproducible(self):
        """The legacy global numpy state is seeded too."""
        seed_everything(7)
        first = np.random.rand(5)  # noqa: NPY002 - legacy API is what is seeded.
        seed_everything(7)
        assert np.allclose(np.random.rand(5), first)  # noqa: NPY002

    def test_different_seeds_give_different_draws(self):
        """Confirms the seed is actually in use."""
        seed_everything(1)
        first = [random.random() for _ in range(5)]
        seed_everything(2)
        assert [random.random() for _ in range(5)] != first


class TestSeederRegistry:
    """Engines register their own seeding, because core cannot import them."""

    def test_a_registered_seeder_is_invoked(self):
        """
        The indirection that lets ``core`` seed PyTorch without importing it.

        ``core`` may not import a training library, so an engine registers a
        callback at import and ``core`` calls it.
        """
        seen: list[tuple[int, str]] = []
        register_seeder("probe", lambda seed, determinism: seen.append((seed, determinism)))
        seed_everything(11, determinism="warn")
        assert seen == [(11, "warn")]

    def test_registering_twice_is_rejected(self):
        """
        A duplicate name is an error, not a silent replacement.

        Two engines claiming one name means one of them is not being seeded,
        and a silent overwrite makes that undetectable.
        """
        register_seeder("probe", lambda seed, determinism: None)
        with pytest.raises(Exception, match="probe"):
            register_seeder("probe", lambda seed, determinism: None)

    def test_replacing_is_possible_when_asked_for_explicitly(self):
        """A test double can take over, but has to say so."""
        register_seeder("probe", lambda seed, determinism: None)
        calls: list[int] = []
        register_seeder("probe", lambda seed, determinism: calls.append(seed), replace=True)
        seed_everything(3)
        assert calls == [3]

    def test_unregistering_removes_the_seeder(self):
        """Needed so a test can clean up after itself."""
        register_seeder("probe", lambda seed, determinism: None)
        unregister_seeder("probe")
        assert "probe" not in registered_seeders()


class TestDeterminismLevels:
    """A failing seeder is tolerated or fatal according to the level chosen."""

    def test_a_failing_seeder_is_tolerated_when_determinism_is_off(self):
        """
        Best effort means best effort.

        With determinism off the caller has not asked for a guarantee, so a
        seeder that cannot comply should not end the run.
        """

        def explode(seed, determinism):
            message = "no deterministic kernel for this operation"
            raise RuntimeError(message)

        register_seeder("explosive", explode)
        assert seed_everything(5, determinism="off") == 5

    def test_a_failing_seeder_is_fatal_when_determinism_is_strict(self):
        """
        Strict means the guarantee is load-bearing.

        This is the defect being fixed: the previous implementation forced
        determinism inside a swallowed ``try``/``except``, so a run could
        report itself as deterministic when it was not. Under ``strict`` the
        failure must surface.
        """

        def explode(seed, determinism):
            message = "no deterministic kernel for this operation"
            raise RuntimeError(message)

        register_seeder("explosive", explode)
        with pytest.raises(SpecError, match="explosive"):
            seed_everything(5, determinism="strict")

    @pytest.mark.parametrize("level", ["off", "warn", "strict"])
    def test_every_level_is_accepted(self, level):
        """All three levels are valid, and none of them is a boolean."""
        assert seed_everything(5, determinism=level) == 5


class TestDeriveSeed:
    """Per-job seeds are derived by hashing, not by adding an index."""

    def test_the_same_labels_derive_the_same_seed(self):
        """
        The reproducibility property.

        Re-running one failed job alone must reproduce the result it would
        have had inside the full job set.
        """
        assert derive_seed(100, "EURUSD") == derive_seed(100, "EURUSD")

    def test_different_labels_derive_different_seeds(self):
        """Two job-set members must not share a seed."""
        assert derive_seed(100, "EURUSD") != derive_seed(100, "USDJPY")

    def test_different_base_seeds_derive_different_seeds(self):
        """Changing the run's seed changes every member's seed."""
        assert derive_seed(100, "EURUSD") != derive_seed(101, "EURUSD")

    def test_derivation_is_order_independent(self):
        """
        A job's seed does not depend on its position in the job list.

        Which is exactly what ``base + index`` would not give: reordering the
        jobs, or re-running a subset, would change every seed.
        """
        first = [derive_seed(1, job) for job in ("a", "b", "c")]
        second = [derive_seed(1, job) for job in ("c", "b", "a")]
        assert first == list(reversed(second))

    def test_multiple_labels_compose(self):
        """A seed can be derived per job and per fold without collision."""
        assert derive_seed(1, "EURUSD", "fold-0") != derive_seed(1, "EURUSD", "fold-1")

    def test_labels_are_not_merely_concatenated(self):
        """
        Label boundaries are significant.

        If labels were joined without a separator, ``("ab", "c")`` and
        ``("a", "bc")`` would derive the same seed -- so two different jobs
        could silently share one.
        """
        assert derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")

    def test_the_derived_seed_is_in_range(self):
        """
        The result fits the range generators accept.

        Numpy rejects a seed outside ``[0, 2**32)``, so a raw digest would
        fail at the point of use rather than here.
        """
        for job in ("EURUSD", "USDJPY", "a very long job identifier indeed"):
            seed = derive_seed(12345, job)
            assert 0 <= seed < 2**32
```

