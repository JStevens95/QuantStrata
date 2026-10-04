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
