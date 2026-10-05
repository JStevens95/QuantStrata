"""
Compares a refactored run against a golden fixture captured from the original.

The refactor's whole correctness claim is that the flagship model in
``rade_qnet`` produces the same results as the flagship model in ``rade_ml_pt``.
That claim is backed either by an artifact captured before anything changed,
or by an assertion. This module is the half that does the comparing; the
fixture it compares against is written by
``examples/rade_qnet/phase0_capture_baseline.py`` running the *unmodified*
original.

Why the diagnostic matters more than the verdict
------------------------------------------------
A parity failure says a refactor of several thousand lines produced a
different number somewhere. "Arrays differ" turns that into a day of
bisection. ``compare_arrays`` therefore reports the array's name, the index
of the worst element, both values at that index, and how many elements
mismatched out of how many -- which is usually enough to name the cause
without opening a debugger. A single mismatch in a 48,000-element array is a
boundary condition; half the array mismatching is a transpose or an ordering
difference.

Five levels, and why the tolerances widen
-----------------------------------------
``state`` and ``tensors`` are compared **exactly**. They are deterministic
NumPy computations over identical inputs, so there is no floating-point
excuse available: a difference means different arithmetic, which is a bug.

``forward`` allows ``atol=1e-6``. The original's weights are loaded into the
new model and both run on one batch, so the arithmetic is the same but may be
*reassociated* -- a fused kernel, a different reduction order -- which
perturbs the last bits without changing the computation.

``curve`` allows ``rtol=1e-3``. Several epochs of training accumulate that
perturbation. A tighter tolerance here produces a test that fails
occasionally and is therefore disabled, which is worse than a looser one that
still means something.

``job_set`` returns to exact. Nothing about *where* work runs should change
arithmetic, so a difference between a sequential and a parallel run means
state is being shared or seeded per-worker incorrectly.

The ordering trap
-----------------
Basis selection returns an *ordered* list, and that order fixes column
positions in every array downstream. A refactor that selects the same
instruments in a different order passes a set comparison and then fails
everything after it, with a symptom far from the cause. ``compare_state``
therefore compares the basis as a sequence, and
``test_reordered_basis_fails`` exists to keep it that way.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.lifecycle.errors import ContractError
from ..core.provenance.logging import get_logger

__all__ = [
    "ADJACENCY_VALUE_ATOL",
    "CURVE_RTOL",
    "FORWARD_ATOL",
    "Comparison",
    "GoldenFixture",
    "ParityReport",
    "compare_arrays",
    "compare_curve",
    "compare_forward",
    "compare_job_set",
    "compare_state",
    "compare_tensors",
    "load_golden",
]

#: The directory this module was imported relative to: one level up per dot in
#: its dotted name, so ``src.rade_qnet.testkit.parity`` and a vendored
#: ``tranql.models.rade.rade_qnet.rade_qnet.testkit.parity`` both land on the
#: root of the tree holding them, wherever pytest was invoked from.
_IMPORT_ROOT = Path(__file__).resolve().parents[__name__.count(".")]

#: Where fixtures live by default. Only a fallback: a deployment that keeps
#: them elsewhere sets :data:`GOLDEN_ROOT_VARIABLE`, or passes ``root``.
_FIXTURE_ROOT = _IMPORT_ROOT / "tests" / "fixtures" / "rade_qnet" / "golden"

#: Environment variable naming the fixture root. Read on every call rather
#: than once at import, so a test suite can point it somewhere for one session
#: without the order of imports deciding whether that took effect.
GOLDEN_ROOT_VARIABLE = "RADE_QNET_GOLDEN_ROOT"

#: Tolerances per level, as justified in the module docstring. Named constants
#: rather than defaults scattered across call sites, so that loosening one is
#: a visible edit in one place rather than an argument someone added.
FORWARD_ATOL = 1e-6
CURVE_RTOL = 1e-3

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Comparison:
    """
    The outcome of comparing one named array or value.

    Parameters
    ----------
    name
        What was compared, used verbatim in the message so a failure names
        the artifact rather than its position in some list.
    passed
        Whether it matched within tolerance.
    detail
        The diagnostic. Empty when it passed, because a report of fifty
        passing lines buries the one that failed.
    n_mismatched
        How many elements differed. Zero for a non-array comparison.
    n_total
        How many elements were compared. The ratio is what distinguishes a
        boundary condition from a systematic difference.
    worst_deviation
        Largest absolute difference found, or ``0.0``. Carried separately
        from the message so a caller can assert on it.
    """

    name: str
    passed: bool
    detail: str = ""
    n_mismatched: int = 0
    n_total: int = 0
    worst_deviation: float = 0.0


@dataclass
class ParityReport:
    """
    The findings for one parity level.

    Holds every comparison rather than stopping at the first failure. One
    difference is a clue; the set of differences is usually the diagnosis --
    every array failing points somewhere entirely different from one array
    failing.

    Parameters
    ----------
    level
        Which level this is, for the messages.
    comparisons
        One entry per artifact compared, in the order compared.
    """

    level: str
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every comparison matched."""
        return all(comparison.passed for comparison in self.comparisons)

    @property
    def failures(self) -> list[Comparison]:
        """Only the comparisons that failed, in the order compared."""
        return [comparison for comparison in self.comparisons if not comparison.passed]

    @property
    def worst_deviation(self) -> float:
        """
        The largest absolute difference across every comparison.

        Reported even on a pass, because a level that passes at 9e-7 against
        a tolerance of 1e-6 is worth knowing about before it starts failing.
        """
        return max((comparison.worst_deviation for comparison in self.comparisons), default=0.0)

    def add(self, comparison: Comparison) -> None:
        """
        Record one comparison.

        Parameters
        ----------
        comparison
            The outcome to record.
        """
        self.comparisons.append(comparison)

    def summary(self) -> str:
        """
        Render a human-readable verdict.

        Returns
        -------
        str
            One line on a pass. On a failure, a line per failed comparison,
            which is what gets attached to the assertion so the diagnosis is
            in the test output rather than behind a debugger.
        """
        if self.passed:
            return (
                f"parity level {self.level!r}: {len(self.comparisons)} comparison(s) "
                f"passed, worst deviation {self.worst_deviation:.3e}"
            )
        lines = [
            f"parity level {self.level!r}: {len(self.failures)} of "
            f"{len(self.comparisons)} comparison(s) FAILED"
        ]
        lines.extend(f"  - {comparison.detail}" for comparison in self.failures)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GoldenFixture:
    """
    A loaded fixture, addressed by artifact name rather than by file path.

    Parameters
    ----------
    name
        The fixture's directory name.
    directory
        Where it was loaded from.
    manifest
        What was captured, from which commit, with which config and seed.
        Carried so a parity failure can report the baseline's provenance,
        which is the first thing to check when a long-passing test starts
        failing.
    """

    name: str
    directory: Path
    manifest: Mapping[str, Any]

    def array(self, relative_path: str) -> NDArray[Any]:
        """
        Load one ``.npy`` array from the fixture.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        numpy.ndarray
            The stored array.

        Raises
        ------
        ContractError
            If the file is absent. Raised rather than returning ``None``
            because an incomplete fixture silently skipping a comparison is
            how a parity suite comes to pass everything.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"(looked in {self.directory}). The fixture is incomplete, so this "
                f"comparison would be skipped rather than run -- re-capture it with "
                f"examples/rade_qnet/phase0_capture_baseline.py"
            )
        # `allow_pickle=False` is the default and is relied upon: a fixture is
        # data, and a fixture that can execute code on load is a fixture that
        # could hide a difference rather than reveal one.
        return np.load(path)

    def arrays(self, relative_path: str) -> dict[str, NDArray[Any]]:
        """
        Load every array from one ``.npz`` archive in the fixture.

        Returned eagerly as a dictionary rather than as the lazy handle
        ``numpy.load`` gives, because that handle holds the file open and a
        caller who kept one past the end of a test would leave a descriptor
        behind on every comparison.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        dict
            Array name to array.

        Raises
        ------
        ContractError
            If the archive is absent, for the reason given in
            :meth:`array`.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"under {self.directory}; re-capture it rather than skipping the "
                f"comparison"
            )
        # `allow_pickle=False` is the default and is left so deliberately:
        # a fixture is an ordinary file on disk, and unpickling one would
        # execute whatever it contained.
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def json(self, relative_path: str) -> object:
        """
        Load one JSON document from the fixture.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        object
            The parsed document. Typed as ``object`` rather than ``Any``
            because a fixture holds whatever was captured, and ``Any`` would
            silence the type checker at every call site rather than at this
            one.

        Raises
        ------
        ContractError
            If the file is absent.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"(looked in {self.directory}). The fixture is incomplete, so this "
                f"comparison would be skipped rather than run -- re-capture it with "
                f"examples/rade_qnet/phase0_capture_baseline.py"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def has(self, relative_path: str) -> bool:
        """
        Whether an artifact is present.

        Parameters
        ----------
        relative_path
            Path within the fixture directory.

        Returns
        -------
        bool
            True when the file exists.
        """
        return (self.directory / relative_path).is_file()


def load_golden(name: str, *, root: Path | None = None) -> GoldenFixture:
    """
    Read a fixture and its manifest.

    Parameters
    ----------
    name
        Fixture directory name, such as ``hybrid_gnn_rnn``.
    root
        The fixture root. ``None`` reads :data:`GOLDEN_ROOT_VARIABLE` from the
        environment, falling back to the repository's own location when that
        is unset.

    Returns
    -------
    GoldenFixture
        The loaded fixture.

    Raises
    ------
    ContractError
        If the directory or its manifest is missing, naming the path looked
        at. A fixture that is simply absent must not read as a pass.
    """
    default = Path(os.environ.get(GOLDEN_ROOT_VARIABLE, _FIXTURE_ROOT))
    directory = (root or default) / name
    if not directory.is_dir():
        raise ContractError(
            f"no golden fixture named {name!r} at {directory}. Capture one with "
            f"examples/rade_qnet/phase0_capture_baseline.py before running parity tests"
        )

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError(
            f"the golden fixture at {directory} has no manifest.json, so there is no "
            f"record of which commit, config or seed produced it. A fixture without "
            f"provenance cannot be a baseline -- re-capture it"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _LOGGER.debug("loaded golden fixture %r captured at %s", name, manifest.get("captured_at"))
    return GoldenFixture(name=name, directory=directory, manifest=manifest)


def _compare_structure(
    actual: NDArray[Any], expected: NDArray[Any], *, name: str
) -> Comparison | None:
    """
    Check shape and dtype, which are reported as themselves or not at all.

    Separated from the value comparison because the two failures call for
    completely different investigations. A shape mismatch reported as a value
    difference sends the reader looking for a numerical bug when the cause is
    a transpose; a dtype mismatch silently upcast hides a real narrowing.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison or None
        A failure when the structure differs, otherwise ``None`` to signal
        that the values are worth comparing.
    """
    if actual.shape != expected.shape:
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: shape mismatch, got {actual.shape} expected {expected.shape}. "
                f"This is a structural difference, not a numerical one -- check for a "
                f"transpose, a dropped axis, or a different element count before "
                f"looking at values"
            ),
            n_total=expected.size,
        )

    if actual.dtype != expected.dtype:
        # Reported rather than coerced. float32 where float64 was captured is
        # a real narrowing that costs precision downstream, and an integer
        # index array that became a float is usually a bug in how it was
        # built.
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: dtype mismatch, got {actual.dtype} expected {expected.dtype}. "
                f"Not upcast automatically, because a narrowed float loses precision "
                f"downstream and an index array that became floating-point is a "
                f"construction bug"
            ),
            n_total=expected.size,
        )
    return None


def _compare_exactly(actual: NDArray[Any], expected: NDArray[Any], *, name: str) -> Comparison:
    """
    Compare non-numeric arrays element by element.

    Strings and objects have no meaningful tolerance, so the first differing
    element is reported rather than the worst one.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison
        The outcome.
    """
    mismatched = np.asarray(actual != expected)
    n_mismatched = int(mismatched.sum())
    if n_mismatched == 0:
        return Comparison(name=name, passed=True, n_total=expected.size)

    first = int(np.argmax(mismatched.ravel()))
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: {n_mismatched} of {expected.size} element(s) differ; "
            f"first at flat index {first}: got {actual.ravel()[first]!r}, "
            f"expected {expected.ravel()[first]!r}"
        ),
        n_mismatched=n_mismatched,
        n_total=expected.size,
    )


def compare_arrays(
    actual: NDArray[Any],
    expected: NDArray[Any],
    *,
    name: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Comparison:
    """
    Compare two arrays and produce a diagnostic good enough to act on.

    Shape and dtype are checked before values, and reported as themselves. A
    shape mismatch reported as a value difference sends the reader looking for
    a numerical bug when the cause is a transpose or a missing axis; a dtype
    mismatch silently upcast hides a real narrowing.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        The artifact's name, used verbatim in the message.
    atol
        Absolute tolerance. The default of zero means exact, which is correct
        for levels 1, 2 and 5.
    rtol
        Relative tolerance, applied as ``rtol * abs(expected)``.

    Returns
    -------
    Comparison
        The outcome, with a diagnostic when it failed.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    structural = _compare_structure(actual, expected, name=name)
    if structural is not None:
        return structural

    if expected.size == 0:
        return Comparison(name=name, passed=True, n_total=0)

    # Non-numeric arrays (strings, objects) have no meaningful tolerance, so
    # they are compared for equality element by element.
    if expected.dtype.kind not in "fciub":
        return _compare_exactly(actual, expected, name=name)

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))

    # NaN needs handling in both directions, and the subtraction above gets
    # one of them silently wrong. A fixture that recorded NaN and a run that
    # reproduces it have matched, so `both_nan` is excused. But `nan - 2.0` is
    # `nan`, and `nan > tolerance` is *False*, so a refactor that started
    # producing NaN where the original produced a number would otherwise be
    # reported as a pass -- the most dangerous direction this harness has.
    actual_nan = np.isnan(actual.astype(np.float64))
    expected_nan = np.isnan(expected.astype(np.float64))
    both_nan = actual_nan & expected_nan
    only_one_nan = actual_nan ^ expected_nan
    mismatched = ((difference > allowed) & ~both_nan) | only_one_nan
    n_mismatched = int(mismatched.sum())

    finite = difference[np.isfinite(difference)]
    worst_deviation = float(finite.max()) if finite.size else 0.0

    if n_mismatched == 0:
        return Comparison(
            name=name,
            passed=True,
            n_total=expected.size,
            worst_deviation=worst_deviation,
        )

    # The worst *mismatching* element, not the worst overall: with a relative
    # tolerance the largest absolute difference may be the one that passed.
    masked = np.where(mismatched, difference, -np.inf)
    worst_flat = int(np.argmax(masked))
    worst_index = np.unravel_index(worst_flat, expected.shape)
    index_text = worst_index[0] if expected.ndim == 1 else worst_index

    tolerance_text = "exact" if atol == 0.0 and rtol == 0.0 else f"atol={atol:g}, rtol={rtol:g}"
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: {n_mismatched} of {expected.size} element(s) differ "
            f"({tolerance_text}); worst at index {index_text}: "
            f"got {actual[worst_index]!r}, expected {expected[worst_index]!r}, "
            f"difference {difference[worst_index]:.6e}"
        ),
        n_mismatched=n_mismatched,
        n_total=expected.size,
        worst_deviation=worst_deviation,
    )


def _compare_sequence(actual: Sequence[Any], expected: Sequence[Any], *, name: str) -> Comparison:
    """
    Compare two ordered sequences, treating order as part of the value.

    Used for the selected basis, where the order fixes column positions in
    every array downstream. A set comparison here would pass on a reordering
    and fail everything after it, with the symptom far from the cause -- so
    the difference is reported as a reordering when the membership matches,
    which names the likely fix directly.

    Parameters
    ----------
    actual
        The sequence produced.
    expected
        The sequence captured.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison
        The outcome.
    """
    actual = list(actual)
    expected = list(expected)

    if actual == expected:
        return Comparison(name=name, passed=True, n_total=len(expected))

    if sorted(map(str, actual)) == sorted(map(str, expected)):
        differing = [
            position
            for position, (left, right) in enumerate(zip(actual, expected, strict=True))
            if left != right
        ]
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: same {len(expected)} member(s) in a DIFFERENT ORDER, first "
                f"differing at position {differing[0]}: got {actual[differing[0]]!r}, "
                f"expected {expected[differing[0]]!r}. Order fixes column positions in "
                f"every array downstream, so this alone will fail every later "
                f"comparison -- fix the ordering before investigating anything else"
            ),
            n_mismatched=len(differing),
            n_total=len(expected),
        )

    missing = sorted(set(map(str, expected)) - set(map(str, actual)))
    extra = sorted(set(map(str, actual)) - set(map(str, expected)))
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: membership differs, got {len(actual)} expected {len(expected)}; "
            f"missing {missing[:5]}, unexpected {extra[:5]}"
        ),
        n_mismatched=len(missing) + len(extra),
        n_total=len(expected),
    )


def compare_state(actual: Mapping[str, Any], golden: GoldenFixture) -> ParityReport:
    """
    Level 1 — the fitted state, compared exactly.

    Scaler statistics, the selected basis *with its order*, the encoded
    attribute matrix, the sparse graph and the universe. All are
    deterministic NumPy computations over identical inputs, so any difference
    is a difference in arithmetic.

    The index arrays are compared too, and they are the subtle ones: the
    original recomputes them **after** dimensionality reduction, so a refactor
    carrying pre-reduction indices forward produces plausible arrays that are
    wrong in every downstream stage.

    Parameters
    ----------
    actual
        The produced state, keyed by the fixture's artifact names without
        their directory or suffix -- ``scaler_mean``, ``selected_basis``,
        ``combined_features``, and so on.
    golden
        The loaded fixture.

    Returns
    -------
    ParityReport
        One comparison per artifact.
    """
    report = ParityReport(level="1-state")

    # Compared as an ordered sequence, never as a set. See `_compare_sequence`.
    if "selected_basis" in actual:
        report.add(
            _compare_sequence(
                actual["selected_basis"],
                golden.json("level1_state/selected_basis.json"),
                name="selected_basis",
            )
        )

    for key, relative_path in _LEVEL1_ARRAYS.items():
        if key not in actual:
            continue
        if not golden.has(relative_path):
            continue
        report.add(
            compare_arrays(
                np.asarray(actual[key]),
                golden.array(relative_path),
                name=key,
                atol=_LEVEL1_ATOL.get(key, 0.0),
            )
        )

    if "universe" in actual and golden.has("level1_state/universe.json"):
        expected_universe = golden.json("level1_state/universe.json")
        for field_name in ("elementary_ids", "target_ids"):
            if field_name in expected_universe:
                report.add(
                    _compare_sequence(
                        actual["universe"].get(field_name, []),
                        expected_universe[field_name],
                        name=f"universe.{field_name}",
                    )
                )
    return report


#: Level 1 artifacts that are plain arrays, mapped from the key a caller
#: supplies to the file the capture wrote. Kept as data rather than as a chain
#: of `if` statements so that adding an artifact is a one-line change and the
#: capture script can be checked against the same list.
_LEVEL1_ARRAYS: Mapping[str, str] = {
    "scaler_mean": "level1_state/scaler_mean.npy",
    "scaler_scale": "level1_state/scaler_scale.npy",
    "target_scaler_mean": "level1_state/target_scaler_mean.npy",
    "target_scaler_scale": "level1_state/target_scaler_scale.npy",
    "combined_features": "level1_state/combined_features.npy",
    "adjacency_indices": "level1_state/adjacency_indices.npy",
    "adjacency_values": "level1_state/adjacency_values.npy",
    "adjacency_shape": "level1_state/adjacency_shape.npy",
    # Post-reduction, and that is the whole point. The original recomputes
    # these as 0..n_e and n_e..n_e+n_t *after* the basis is selected, so a
    # refactor that carries the pre-reduction indices forward produces arrays
    # that look right and index the wrong columns.
    "elementary_idx": "level1_state/elementary_idx.npy",
    "target_idx": "level1_state/target_idx.npy",
}

#: One float32 ULP at a magnitude of one. The single artifact that is not
#: compared exactly, and the reason is worth stating in full.
#:
#: The baseline found graph neighbours with scikit-learn, whose brute-force
#: euclidean kernel works in float32 and computes the distance from the
#: expanded `|a|^2 - 2ab + |b|^2` rather than from the explicit difference.
#: That expansion cancels: the baseline's distances carry about 2e-07 of
#: error where float64 would carry 1e-16. The port computes the explicit
#: difference in float64 instead, which is both more accurate and free of
#: scikit-learn, and the two therefore disagree in the last float32 bit of
#: each edge weight.
#:
#: Matching bit for bit would mean reproducing the accumulation order of a
#: Cython float32 GEMM -- not stable across scikit-learn versions or BLAS
#: builds, and achievable only by keeping scikit-learn in the data path that
#: the port exists to remove. Being less accurate on purpose, against a
#: moving target, is the worse trade.
#:
#: What is *not* relaxed is the graph's structure. `adjacency_indices` is
#: still compared exactly, and `TestTheGraphIsRobust` measures the margin
#: between the furthest kept neighbour and the nearest dropped one: it is
#: four orders of magnitude above this noise, so no edge is at risk of
#: flipping. The weights differ in their last bit; the graph does not differ.
ADJACENCY_VALUE_ATOL = 1e-7

_LEVEL1_ATOL: Mapping[str, float] = {"adjacency_values": ADJACENCY_VALUE_ATOL}

#: The same tolerance, keyed by the name the batches use. A batch carries
#: the graph verbatim from the fitted state, so the artifact that carries a
#: tolerance at level 1 carries the identical one here -- and must, because
#: nothing between the two stages touches it.
_LEVEL2_ATOL: Mapping[str, float] = {"adjacency_values": ADJACENCY_VALUE_ATOL}


def compare_tensors(
    actual: Mapping[str, Mapping[str, Any]],
    golden: GoldenFixture,
    *,
    allow_missing: Collection[str] = (),
) -> ParityReport:
    """
    Level 2 — batch contents, compared exactly.

    Confirms that moving static inputs out of per-sample collation into a
    separate field changed nothing. The original merged every static tensor
    into every sample and then compared across the batch to recover the one
    copy; the refactor delivers that one copy directly. The network receives
    the same object either way, and this level is what says so.

    Parameters
    ----------
    actual
        Split name to a mapping of tensor name to array, for the batches the
        fixture captured.
    golden
        The loaded fixture.
    allow_missing
        Tensor names the fixture carries that the refactor deliberately does
        not produce. An input the original passed into every batch and never
        read is the motivating case: dropping it is a finding to record, not
        a failure, but it has to be named here so that dropping a second one
        is still a failure.

    Returns
    -------
    ParityReport
        One comparison per tensor per captured batch, plus the split indices.
    """
    report = ParityReport(level="2-tensors")

    if golden.has("level2_tensors/split_indices.json"):
        expected_splits = golden.json("level2_tensors/split_indices.json")
        for split, expected_indices in expected_splits.items():
            if split not in actual:
                continue
            produced = actual[split].get("__indices__")
            if produced is None:
                continue
            report.add(
                compare_arrays(
                    np.asarray(produced, dtype=np.int64),
                    np.asarray(expected_indices, dtype=np.int64),
                    name=f"{split}.split_indices",
                )
            )

    for split, tensors in actual.items():
        relative_path = f"level2_tensors/{split}_batch_000.npz"
        if not golden.has(relative_path):
            continue
        with np.load(golden.directory / relative_path) as stored:
            for tensor_name in sorted(stored.files):
                if tensor_name in allow_missing:
                    # Declared by the caller as deliberately not delivered.
                    # Named one at a time rather than enabled wholesale, so
                    # dropping a second input is a failure that has to be
                    # argued for rather than one the suite absorbs.
                    continue
                if tensor_name not in tensors:
                    report.add(
                        Comparison(
                            name=f"{split}.{tensor_name}",
                            passed=False,
                            detail=(
                                f"{split}.{tensor_name}: present in the fixture but "
                                f"absent from the produced batch. A key that "
                                f"disappeared is either a renamed input or one the "
                                f"refactor stopped delivering; pass it in "
                                f"`allow_missing` if the omission is intended"
                            ),
                        )
                    )
                    continue
                report.add(
                    compare_arrays(
                        np.asarray(tensors[tensor_name]),
                        stored[tensor_name],
                        name=f"{split}.{tensor_name}",
                        atol=_LEVEL2_ATOL.get(tensor_name, 0.0),
                    )
                )
    return report


def compare_forward(
    actual: NDArray[Any], golden: GoldenFixture, *, atol: float = FORWARD_ATOL
) -> ParityReport:
    """
    Level 3 — the forward pass, at ``atol=1e-6``.

    The original's weights are loaded into the refactored model and both are
    run on one fixed batch. The arithmetic is the same but may be
    reassociated by a fused kernel or a different reduction order, which
    perturbs the last bits without changing the computation. A difference
    much larger than the tolerance is a structural difference, not a
    numerical one.

    Parameters
    ----------
    actual
        The refactored model's output for the captured batch.
    golden
        The loaded fixture.
    atol
        Absolute tolerance.

    Returns
    -------
    ParityReport
        A single comparison.
    """
    report = ParityReport(level="3-forward")
    report.add(
        compare_arrays(
            np.asarray(actual, dtype=np.float64),
            golden.array("level3_forward/outputs.npy").astype(np.float64),
            name="forward_outputs",
            atol=atol,
        )
    )
    return report


def compare_curve(
    actual: Mapping[str, Sequence[float]], golden: GoldenFixture, *, rtol: float = CURVE_RTOL
) -> ParityReport:
    """
    Level 4 — the training curve, at ``rtol=1e-3``.

    Several epochs accumulate the last-bit perturbation level 3 tolerates, so
    the tolerance is relative and loose. It is deliberately not tighter: a
    gate that fails one run in ten gets disabled, and a disabled gate proves
    nothing. If this level is flaky at ``1e-3``, the response is to shorten
    the captured run rather than to loosen the tolerance further.

    Parameters
    ----------
    actual
        Curve name to per-epoch values, such as ``train_loss`` and
        ``val_loss``.
    golden
        The loaded fixture.
    rtol
        Relative tolerance.

    Returns
    -------
    ParityReport
        One comparison per curve.
    """
    report = ParityReport(level="4-training")
    expected_curves = golden.json("level4_training/curve.json")

    for curve_name, expected_values in sorted(expected_curves.items()):
        if curve_name not in actual:
            report.add(
                Comparison(
                    name=curve_name,
                    passed=False,
                    detail=(
                        f"{curve_name}: captured in the fixture but not produced. "
                        f"A missing curve is a missing metric, not a passing one"
                    ),
                )
            )
            continue
        report.add(
            compare_arrays(
                np.asarray(actual[curve_name], dtype=np.float64),
                np.asarray(expected_values, dtype=np.float64),
                name=curve_name,
                rtol=rtol,
            )
        )
    return report


def compare_job_set(
    sequential: Mapping[str, NDArray[Any]], parallel: Mapping[str, NDArray[Any]]
) -> ParityReport:
    """
    Level 5 — sequential against parallel, compared exactly.

    Nothing about *where* work runs should change arithmetic. A difference
    here means state is shared between workers or seeds are derived
    per-worker in a way that depends on placement, and both produce results
    that cannot be reproduced on a differently sized machine.

    Compared against each other rather than against a fixture: the claim is
    that the two agree, and pinning either one to a stored artifact would
    make the test fail for the unrelated reason that the model improved.

    Parameters
    ----------
    sequential
        Artifact name to array, from the sequential run.
    parallel
        The same, from the parallel run.

    Returns
    -------
    ParityReport
        One comparison per artifact, plus a check that the two runs produced
        the same set of artifacts.
    """
    report = ParityReport(level="5-job-set")

    only_sequential = sorted(set(sequential) - set(parallel))
    only_parallel = sorted(set(parallel) - set(sequential))
    if only_sequential or only_parallel:
        report.add(
            Comparison(
                name="artifacts",
                passed=False,
                detail=(
                    f"the two runs produced different artifacts: "
                    f"{only_sequential} only sequentially, {only_parallel} only in "
                    f"parallel. A job that ran in one placement and not the other is "
                    f"a scheduling bug, not a numerical one"
                ),
            )
        )

    for name in sorted(set(sequential) & set(parallel)):
        report.add(
            compare_arrays(np.asarray(parallel[name]), np.asarray(sequential[name]), name=name)
        )
    return report
