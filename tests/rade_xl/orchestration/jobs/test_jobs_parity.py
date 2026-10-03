"""
Parity level 5: placement must not change results.

The whole proposition of a job set is that *where* a job runs is an
operational choice. If running forty jobs across eight processes produced
different models than running them one after another, the parallelism would
not be an optimisation -- it would be a second, undocumented experiment.

This is the Phase 4 gate, and it runs the flagship model rather than the
synthetic one the rest of this package uses. A toy model with a closed-form
fit would pass trivially; what is actually at risk is the real thing, with an
optimiser, a graph, and a dozen library calls whose behaviour depends on the
process they run in.

What is compared, and what is not
---------------------------------
"Identical artifacts, byte for byte" is unachievable taken literally: a
bundle records when it was written and how long it took, and those *must*
differ between two runs. So the gate compares a named, reviewed set of things
placement must not change, with the exclusions equally named -- both as
module-level constants, so widening either is a visible change rather than a
quiet edit to an assertion. See `PHASE_4_JOB_SETS.md` §8.1.

Metrics are compared **exactly** rather than to a tolerance. Placement
changing the eighth decimal place is still placement changing results.

Two preconditions, and why they are constraints rather than evasions
--------------------------------------------------------------------
Both were found by running this comparison and watching it fail, and both
turned out to be real defects rather than reasons to soften the criterion.

*The thread budget must be pinned in the specification.* The number of
intra-op threads fixes the order in which a reduction accumulates, so it
fixes the last few significant figures. A worker process gets its budget from
an environment variable read before it imports anything; the process that
launched it has already imported Torch, so the same variable does nothing
there. The two paths therefore ran at different thread counts and scored
differently. That is defect 13, and the fix makes the budget a specification
field applied in-process. What remains is genuine: a run that does not say
how many threads it wants gets whatever the host decided.

*The device must be pinned to the CPU.* Under ``device: auto`` on Apple
silicon the model is not reproducible **against itself** -- two sequential
runs, same process, ``determinism: strict``, differ. MPS kernels are
non-deterministic and ``torch.use_deterministic_algorithms`` does not cover
them. Nothing in the framework can fix that, and a gate that blamed placement
for it would be measuring the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.rade_xl.core.contract.bundle import WEIGHTS_FILENAME
from src.rade_xl.core.spec.jobs import parse_job_set_spec
from src.rade_xl.orchestration.compute.local import LocalExecutor
from src.rade_xl.orchestration.compute.processes import ProcessExecutor
from src.rade_xl.orchestration.jobs.set import JobSetRunner

#: The Phase 0 capture's input, which the flagship model's data module reads.
FIXTURE = Path("tests/fixtures/rade_xl/golden/hybrid_gnn_rnn/input")

#: Fields of a job record that placement must not change. Listed rather than
#: compared wholesale so that adding a field to `JobRecord` is a decision
#: about whether placement may affect it, not a silent widening.
COMPARED_FIELDS = (
    "job_id",
    "status",
    "seed",
    "epochs",
    "stopped_early",
    "metrics",
    "bundle_directory",
    "bundle_version",
    "model_name",
    "failure_kind",
    "failure_message",
)

#: Fields that *must* differ, or may. Excluded deliberately: a record that
#: took the same number of seconds in two different placements would mean the
#: placement had not changed anything, which is not the claim being tested.
EXCLUDED_FIELDS = (
    "wall_seconds",
    "failure_traceback",
)


def job_set(tmp_path, executor_name):
    """
    Build a two-job set over the flagship model, fully pinned.

    Pinned in every respect that affects arithmetic: device, determinism and
    thread budget. The two jobs differ in width, so a run that silently
    trained one model twice would be visible as two identical scores rather
    than hiding behind a comparison that happens to pass.

    Parameters
    ----------
    tmp_path
        Where the set writes. Separate per executor, because the comparison
        is of what each produced, not of one overwriting the other.
    executor_name
        Recorded in the specification, though the executor is supplied
        directly so the test does not depend on the placement policy's view
        of the machine.

    Returns
    -------
    JobSetSpec
        The specification.
    """
    return parse_job_set_spec(
        {
            "model": "hybrid_gnn_rnn",
            "name": "parity",
            "output_root": str(tmp_path),
            "defaults": {
                "source": {
                    "kind": "model",
                    "params": {"directory": str(FIXTURE.resolve())},
                    "transforms": {"sequence": {"length": 4}},
                },
                "training": {"engine": "torch", "epochs": 2},
                "hardware": {
                    "device": "cpu",
                    "determinism": "strict",
                    "threads_per_worker": 1,
                },
                "reports": {"enabled": []},
            },
            "jobs": [
                {"id": "wide", "overrides": {"model": {"params": {"units": 16}}}},
                {"id": "narrow", "overrides": {"model": {"params": {"units": 8}}}},
            ],
            "placement": {"executor": executor_name},
        }
    )


@dataclass(frozen=True)
class Run:
    """
    One completed job set, and where it wrote.

    Paired because a manifest records bundle paths *relative* to the set's
    directory -- which is what lets it survive being copied -- so reading
    one back means rejoining the two.

    Parameters
    ----------
    manifest
        The set's manifest.
    directory
        The set's output directory.
    """

    manifest: object
    directory: Path


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    """
    Run the same set twice: once sequentially, once across two processes.

    Module-scoped because this trains four models, which is slow enough that
    repeating it per assertion would dominate the suite. The runs are
    independent and nothing mutates them, so sharing is safe.

    Returns
    -------
    tuple
        The sequential manifest and the pooled one.
    """
    # Imported here rather than at module level for its registration side
    # effect, which is what makes `hybrid_gnn_rnn` resolvable by name.
    import src.rade_xl.models.hybrid_gnn_rnn.register  # noqa: F401, PLC0415

    sequential_root = tmp_path_factory.mktemp("sequential")
    pooled_root = tmp_path_factory.mktemp("pooled")

    sequential = JobSetRunner(job_set(sequential_root, "local"), executor=LocalExecutor())
    pooled = JobSetRunner(
        job_set(pooled_root, "processes"),
        executor=ProcessExecutor(workers=2, threads_per_worker=1),
    )

    return (
        Run(sequential.run(), sequential.output_directory),
        Run(pooled.run(), pooled.output_directory),
    )


class TestPlacementDoesNotChangeResults:
    """The gate."""

    def test_both_placements_ran_every_job(self, manifests):
        """
        A precondition for the rest.

        Asserted separately so a failure here does not read as a parity
        failure. A pool that silently dropped a job would otherwise make every
        comparison below vacuous.
        """
        sequential, pooled = manifests

        assert len(sequential.manifest.succeeded) == 2
        assert len(pooled.manifest.succeeded) == 2

    def test_the_jobs_are_the_same_jobs_in_the_same_order(self, manifests):
        """
        Declaration order, not completion order.

        A pool finishes jobs in whatever order they happen to end, and a
        manifest that followed that would differ between two identical
        runs for no reason anyone could act on.
        """
        sequential, pooled = manifests

        assert [record.job_id for record in sequential.manifest.jobs] == ["wide", "narrow"]
        assert [record.job_id for record in pooled.manifest.jobs] == ["wide", "narrow"]

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_metrics_are_identical(self, manifests, job_id):
        """
        Exactly identical, not close.

        A tolerance here would pass a framework in which placement moved
        every score slightly, which is precisely the thing a job set
        promises it does not do.
        """
        sequential, pooled = manifests

        assert sequential.manifest.record(job_id).metrics == pooled.manifest.record(job_id).metrics

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_every_compared_field_agrees(self, manifests, job_id):
        """
        The named list, field by field.

        Reported per field rather than by comparing whole records, so a
        failure says which property placement changed.
        """
        sequential, pooled = manifests
        left = sequential.manifest.record(job_id)
        right = pooled.manifest.record(job_id)

        differing = [
            field for field in COMPARED_FIELDS if getattr(left, field) != getattr(right, field)
        ]

        assert differing == []

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_the_weights_are_byte_identical(self, manifests, job_id):
        """
        The artifact itself, not a summary of it.

        Matching metrics with differing weights would mean the two runs
        reached different models that happen to score the same on this
        data -- which is a far worse failure than differing metrics,
        because nothing downstream would notice.
        """
        sequential, pooled = manifests

        left = _weights(sequential, job_id)
        right = _weights(pooled, job_id)

        assert left == right

    def test_the_two_jobs_are_actually_different_models(self, manifests):
        """
        A guard against the comparison passing for the wrong reason.

        If the per-job overrides were silently dropped, both jobs would
        train the same model and every assertion above would hold while
        testing nothing about overrides at all.
        """
        sequential, _ = manifests

        assert (
            sequential.manifest.record("wide").metrics
            != sequential.manifest.record("narrow").metrics
        )


class TestTheExclusionsAreDeliberate:
    """What the gate declines to compare, and why that is honest."""

    def test_durations_are_excluded_and_do_differ(self, manifests):
        """
        The exclusion is real, not defensive.

        A pooled run and a sequential one taking identical wall time would
        suggest the placement had not taken effect.
        """
        sequential, pooled = manifests

        assert "wall_seconds" in EXCLUDED_FIELDS
        assert sequential.manifest.wall_seconds != pooled.manifest.wall_seconds

    def test_no_field_is_both_compared_and_excluded(self):
        """
        The two lists partition what a record carries.

        An overlap would let a field look checked while being exempt.
        """
        assert set(COMPARED_FIELDS).isdisjoint(EXCLUDED_FIELDS)

    def test_the_lists_cover_every_field_a_record_has(self):
        """
        Nothing is unaccounted for.

        A field added to `JobRecord` later should fail this test, forcing a
        decision about whether placement may affect it -- which is the
        whole point of naming the lists rather than comparing records
        wholesale.
        """
        from src.rade_xl.orchestration.jobs.manifest import JobRecord  # noqa: PLC0415

        assert set(JobRecord.model_fields) == set(COMPARED_FIELDS) | set(EXCLUDED_FIELDS)


def _weights(run: Run, job_id: str) -> bytes:
    """
    Read a job's weights file.

    Parameters
    ----------
    run
        The completed set.
    job_id
        Which job.

    Returns
    -------
    bytes
        The file's contents.
    """
    record = run.manifest.record(job_id)
    return (run.directory / record.bundle_directory / WEIGHTS_FILENAME).read_bytes()
