"""
Tests for the executors, and for the rules they all share.

The conformance rules are parametrised over every executor rather than
written once per class. That is the whole proposition of this layer: a
pipeline cannot observe which executor is running it, so a rule that held for
one and not another would mean placement changes results -- which is exactly
what parity level 5 is the gate against.

The sequential executor is the reference. Where a test compares two
executors, it compares the other one against this.
"""

from __future__ import annotations

import os

import pytest

from src.rade_xl.orchestration.compute.base import (
    Executor,
    ResultSummary,
    WorkFailure,
    WorkItem,
    WorkResult,
    execute_item,
)
from src.rade_xl.orchestration.compute.gpus import GpuExecutor, visible_device_ids
from src.rade_xl.orchestration.compute.local import LocalExecutor
from src.rade_xl.orchestration.compute.processes import (
    CUDA_VISIBILITY_VARIABLE,
    THREAD_VARIABLES,
    ProcessExecutor,
)

from .workers import (
    double,
    explode,
    fail_unless_even,
    read_environment_variable,
    read_process_id,
    read_thread_budget,
)

#: Two workers, not more. Enough to be a pool; cheap enough that nobody is
#: tempted to skip the tests that use it.
POOL_WORKERS = 2


def items(values, function=double):
    """
    Build one work item per value, keyed by the value.

    Parameters
    ----------
    values
        Payloads.
    function
        The module-level function to call.

    Returns
    -------
    list of WorkItem
        The items.
    """
    return [WorkItem(key=str(value), function=function, payload=value) for value in values]


@pytest.fixture(params=["local", "processes"])
def executor(request):
    """Return each executor that can run on any machine, in turn."""
    if request.param == "local":
        return LocalExecutor()
    return ProcessExecutor(workers=POOL_WORKERS)


class TestTheProtocol:
    """Structural conformance, which is what lets a user's executor substitute."""

    @pytest.mark.parametrize(
        "built",
        [LocalExecutor(), ProcessExecutor(workers=POOL_WORKERS)],
        ids=["local", "processes"],
    )
    def test_an_executor_satisfies_the_protocol(self, built):
        """Without inheriting from it, which is the point of a protocol."""
        assert isinstance(built, Executor)

    def test_each_executor_describes_itself(self, executor):
        """
        Recorded in the manifest, so a set remembers where it actually ran.

        That differs from where it was configured to run whenever the
        placement policy chose, which is the default.
        """
        assert executor.description


class TestTheSharedRules:
    """The three rules every executor is held to."""

    def test_one_result_per_item(self, executor):
        """No coalescing, no dropping, even when payloads repeat."""
        results = executor.map(items([1, 1, 1]))
        assert len(results) == 3

    def test_results_come_back_in_input_order(self, executor):
        """
        Never in completion order.

        A pool finishes jobs in whatever order the scheduler and the data
        happen to produce. A manifest ordered by that would differ between
        two identical runs, which would make a job set unreproducible for a
        reason that has nothing to do with the model.
        """
        results = executor.map(items([5, 1, 4, 2, 3]))
        assert [result.key for result in results] == ["5", "1", "4", "2", "3"]
        assert [result.value for result in results] == [10, 2, 8, 4, 6]

    def test_a_failure_is_returned_rather_than_raised(self, executor):
        """
        The property that makes partial failure possible at all.

        Raising would discard every other job in the set, which is the
        behaviour that makes a framework untrustworthy at scale -- and the
        workaround people adopt is running jobs one at a time by hand.
        """
        results = executor.map(items(["boom"], function=explode))
        assert not results[0].succeeded
        assert results[0].failure.kind == "ValueError"
        assert results[0].failure.message == "boom"

    def test_the_same_items_twice_give_the_same_values(self, executor):
        """Placement is not allowed to change results, and nor is repetition."""
        first = executor.map(items([1, 2, 3]))
        second = executor.map(items([1, 2, 3]))
        assert [r.value for r in first] == [r.value for r in second]

    def test_an_empty_set_is_an_empty_list(self, executor):
        """
        Rather than an error or a started-and-idle pool.

        A job set filtered down to nothing is a legitimate outcome -- every
        cluster excluded, say -- and the caller should not have to guard it.
        """
        assert executor.map([]) == []


class TestPartialFailure:
    """A failed item must not take its neighbours with it."""

    def test_the_others_still_complete(self, executor):
        """Thirty-nine good models are not discarded for one bad input."""
        results = executor.map(items([0, 1, 2, 3, 4], function=fail_unless_even))
        assert [result.succeeded for result in results] == [True, False, True, False, True]
        assert [result.value for result in results if result.succeeded] == [0, 2, 4]

    def test_a_failure_keeps_its_place_in_the_order(self, executor):
        """
        So a manifest row still lines up with the job that produced it.

        A failure dropped from the result list rather than returned in place
        would silently shift every later row onto the wrong job.
        """
        results = executor.map(items([1, 2], function=fail_unless_even))
        assert [result.key for result in results] == ["1", "2"]

    def test_the_traceback_survives(self, executor):
        """
        Which it does not if an exception is allowed to cross the boundary.

        A traceback is discarded by pickling, so capturing it has to happen
        in the worker. Under the sequential executor this is free and the
        test is trivial; under the pool it is the whole point.
        """
        results = executor.map(items(["boom"], function=explode))
        assert "ValueError" in results[0].failure.traceback_text
        assert "explode" in results[0].failure.traceback_text


class TestTheLocalExecutorIsTheReference:
    """What the other executors are measured against."""

    def test_it_runs_in_the_calling_process(self):
        """Which is what makes it the only sane thing to debug with."""
        results = LocalExecutor().map(items([None], function=read_process_id))
        assert results[0].value == os.getpid()

    def test_the_pool_agrees_with_it(self):
        """
        Parity level 5 in miniature, over a trivial payload.

        The expensive version runs a real model once; this one runs on every
        commit, and catches an ordering or capture regression long before
        the gate does.
        """
        work = items([5, 1, 4, 2, 3])
        reference = LocalExecutor().map(work)
        pooled = ProcessExecutor(workers=POOL_WORKERS).map(work)

        assert [r.key for r in pooled] == [r.key for r in reference]
        assert [r.value for r in pooled] == [r.value for r in reference]
        assert [r.succeeded for r in pooled] == [r.succeeded for r in reference]


class TestTheProcessPool:
    """What only the pool can get wrong."""

    def test_work_actually_leaves_this_process(self):
        """
        The one thing no other test here would notice.

        A pool that silently ran everything in-process -- a fallback, a
        misconfiguration, a mocked-out context -- would pass every
        conformance rule above while delivering none of the parallelism.
        """
        results = ProcessExecutor(workers=POOL_WORKERS).map(
            items([None, None], function=read_process_id)
        )
        assert all(result.value != os.getpid() for result in results)

    def test_the_thread_budget_is_applied_in_the_worker(self):
        """
        Read back from the worker's own environment, not asserted in ours.

        This is the setting that decides whether a parallel run is faster or
        slower than a sequential one: eight workers each defaulting to every
        core produces eight times the core count in threads, and throughput
        collapses with nothing in any log to say why.
        """
        results = ProcessExecutor(workers=POOL_WORKERS, threads_per_worker=3).map(
            items([None], function=read_thread_budget)
        )
        assert results[0].value == dict.fromkeys(THREAD_VARIABLES, "3")

    def test_every_library_in_the_stack_is_capped(self):
        """
        Because a budget that capped some of them would be no budget.

        Whichever library was left uncapped would claim the whole machine on
        its own, and the one that does so varies with how NumPy was built --
        so the partial case is a bug that appears on one machine and not
        another.
        """
        assert {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"} <= set(
            THREAD_VARIABLES
        )

    def test_a_default_budget_divides_the_machine(self):
        """
        Rather than leaving the libraries' own defaults in place.

        "Leave it alone" means every worker claims every core, which is the
        failure this exists to prevent -- so the default has to be a real
        number, not an absence.
        """
        results = ProcessExecutor(workers=POOL_WORKERS).map(
            items([None], function=read_thread_budget)
        )
        budget = results[0].value["OMP_NUM_THREADS"]
        assert budget is not None
        assert int(budget) >= 1

    def test_the_worker_count_never_exceeds_the_work(self):
        """
        A surplus worker has nothing to do but consume memory.

        Not observable from the results, so it is asserted against the
        resolution directly.
        """
        pool = ProcessExecutor(workers=32)
        assert pool._resolve_workers(n_items=3) == 3

    def test_spawn_is_the_default_start_method(self):
        """
        Rather than `fork`.

        `fork` is unsafe once a threaded library or a GPU context exists in
        the parent, and the failure is an intermittent hang rather than an
        error -- close to the hardest thing there is to attribute to a cause.
        """
        assert ProcessExecutor().start_method == "spawn"


class TestDeviceVisibility:
    """
    Pinning each worker to its own accelerator.

    The parts that can be wrong in an interesting way -- the variable's name,
    its value, and whether it is set before the worker imports anything --
    are all observable on a machine with no accelerator at all, by reading
    the environment back from inside a worker. Only the question of whether
    a GPU library then honours it needs real hardware, and that is one
    assertion.
    """

    def test_a_worker_is_pinned_to_a_declared_device(self):
        """
        One identifier per worker, handed out through a shared queue.

        A pool initialiser receives identical arguments for every worker, so
        there is no index to assign a device from -- which is why this needs
        a queue rather than an argument, and why it is worth a test.

        Asserted as "whatever ran, ran somewhere declared" rather than as
        "both devices were used", because a pool creates workers on demand:
        with items this cheap the first worker can finish all of them before
        a second exists. That is correct pool behaviour, so a test demanding
        otherwise would be flaky by construction. The even-handedness of the
        distribution is asserted below, where no scheduler is involved.
        """
        pool = ProcessExecutor(workers=2, device_ids=(0, 1))
        results = pool.map(
            items([CUDA_VISIBILITY_VARIABLE] * 4, function=read_environment_variable)
        )
        observed = {result.value for result in results}
        assert observed
        assert observed <= {"0", "1"}

    def test_visibility_is_not_set_when_it_is_not_managed(self):
        """
        A CPU pool must not invent a device setting.

        Setting it to anything -- including an empty string, which means
        "no devices" -- would change the behaviour of a job that was
        correctly configured to use one.
        """
        inherited = os.environ.get(CUDA_VISIBILITY_VARIABLE)
        results = ProcessExecutor(workers=1).map(
            items([CUDA_VISIBILITY_VARIABLE], function=read_environment_variable)
        )
        assert results[0].value == inherited

    def test_more_workers_than_devices_pack_evenly(self):
        """
        Several small jobs per card is a legitimate configuration.

        It is the caller's decision to make, so the executor packs rather
        than refusing -- but it must pack evenly rather than putting every
        surplus worker on device zero.

        Checked against the assignment directly, because that is where the
        decision is made. Observing it through a running pool would mean
        waiting for every worker to be created, which a pool does not
        promise.
        """
        pool = ProcessExecutor(workers=4, device_ids=(0, 1))
        assert pool.device_assignment(workers=4) == (0, 1, 0, 1)

    def test_the_gpu_executor_refuses_to_run_without_a_device(self):
        """
        Rather than falling back to the CPU.

        A silent fallback turns a four-hour run into a four-day one, and the
        only evidence is the wall time -- by which point the run is days old
        and the cause is three layers down.
        """
        with pytest.raises(ValueError, match="at least one device"):
            GpuExecutor(device_ids=())

    def test_the_gpu_executor_defaults_to_one_worker_per_device(self):
        """The normal arrangement, so it should need no configuration."""
        assert GpuExecutor(device_ids=(0, 1, 2)).workers == 3

    @pytest.mark.skipif(not visible_device_ids(), reason="no accelerator declared on this machine")
    def test_a_worker_sees_exactly_one_device(self):
        """
        The one assertion that genuinely needs hardware.

        Skipped rather than omitted: a test that does not exist cannot be
        run on the machine that has the hardware, and this is the assertion
        that would catch the variable being set too late to take effect.
        """
        devices = visible_device_ids()
        results = GpuExecutor(device_ids=devices).map(
            items([CUDA_VISIBILITY_VARIABLE] * len(devices), function=read_environment_variable)
        )
        assert all("," not in str(result.value) for result in results)


class TestDiscoveringDevices:
    """Reading the declared accelerators without importing one."""

    def test_an_unset_variable_means_none(self, monkeypatch):
        """
        The safe way round.

        A set that runs on the CPU when it could have used a GPU is slow; a
        set that assumes four GPUs it does not have fails outright.
        """
        monkeypatch.delenv(CUDA_VISIBILITY_VARIABLE, raising=False)
        assert visible_device_ids() == ()

    def test_ordinals_are_read_in_order(self, monkeypatch):
        """Because the order is what assigns a worker to a device."""
        monkeypatch.setenv(CUDA_VISIBILITY_VARIABLE, "2,0,1")
        assert visible_device_ids() == (2, 0, 1)

    def test_uuids_are_not_mistaken_for_ordinals(self, monkeypatch):
        """
        The variable also accepts GPU UUIDs, which cannot be counted.

        Parsing one as an ordinal would produce a device list this framework
        could index but the driver could not, so the honest answer is that
        there is nothing here it can order.
        """
        monkeypatch.setenv(CUDA_VISIBILITY_VARIABLE, "GPU-8d6a1f3e,GPU-2b7c9e01")
        assert visible_device_ids() == ()

    def test_whitespace_and_empty_entries_are_tolerated(self, monkeypatch):
        """A hand-edited variable should not be a parse failure."""
        monkeypatch.setenv(CUDA_VISIBILITY_VARIABLE, " 0, 1, ")
        assert visible_device_ids() == (0, 1)


class TestResultValues:
    """The values an executor hands back."""

    def test_unwrap_returns_the_value(self):
        """For a caller that genuinely cannot continue without it."""
        assert WorkResult(key="a", value=7).unwrap() == 7

    def test_unwrap_raises_with_the_worker_s_traceback(self):
        """
        As text, because the original exception cannot be re-raised.

        It was captured in another process and only its rendering survived,
        so a caller re-raising the "same" exception would be raising a
        reconstruction -- better to be explicit that this is a report.
        """
        failure = WorkFailure(kind="ValueError", message="boom", traceback_text="trace here")
        with pytest.raises(RuntimeError, match="trace here"):
            WorkResult(key="a", failure=failure).unwrap()

    def test_timing_is_recorded_for_a_failure_too(self):
        """
        A failed item still reports how long it took.

        "It failed after four hours" and "it failed immediately" call for
        completely different responses, and only one of them is a
        configuration mistake.
        """
        result = execute_item(WorkItem(key="a", function=explode, payload="boom"))
        assert not result.succeeded
        assert result.wall_seconds >= 0.0

    def test_a_failure_summarises_to_one_line(self):
        """For a log line and a manifest row, where a traceback will not fit."""
        failure = WorkFailure(kind="SpecError", message="no such job", traceback_text="...")
        assert failure.summary() == "SpecError: no such job"

    def test_a_failure_with_no_message_still_summarises(self):
        """Some exceptions carry nothing but their type."""
        assert WorkFailure(kind="KeyboardInterrupt", message="", traceback_text="").summary() == (
            "KeyboardInterrupt"
        )


class TestSummarising:
    """The aggregate view a job set reports."""

    def test_successes_and_failures_are_separated(self):
        """In input order within each, so a reader can match them to jobs."""
        results = LocalExecutor().map(items([0, 1, 2, 3], function=fail_unless_even))
        summary = ResultSummary(tuple(results))
        assert [r.key for r in summary.succeeded] == ["0", "2"]
        assert [r.key for r in summary.failed] == ["1", "3"]

    def test_the_total_is_the_sum_of_the_items(self):
        """
        Not the elapsed time of the set.

        Under a pool those differ by the parallel speed-up, and this is the
        figure that is comparable between a sequential and a parallel run of
        the same work.
        """
        summary = ResultSummary(
            (
                WorkResult(key="a", value=1, wall_seconds=1.5),
                WorkResult(key="b", value=2, wall_seconds=2.5),
            )
        )
        assert summary.wall_seconds == pytest.approx(4.0)
