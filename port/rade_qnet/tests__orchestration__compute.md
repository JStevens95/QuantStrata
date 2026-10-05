# `tranql/models/rade/rade_qnet/tests/orchestration/compute`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 17 | 833 | `31dd72a61b33b7be` |
| 2 | `test_compute_executors.py` | 486 | 19221 | `0d8b5b6ec5992161` |
| 3 | `test_compute_placement.py` | 285 | 11532 | `fe7229d72769ca53` |
| 4 | `workers.py` | 149 | 3403 | `84d601ed094a99f4` |

---

## 1. `tranql/models/rade/rade_qnet/tests/orchestration/compute/__init__.py`

833 bytes · SHA-256 `31dd72a61b33b7be`

```python
"""
Tests for the execution layer.

Almost everything here is testing a property that holds *across* executors
rather than a property of one of them. That shape is deliberate: the
proposition this phase rests on is that placement cannot change results, and
a test written against one executor cannot say anything about it.

So the conformance rules -- input ordering, returned failures, repeatability
-- are parametrised over every executor, and the sequential one is treated as
the reference the others are compared against.

Tests that spawn processes are kept deliberately cheap: trivial payloads, two
workers, no model. A pool test slow enough to be annoying is a pool test
somebody eventually marks as skipped, and the gate goes with it. The one
place a real model runs under a real pool is parity level 5, which runs once.
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/orchestration/compute/test_compute_executors.py`

19221 bytes · SHA-256 `0d8b5b6ec5992161`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.base import (
    Executor,
    ResultSummary,
    WorkFailure,
    WorkItem,
    WorkResult,
    execute_item,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.gpus import (
    GpuExecutor,
    visible_device_ids,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.local import LocalExecutor
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.processes import (
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
```

---

## 3. `tranql/models/rade/rade_qnet/tests/orchestration/compute/test_compute_placement.py`

11532 bytes · SHA-256 `fe7229d72769ca53`

```python
"""
Tests for the placement policy.

`executor: auto` is the default, so this module is what the default *means*.
A policy that chose badly would be hard to notice -- the run still produces
the right numbers, just slowly, or not in parallel at all -- so the tests
assert the choice directly rather than through an outcome.

The hardware is simulated. Asserting against the machine the tests happen to
run on would mean the suite passing on a laptop and failing on a build
server, which is the opposite of what a test of a hardware-dependent decision
should do.
"""

from __future__ import annotations

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.jobs import PlacementSpec
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute import placement as chooser
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.gpus import GpuExecutor
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.local import LocalExecutor
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.placement import (
    available_memory_gb,
    choose_placement,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.processes import ProcessExecutor


@pytest.fixture
def machine(monkeypatch):
    """
    Return a function that simulates a machine's cores, devices and memory.

    Returns
    -------
    Callable
        Call it with the hardware to pretend the policy can see.
    """

    def configure(*, cores=8, devices=(), memory_gb=64.0):
        """Pretend the policy is running on this hardware."""
        monkeypatch.setattr(chooser.os, "cpu_count", lambda: cores)
        monkeypatch.setattr(chooser, "visible_device_ids", lambda: devices)
        monkeypatch.setattr(chooser, "available_memory_gb", lambda: memory_gb)

    return configure


class TestChoosingAnExecutor:
    """What `auto` resolves to."""

    def test_one_job_runs_sequentially(self, machine):
        """
        A pool for one job is pure overhead.

        It adds a process launch, a pickle round trip and an unusable
        debugger, in exchange for no parallelism at all.
        """
        machine(cores=8)
        placement = choose_placement(PlacementSpec(), n_jobs=1)
        assert isinstance(placement.executor, LocalExecutor)
        assert placement.name == "local"

    def test_several_jobs_on_a_cpu_machine_use_processes(self, machine):
        """The ordinary case, and the one most runs will take."""
        machine(cores=8, devices=())
        placement = choose_placement(PlacementSpec(), n_jobs=10)
        assert isinstance(placement.executor, ProcessExecutor)
        assert placement.name == "processes"

    def test_several_jobs_with_several_devices_use_the_gpus(self, machine):
        """Because a job that can use a card should have one to itself."""
        machine(cores=8, devices=(0, 1, 2, 3))
        placement = choose_placement(PlacementSpec(), n_jobs=10)
        assert isinstance(placement.executor, GpuExecutor)
        assert placement.name == "gpus"

    def test_a_single_device_does_not_trigger_the_gpu_executor(self, machine):
        """
        One card is not a reason to pay for a pool of one.

        The GPU executor exists to put one job on each of several devices.
        With one device it would be a process pool with extra steps, and the
        job would reach the card perfectly well through the ordinary path.
        """
        machine(cores=8, devices=(0,))
        placement = choose_placement(PlacementSpec(), n_jobs=10)
        assert placement.name == "processes"

    def test_asking_for_one_worker_means_sequential(self, machine):
        """
        Because that is what the user said, in their own words.

        A pool of one is strictly worse than no pool: the same serialism,
        plus a process boundary that makes debugging impossible.
        """
        machine(cores=8)
        placement = choose_placement(PlacementSpec(workers=1), n_jobs=10)
        assert isinstance(placement.executor, LocalExecutor)


class TestHonouringAnExplicitChoice:
    """`auto` is a default, not a policy that overrides the user."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("local", LocalExecutor), ("processes", ProcessExecutor)],
    )
    def test_a_named_executor_is_built(self, machine, name, expected):
        """Even where the policy would have chosen otherwise."""
        machine(cores=8, devices=(0, 1, 2, 3))
        placement = choose_placement(PlacementSpec(executor=name), n_jobs=10)
        assert isinstance(placement.executor, expected)
        assert placement.name == name

    def test_an_explicit_worker_count_is_not_capped(self, machine):
        """
        The ceiling applies to the policy's own choice, not to the user's.

        Somebody asking for sixty-four workers on a machine that can take
        them should get them; the cap exists to stop an unaided policy from
        doing something unreasonable, not to overrule a decision.
        """
        machine(cores=128)
        placement = choose_placement(PlacementSpec(workers=64), n_jobs=200)
        assert placement.executor.workers == 64


class TestWorkerCounts:
    """How many, when nobody said."""

    def test_never_more_workers_than_jobs(self, machine):
        """A surplus worker has nothing to do but consume memory."""
        machine(cores=64)
        placement = choose_placement(PlacementSpec(), n_jobs=3)
        assert placement.executor.workers == 3

    def test_never_more_workers_than_cores(self, machine):
        """Past that point the workers contend rather than parallelise."""
        machine(cores=4)
        placement = choose_placement(PlacementSpec(), n_jobs=100)
        assert placement.executor.workers == 4

    def test_the_automatic_count_has_a_ceiling(self, machine):
        """
        A pool far wider than this is almost always a mistake unaided.

        The jobs contend for memory and for the filesystem, and the
        per-worker thread budget reaches one long before the worker count
        does -- so the extra workers buy nothing and cost memory.
        """
        machine(cores=256)
        placement = choose_placement(PlacementSpec(), n_jobs=256)
        assert placement.executor.workers == chooser._MAX_AUTOMATIC_WORKERS


class TestMemoryCapping:
    """Reducing the worker count to what the machine can hold."""

    def test_a_per_job_estimate_caps_the_workers(self, machine):
        """
        Eight workers at 20 GiB each will not fit in 64 GiB.

        The alternative is an out-of-memory kill, which arrives as a signal
        with no explanation attached and takes the whole worker's run with
        it.
        """
        machine(cores=16, memory_gb=64.0)
        placement = choose_placement(PlacementSpec(memory_per_job_gb=20.0), n_jobs=16)
        # Three-quarters of 64 GiB is 48; at 20 GiB per job that is two.
        assert placement.executor.workers == 2

    def test_headroom_is_held_back(self, machine):
        """
        A quarter of the machine, because a job's peak is an estimate.

        The operating system needs room too, and erring low costs some
        throughput while erring high costs an entire job.
        """
        machine(cores=16, memory_gb=64.0)
        uncapped = choose_placement(PlacementSpec(), n_jobs=16).executor.workers
        capped = choose_placement(PlacementSpec(memory_per_job_gb=8.0), n_jobs=16).executor.workers
        assert capped == 6
        assert capped < uncapped

    def test_the_cap_never_reaches_zero(self, machine):
        """
        One worker that might be killed still beats no workers at all.

        A cap of zero would make the set unrunnable rather than risky, and
        the user would have no way to proceed without editing a figure they
        supplied in good faith.
        """
        machine(cores=16, memory_gb=4.0)
        placement = choose_placement(PlacementSpec(memory_per_job_gb=1000.0), n_jobs=16)
        assert placement.executor.workers == 1

    def test_no_estimate_means_no_cap(self, machine):
        """
        Because the framework cannot estimate a job's memory for the user.

        Guessing would produce a confident, wrong cap -- which is worse than
        none, since no cap at least fails visibly.
        """
        machine(cores=8, memory_gb=1.0)
        placement = choose_placement(PlacementSpec(), n_jobs=8)
        assert placement.executor.workers == 8

    def test_unknown_memory_means_no_cap(self, machine):
        """
        On a platform where the total cannot be read without a dependency.

        Returning `None` rather than a guess is what makes this safe: the
        cap is skipped, rather than applied against a fabricated figure.
        """
        machine(cores=8, memory_gb=None)
        placement = choose_placement(PlacementSpec(memory_per_job_gb=1000.0), n_jobs=8)
        assert placement.executor.workers == 8


class TestReadingTheMachine:
    """The two things the policy looks at directly."""

    def test_memory_is_a_positive_figure_or_nothing(self):
        """
        Either a real figure or an honest refusal to answer.

        Run against the real machine, because that is the code path a user
        takes and the one platform differences show up in.
        """
        total = available_memory_gb()
        assert total is None or total > 0.0

    def test_the_machine_description_is_renderable(self):
        """It goes into a log line and a manifest, so it must always exist."""
        assert "core(s)" in chooser.describe_machine()

    def test_unknown_memory_still_describes_the_machine(self, machine):
        """
        Rather than omitting the line or printing `None GiB`.

        The description is the only record of what the policy was looking
        at, so it has to survive the policy not knowing something.
        """
        machine(cores=8, memory_gb=None)
        assert "memory unknown" in chooser.describe_machine()


class TestTheReasonIsRecorded:
    """
    Why the choice was made, carried alongside the choice.

    A set that chose its own placement and a set that was told one are not
    the same experiment, and six months later the manifest is the only thing
    that remembers which this was. A set that ran slowly is otherwise a
    mystery: the configuration says `auto` and the output says nothing.
    """

    def test_an_automatic_choice_explains_itself(self, machine):
        """Naming the job count, which is the input that drove it."""
        machine(cores=8)
        assert "1 job" in choose_placement(PlacementSpec(), n_jobs=1).reason

    def test_an_explicit_choice_says_so(self, machine):
        """
        So nobody later attributes the user's decision to the policy.

        These call for different fixes: a bad automatic choice is a policy
        bug, and a bad explicit one is a configuration change.
        """
        machine(cores=8)
        reason = choose_placement(PlacementSpec(executor="local"), n_jobs=10).reason
        assert "explicitly" in reason

    def test_the_reason_names_the_hardware_it_saw(self, machine):
        """
        Because the same policy on two machines makes two different choices.

        Without this, a manifest from a build server and one from a
        workstation would look identically configured and have run quite
        differently.
        """
        machine(cores=8, memory_gb=64.0)
        assert "8 core(s)" in choose_placement(PlacementSpec(), n_jobs=10).reason
```

---

## 4. `tranql/models/rade/rade_qnet/tests/orchestration/compute/workers.py`

3403 bytes · SHA-256 `84d601ed094a99f4`

```python
"""
Work functions for the executor tests.

In their own module, and module-level within it, because that is the
constraint being tested. A function defined inside a test -- a closure, a
lambda, a local `def` -- cannot cross a process boundary under the spawn
start method, so a test that used one would pass sequentially and fail under
the pool for a reason that has nothing to do with the executor.

Keeping them here also means the test module itself needs nothing picklable
in it, so a future test can use a fixture or a closure freely without
accidentally breaking the pool tests next to it.

Every function here is trivial and fast. A pool test slow enough to be
annoying is a pool test somebody eventually skips.
"""

from __future__ import annotations

import os

from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.processes import THREAD_VARIABLES


def double(value: int) -> int:
    """
    Return twice the value.

    Parameters
    ----------
    value
        Any integer.

    Returns
    -------
    int
        Twice it.
    """
    return value * 2


def explode(message: str) -> int:
    """
    Raise a ``ValueError`` carrying the message.

    Parameters
    ----------
    message
        What the error should say.

    Returns
    -------
    int
        Never; the annotation exists so the item type-checks alongside the
        functions it is mixed with.

    Raises
    ------
    ValueError
        Always.
    """
    raise ValueError(message)


def fail_unless_even(value: int) -> int:
    """
    Return the value, failing on odd inputs.

    Used for partial-failure tests, where some items must succeed and others
    must not within a single call.

    Parameters
    ----------
    value
        Any integer.

    Returns
    -------
    int
        The value, if it is even.

    Raises
    ------
    ValueError
        If the value is odd.
    """
    if value % 2:
        message = f"{value} is odd"
        raise ValueError(message)
    return value


def read_thread_budget(_: object) -> dict[str, str | None]:
    """
    Report the thread-budget variables as this worker sees them.

    The only way to check a budget that is applied in another process. An
    assertion made in the parent would be testing the parent's environment,
    which the executor never touches.

    Parameters
    ----------
    _
        Ignored; the item interface requires a payload.

    Returns
    -------
    dict
        Variable name to value, or ``None`` where unset.
    """
    return {variable: os.environ.get(variable) for variable in THREAD_VARIABLES}


def read_environment_variable(name: str) -> str | None:
    """
    Report one environment variable as this worker sees it.

    Parameters
    ----------
    name
        The variable to read.

    Returns
    -------
    str or None
        Its value, or ``None`` if unset.
    """
    return os.environ.get(name)


def read_process_id(_: object) -> int:
    """
    Report this worker's process identifier.

    Used to confirm that work actually left the calling process, which no
    other observation establishes -- a pool that silently ran everything
    in-process would pass every other test here.

    Parameters
    ----------
    _
        Ignored; the item interface requires a payload.

    Returns
    -------
    int
        The process identifier.
    """
    return os.getpid()
```

