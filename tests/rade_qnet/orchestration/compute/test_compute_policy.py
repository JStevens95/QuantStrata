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

from src.rade_qnet.core.spec.jobs import PlacementSpec
from src.rade_qnet.orchestration.compute import policy
from src.rade_qnet.orchestration.compute.gpus import GpuExecutor
from src.rade_qnet.orchestration.compute.local import LocalExecutor
from src.rade_qnet.orchestration.compute.policy import available_memory_gb, choose_placement
from src.rade_qnet.orchestration.compute.processes import ProcessExecutor


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
        monkeypatch.setattr(policy.os, "cpu_count", lambda: cores)
        monkeypatch.setattr(policy, "visible_device_ids", lambda: devices)
        monkeypatch.setattr(policy, "available_memory_gb", lambda: memory_gb)

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
        assert placement.executor.workers == policy._MAX_AUTOMATIC_WORKERS


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
        assert "core(s)" in policy.describe_machine()

    def test_unknown_memory_still_describes_the_machine(self, machine):
        """
        Rather than omitting the line or printing `None GiB`.

        The description is the only record of what the policy was looking
        at, so it has to survive the policy not knowing something.
        """
        machine(cores=8, memory_gb=None)
        assert "memory unknown" in policy.describe_machine()


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
