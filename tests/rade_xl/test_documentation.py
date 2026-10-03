"""
Tests that the configuration examples in the documentation actually work.

``ARCHITECTURE.md`` §12 is the first thing anyone reads and the place they
copy their first configuration from. It had drifted: the job-set example was
written before Phase 1 settled the schema and used field names that no
longer existed, so a reader following it got a validation error on their
first run -- from the document that was supposed to be teaching them the
format.

Documentation drifts because nothing checks it. This checks it: the YAML is
extracted from the file and put through the real loader, so a schema change
that invalidates the example fails here rather than on a new user's machine.

Only the configuration blocks are checked, not the prose. A test that tried
to verify explanatory text would be a test nobody could keep passing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.rade_xl.core.spec.jobs import parse_job_set_spec

#: The document under test.
ARCHITECTURE = Path("src/rade_xl/docs/ARCHITECTURE.md")

#: The example's filename, used to find its block rather than relying on
#: the block's position -- which changes whenever a section is added above.
JOB_SET_EXAMPLE = "configs/hybrid_portfolio.yaml"


@pytest.fixture(scope="module")
def job_set():
    """
    Parse the job-set example out of the architecture document.

    Returns
    -------
    JobSetSpec
        The validated job set.
    """
    # Imported for its registration side effect: the example names the
    # flagship, and validating it means resolving that name.
    import src.rade_xl.models.hybrid_gnn_rnn.register  # noqa: F401, PLC0415

    text = ARCHITECTURE.read_text(encoding="utf-8")
    match = re.search(rf"```yaml\n# {re.escape(JOB_SET_EXAMPLE)}\n(.*?)```", text, re.DOTALL)
    assert match is not None, f"no yaml block for {JOB_SET_EXAMPLE} in {ARCHITECTURE}"

    return parse_job_set_spec(yaml.safe_load(match.group(1)))


class TestTheDocumentedJobSet:
    """The example a reader copies first."""

    def test_it_parses(self, job_set):
        """
        The minimum bar, and the one the example previously failed.

        A document teaching a file format should produce a file that
        loads.
        """
        assert job_set.job_ids == ("FX__G10", "FX__EM", "RATES__USD")

    def test_every_job_validates(self, job_set):
        """
        Parsing is not enough: the jobs have to be real run specifications.

        A job set validates lazily enough that a malformed job survives
        loading and fails at dispatch, which is exactly the drift this
        guards against.
        """
        assert set(job_set.validate_jobs()) == {"FX__G10", "FX__EM", "RATES__USD"}

    def test_the_prose_about_merging_is_true(self, job_set):
        """
        The document claims naming one field leaves its siblings alone.

        `FX__EM` changes two model parameters. If the claim were false it
        would have lost the set's epochs, its thread budget and its
        sequence length -- so this asserts the explanation, not just the
        syntax.
        """
        run = job_set.validate_jobs()["FX__EM"]

        assert run.model.params["units"] == 64
        assert run.training.epochs == 200
        assert run.hardware.threads_per_worker == 1
        assert run.source.transforms.sequence.length == 20

    def test_the_thread_budget_is_pinned_as_the_text_says(self, job_set):
        """
        The example is also an instruction about reproducibility.

        It pins the budget and explains why; a later edit that dropped the
        field would leave the explanation attached to a file that no
        longer does it.
        """
        assert all(
            job_set.run_spec_for(job_id).hardware.threads_per_worker is not None
            for job_id in job_set.job_ids
        )

    def test_the_per_cluster_complexity_differs(self, job_set):
        """
        The point the example is making.

        An example where every job had the same model would illustrate
        nothing about why a portfolio is a job set.
        """
        widths = {
            job_id: job_set.run_spec_for(job_id).model.params["units"] for job_id in job_set.job_ids
        }

        assert len(set(widths.values())) > 1
