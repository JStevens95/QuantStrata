"""
Tests for figure export.

This is the only module in ``visuals`` permitted to touch the filesystem, so
it is the only place these tests look for files. The division is what lets one
figure factory serve a report, a notebook and a test.

The behaviour most worth pinning is the clearing of figures after writing. A
figure built directly is an ordinary object and will be collected eventually,
but "eventually" is doing a lot of work when a job set produces thousands of
small objects pointing at large array buffers -- the interpreter has no
pressure to collect them promptly.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_xl.analysis.visuals.export import SUPPORTED_FORMATS, save_figure
from src.rade_xl.core.runtime.errors import SpecError


@pytest.fixture
def figure():
    """
    Provide a small figure with one line.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    built = Figure(figsize=(4, 3))
    built.add_subplot().plot(np.arange(10), np.arange(10))
    return built


class TestWriting:
    """Where the file goes, and what it is called."""

    def test_the_written_path_is_returned(self, figure, tmp_path):
        """
        So a caller can record it as an artifact without reconstructing it.

        Rebuilding the path at the call site is how a report ends up
        recording an artifact that does not exist.
        """
        assert save_figure(figure, tmp_path, "curve") == tmp_path / "curve.png"

    def test_the_file_is_actually_written(self, figure, tmp_path):
        """And is not empty."""
        assert save_figure(figure, tmp_path, "curve").stat().st_size > 0

    def test_the_directory_is_created(self, figure, tmp_path):
        """
        A report's figure directory does not exist before its first figure.

        Requiring the caller to create it would put the same two lines in
        every report.
        """
        destination = tmp_path / "reports" / "figures"
        assert save_figure(figure, destination, "curve").is_file()

    @pytest.mark.parametrize("figure_format", sorted(SUPPORTED_FORMATS))
    def test_each_supported_format_writes(self, figure, tmp_path, figure_format):
        """
        All three are lossless or vector.

        A figure that will be read carefully should not be a JPEG, which is
        why the set is restricted rather than passed through.
        """
        written = save_figure(figure, tmp_path, "curve", figure_format=figure_format)
        assert written.suffix == f".{figure_format}"
        assert written.is_file()

    def test_an_unsupported_format_is_rejected(self, figure, tmp_path):
        """
        Rather than handed to matplotlib to fail less helpfully.

        And the message lists what is available, since the usual cause is a
        configuration typo.
        """
        with pytest.raises(SpecError, match="pdf"):
            save_figure(figure, tmp_path, "curve", figure_format="jpeg")

    def test_an_extension_in_the_name_is_stripped(self, figure, tmp_path):
        """
        The format argument decides the extension, not the name.

        So ``"curve.png"`` with ``figure_format="svg"`` is not
        ``curve.png.svg``, which is an easy mistake to make from a report
        that builds its figure names from a template.
        """
        written = save_figure(figure, tmp_path, "curve.png", figure_format="svg")
        assert written.name == "curve.svg"

    def test_an_existing_file_is_replaced(self, figure, tmp_path):
        """
        Re-running a report over the same directory is normal.

        Refusing would mean a second run produces a half-updated report,
        which is worse than either outcome.
        """
        save_figure(figure, tmp_path, "curve")
        second = Figure(figsize=(4, 3))
        second.add_subplot().plot([0, 1], [1, 0])
        assert save_figure(second, tmp_path, "curve").is_file()


class TestResourceRelease:
    """Promptness, not leak avoidance."""

    def test_the_figure_is_cleared_by_default(self, figure, tmp_path):
        """
        Releasing the axes and the array buffers they reference.

        A job set producing thousands of figures would otherwise hold every
        buffer until the interpreter felt like collecting them.
        """
        save_figure(figure, tmp_path, "curve")
        assert figure.axes == []

    def test_clearing_can_be_declined(self, figure, tmp_path):
        """
        For a caller that intends to display the figure as well.

        A notebook writing a figure and then showing it should not get a
        blank one.
        """
        save_figure(figure, tmp_path, "curve", close=False)
        assert figure.axes != []

    def test_a_cleared_figure_is_still_reusable(self, figure, tmp_path):
        """
        ``clf`` clears, it does not destroy.

        So a caller that cleared by accident gets an empty figure rather
        than an exception from somewhere unrelated.
        """
        save_figure(figure, tmp_path, "curve")
        figure.add_subplot().plot([0, 1], [1, 0])
        assert save_figure(figure, tmp_path, "second").is_file()


class TestResolution:
    """Raster output has to be legible."""

    def test_a_higher_dpi_produces_a_larger_file(self, tmp_path):
        """
        Confirming the setting reaches matplotlib rather than being ignored.

        A dpi argument that quietly did nothing would leave every figure in
        a printed report blurry.
        """
        small = Figure(figsize=(4, 3))
        small.add_subplot().plot(np.arange(10), np.arange(10))
        low = save_figure(small, tmp_path, "low", dpi=50).stat().st_size

        large = Figure(figsize=(4, 3))
        large.add_subplot().plot(np.arange(10), np.arange(10))
        high = save_figure(large, tmp_path, "high", dpi=200).stat().st_size

        assert high > low

    def test_dpi_is_irrelevant_to_a_vector_format(self, figure, tmp_path):
        """
        Accepted and ignored, rather than rejected.

        A report writes every figure through one call, so the format and
        the dpi are set independently of each other.
        """
        assert save_figure(figure, tmp_path, "curve", figure_format="svg", dpi=600).is_file()
