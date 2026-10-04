"""
Tests for the house plotting style.

The claim being protected is that importing ``rade_qnet`` does not change
anybody's matplotlib defaults. A framework that permanently restyled a user's
plots because they imported it has overstepped, and the failure is
particularly annoying because it shows up in unrelated figures much later.

The second claim is that nothing here touches ``pyplot``. That module keeps a
global figure registry, so a job set producing twelve figures across three
hundred members would accumulate thirty-six hundred live figures and exhaust
memory -- and it selects a backend at import, which in a headless worker has
historically meant a crash.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis import visuals
from src.rade_qnet.analysis.visuals.style import PALETTE, RC_PARAMS, figure_style


class TestTheStyleIsTemporary:
    """``rc_context``, not mutation."""

    def test_settings_are_applied_inside_the_block(self):
        """Otherwise the module would do nothing at all."""
        with figure_style():
            assert mpl.rcParams["axes.grid"] is True

    def test_settings_are_restored_on_exit(self):
        """
        The central guarantee.

        A user's own defaults must survive importing and using this
        framework.
        """
        before = mpl.rcParams["font.size"]
        with figure_style({"font.size": 42}):
            pass
        assert mpl.rcParams["font.size"] == before

    def test_settings_are_restored_after_an_exception(self):
        """
        A failed plot must not leave the style applied.

        Which is precisely what a bare ``rcParams.update`` would do, and the
        reason a context manager is used.
        """
        before = mpl.rcParams["font.size"]
        with pytest.raises(RuntimeError), figure_style({"font.size": 42}):
            raise RuntimeError("plotting failed")
        assert mpl.rcParams["font.size"] == before

    def test_importing_the_module_changes_nothing(self):
        """
        The style is opted into, never applied on import.

        A user whose unrelated notebook figures changed appearance because
        of a transitive import would be right to be annoyed.
        """
        assert mpl.rcParams["axes.spines.top"] is True


class TestOverrides:
    """Per-block customisation, scoped to the block."""

    def test_an_override_takes_effect(self):
        """A report may legitimately want a larger font."""
        with figure_style({"font.size": 14}):
            assert mpl.rcParams["font.size"] == 14

    def test_an_override_does_not_disturb_the_rest_of_the_style(self):
        """
        Overrides are merged, not substituted.

        Replacing the whole set would make a one-key override silently drop
        the grid, the spine settings and the palette.
        """
        with figure_style({"font.size": 14}):
            assert mpl.rcParams["axes.grid"] is True

    def test_an_override_does_not_mutate_the_shared_defaults(self):
        """
        The merge must copy.

        Otherwise one report's override would leak into every later figure
        in the process, which is the hardest kind of styling bug to trace.
        """
        with figure_style({"font.size": 14}):
            pass
        assert RC_PARAMS["font.size"] == 10


class TestThePalette:
    """Chosen for where figures actually end up."""

    def test_the_palette_drives_the_colour_cycle(self):
        """So two figures in one report agree on what "model" looks like."""
        with figure_style():
            cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        assert tuple(cycle) == PALETTE

    def test_every_colour_is_distinct(self):
        """A repeated colour makes two series indistinguishable."""
        assert len(set(PALETTE)) == len(PALETTE)

    def test_the_first_two_colours_differ_in_lightness(self):
        """
        Figures end up in printed documents more often than anyone plans.

        The two-series case is the common one, so those two have to survive
        greyscale conversion rather than merging into one shade.
        """

        def luminance(colour):
            """
            Return a rough perceptual lightness for a hex colour.

            Parameters
            ----------
            colour
                A ``#rrggbb`` string.

            Returns
            -------
            float
                Weighted lightness in ``[0, 255]``.
            """
            red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
            return 0.299 * red + 0.587 * green + 0.114 * blue

        assert abs(luminance(PALETTE[0]) - luminance(PALETTE[1])) > 20


class TestNoPyplot:
    """Figures are ordinary objects, not registry entries."""

    def test_a_figure_built_under_the_style_is_not_registered(self):
        """
        The property that stops a job set exhausting memory.

        A ``pyplot`` figure stays alive until explicitly closed; this one is
        collected like anything else.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415 - imported only to observe the registry

        before = plt.get_fignums()
        with figure_style():
            figure = Figure(figsize=(4, 3))
            figure.add_subplot().plot([0, 1], [1, 0])
        assert plt.get_fignums() == before

    @pytest.mark.parametrize(
        "module_path",
        sorted(Path(visuals.__file__).parent.glob("*.py")),
        ids=lambda path: path.name,
    )
    def test_no_module_in_visuals_imports_pyplot(self, module_path):
        """
        Checked across the package, because one import is enough.

        ``pyplot`` selects a backend at import time, which in a headless
        worker has historically meant an attempted GUI backend and a crash
        -- and it does so for every caller, not just the module that
        imported it.

        Read from the syntax tree rather than from the text, so a mention in
        a docstring (this file's own module docstring, for instance) is not
        mistaken for an import.
        """
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported = {
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("pyplot" in (name or "") for name in imported)
