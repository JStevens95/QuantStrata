"""
A consistent house style, applied without touching global state.

Why this module avoids ``pyplot`` entirely
------------------------------------------
Everything in ``visuals`` constructs :class:`matplotlib.figure.Figure` objects
directly rather than going through ``pyplot``, and the reason is not stylistic
preference.

``pyplot`` maintains a **global figure registry**. A figure created with
``plt.subplots()`` stays alive until explicitly closed, so a job set producing
twelve figures per member across three hundred members accumulates thirty-six
hundred live figures and exhausts memory. Worse, the warning matplotlib emits
about this is easy to suppress and easy to miss.

``pyplot`` also performs **global backend selection** at import. In a worker
process with no display, that has historically meant an attempted GUI backend
and a crash. Constructing a ``Figure`` requires no backend at all until the
figure is saved.

So: construct ``Figure`` directly, and let it be garbage collected like any
other object. :func:`figure_style` likewise uses ``rc_context``, which restores
the previous settings on exit, rather than mutating ``matplotlib.rcParams`` --
a framework that permanently changes a user's plotting defaults because they
imported it is a framework that has overstepped.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import matplotlib as mpl

__all__ = ["PALETTE", "RC_PARAMS", "figure_style"]

#: The house palette. Ordered so the first entries are the ones a two-series
#: plot will use, and chosen to stay distinguishable in greyscale -- figures
#: end up in printed documents more often than anyone plans for.
PALETTE: tuple[str, ...] = (
    "#1f4e79",  # deep blue: the primary series, usually the model
    "#c0504d",  # brick red: the comparison, usually the baseline or target
    "#4f81bd",  # mid blue
    "#9bbb59",  # olive
    "#8064a2",  # violet
    "#4bacc6",  # teal
    "#f79646",  # orange
    "#808080",  # grey: for de-emphasised or reference content
)

#: Style parameters applied inside :func:`figure_style`. Deliberately modest:
#: readable sizes, light gridlines, no chart junk.
RC_PARAMS: Mapping[str, object] = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


@contextmanager
def figure_style(overrides: Mapping[str, object] | None = None) -> Iterator[None]:
    """
    Apply the house style for the duration of a block.

    Uses ``rc_context``, so the caller's settings are restored on exit --
    including when the block raises. See the module docstring for why that
    matters.

    Parameters
    ----------
    overrides
        Additional or replacement rc parameters for this block only.

    Yields
    ------
    None
        The block runs with the style applied.

    Examples
    --------
    >>> from matplotlib.figure import Figure
    >>> with figure_style({"font.size": 14}):
    ...     figure = Figure(figsize=(4, 3))
    ...     axes = figure.add_subplot()
    ...     _ = axes.plot([0, 1], [1, 0])
    """
    parameters = dict(RC_PARAMS)
    if overrides:
        parameters.update(overrides)
    with mpl.rc_context(parameters):
        yield
