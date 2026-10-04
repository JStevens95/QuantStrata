"""
Writing figures to disk.

The only module in ``visuals`` that touches the filesystem.

Keeping writing here, and out of the figure factories, is what lets the same
factory serve a report, a notebook and a test. See
:mod:`rade_qnet.analysis.visuals.figures` for the other half of that rule.

Why this module closes figures
------------------------------
:func:`save_figure` calls ``figure.clf()`` after writing. A ``Figure`` built
directly is an ordinary object and will be collected eventually, but
"eventually" is doing a lot of work when a job set produces thousands of
figures holding array data -- the interpreter has no reason to collect
promptly, because the figures are small objects pointing at large buffers.
Clearing explicitly releases the buffers at a predictable point.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ...core.lifecycle.errors import SpecError
from ...core.provenance.logging import get_logger

__all__ = ["SUPPORTED_FORMATS", "save_figure"]

_LOGGER = get_logger(__name__)

#: Formats the framework will write. Restricted deliberately: every one of
#: these is lossless or vector, because a figure that will be read carefully
#: should not be a JPEG.
SUPPORTED_FORMATS = frozenset({"png", "svg", "pdf"})


def save_figure(
    figure: Figure,
    directory: Path,
    name: str,
    *,
    figure_format: str = "png",
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """
    Write a figure and return where it went.

    Parameters
    ----------
    figure
        The figure to write.
    directory
        Destination directory, created if absent.
    name
        File stem, without an extension. Any extension present is stripped,
        so passing ``"curve.png"`` with ``figure_format="svg"`` does not
        produce ``curve.png.svg``.
    figure_format
        One of :data:`SUPPORTED_FORMATS`.
    dpi
        Resolution, for raster formats. Ignored for ``svg`` and ``pdf``.
    close
        Whether to release the figure's buffers after writing. See the module
        docstring. Pass ``False`` when the caller intends to display the
        figure as well.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    SpecError
        If the format is not supported.
    """
    if figure_format not in SUPPORTED_FORMATS:
        raise SpecError(
            f"unsupported figure format {figure_format!r}; "
            f"expected one of {sorted(SUPPORTED_FORMATS)}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{Path(name).stem}.{figure_format}"
    figure.savefig(destination, format=figure_format, dpi=dpi, bbox_inches="tight")
    if close:
        # Releases the axes and the array buffers they reference. Figures are
        # not in a global registry here, so this is about promptness rather
        # than about avoiding a leak.
        figure.clf()
    _LOGGER.debug("wrote figure %s", destination)
    return destination
