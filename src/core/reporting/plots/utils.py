from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from matplotlib.figure import Figure

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PlotConfig:
    show: bool = True
    save: bool = False
    out_dir: Path = Path("outputs")
    filename: Optional[str] = None   # optional default name
    dpi: int = 150
    block: bool = True               # keeps IDE windows alive
    close: bool = False              # close figures after rendering
    save_pdf: bool = True            # when save=True, also save PDF for report quality

def ensure_dir(path: PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_fig(fig: Figure, path: PathLike, *, dpi: int = 160) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, dpi=int(dpi), bbox_inches="tight")
    return p


def render_fig(fig, *, cfg: PlotConfig, filename: Optional[str] = None) -> None:
    """
    Render a matplotlib figure according to cfg:
      - save if cfg.save
      - show if cfg.show
    """
    # ---- Save (optional) ----
    if cfg.save:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        name = filename or cfg.filename
        if not name:
            raise ValueError("render_fig: filename must be provided when cfg.save=True.")

        base_path = out_dir / name
        fig.savefig(str(base_path), dpi=int(cfg.dpi), bbox_inches="tight")
        if cfg.save_pdf:
            pdf_path = base_path.with_suffix(".pdf")
            fig.savefig(str(pdf_path), bbox_inches="tight")

    # ---- Show (optional) ----
    if cfg.show:
        # In many IDEs, block=True avoids “flash then disappear”
        plt.show(block=bool(cfg.block))

    # ---- Close (optional) ----
    if cfg.close:
        plt.close(fig)


def display_fig(fig=None, *, block: bool = True) -> None:
    """
    Display a matplotlib figure. In scripts, block=True keeps the window open.
    """

    if fig is not None:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
    plt.show(block=block)


def close_fig(fig: Figure) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass


def save_report_figures(
    figures: List[Tuple[Figure, str]],
    out_dir: PathLike,
    *,
    prefix: str = "report",
    dpi: int = 160,
    save_pdf: bool = True,
) -> List[Path]:
    """
    Save a batch of figures with consistent naming for reports.

    Names are prefix_01_name.pdf, prefix_02_name.pdf, ... (and .png if dpi used).
    Returns list of saved paths (first format per figure).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for i, (fig, name) in enumerate(figures, start=1):
        stem = f"{prefix}_{i:02d}_{name}"
        png_path = out / f"{stem}.png"
        fig.savefig(str(png_path), dpi=int(dpi), bbox_inches="tight")
        saved.append(png_path)
        if save_pdf:
            fig.savefig(str(out / f"{stem}.pdf"), bbox_inches="tight")
    return saved