from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional
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

        fig.savefig(str(out_dir / name), dpi=int(cfg.dpi), bbox_inches="tight")

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