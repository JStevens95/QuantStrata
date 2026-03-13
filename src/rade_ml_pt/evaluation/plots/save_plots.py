"""
Save evaluation plots to disk without displaying (for use in pipelines).

Uses a non-interactive backend and closes figures after saving so the pipeline
can run headless. Skips plots when required data is missing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import EvaluationResult

logger = logging.getLogger(__name__)


def save_evaluation_plots(
    eval_result: "EvaluationResult",
    save_dir: Path,
) -> None:
    """
    Generate and save standard evaluation plots to *save_dir*.

    Uses Agg backend; no figures are displayed. Skips individual plots when
    required data (predictions, targets, residuals) is missing. Logs and
    continues on plot errors so one failure does not stop the rest.

    :param eval_result: EvaluationResult from Evaluator.run().
    :param save_dir: directory to write PNG files into (created if needed).
    """
    import matplotlib
    matplotlib.use("Agg")

    from src.rade_ml_pt.evaluation.plots.predictions import (
        plot_predicted_vs_actual,
        plot_error_distribution,
        plot_cumulative_error,
    )
    from src.rade_ml_pt.evaluation.plots.residuals import (
        plot_residual_distribution,
        plot_qq,
        plot_residual_scatter,
        plot_residual_by_target,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir)

    has_pred_tgt = (
        eval_result.predictions is not None
        and eval_result.targets is not None
    )
    has_residuals = eval_result.residuals is not None

    # Try to get target labels from metadata; fall back to default T0, T1, ...
    target_labels = None
    if eval_result.metadata:
        target_labels = eval_result.metadata.get("target_labels")

    plots = []

    if has_pred_tgt:
        plots.append(("predicted_vs_actual", lambda: plot_predicted_vs_actual(
            eval_result, save_path=save_path, show=False,
        )))
    if has_residuals:
        plots.append(("residual_distribution", lambda: plot_residual_distribution(
            eval_result, save_path=save_path, show=False,
        )))
        plots.append(("error_distribution", lambda: plot_error_distribution(
            eval_result, save_path=save_path, show=False,
        )))
        plots.append(("cumulative_error", lambda: plot_cumulative_error(
            eval_result, save_path=save_path, show=False,
        )))
        plots.append(("residual_qq", lambda: plot_qq(
            eval_result, save_path=save_path, show=False,
        )))
    if has_pred_tgt and has_residuals:
        plots.append(("residual_scatter", lambda: plot_residual_scatter(
            eval_result, save_path=save_path, show=False,
        )))
    if has_residuals:
        plots.append(("residual_by_target", lambda: plot_residual_by_target(
            eval_result, target_labels=target_labels,
            save_path=save_path, show=False,
        )))

    for name, fn in plots:
        try:
            fn()
        except Exception as exc:
            logger.warning("Could not generate evaluation plot %s: %s", name, exc)
