"""
Evaluation plots: residual diagnostics, prediction analysis, and model comparison.

Usage:
    from rade_ml_pt.evaluation.plots import plot_residual_distribution, plot_predicted_vs_actual
"""
from src.rade_ml_pt.evaluation.plots.residuals import (
    plot_residual_distribution,
    plot_qq,
    plot_residual_scatter,
    plot_residual_by_target,
)
from src.rade_ml_pt.evaluation.plots.predictions import (
    plot_predicted_vs_actual,
    plot_error_distribution,
    plot_cumulative_error,
    plot_prediction_timeseries,
)

__all__ = [
    # residuals
    "plot_residual_distribution",
    "plot_qq",
    "plot_residual_scatter",
    "plot_residual_by_target",
    # predictions
    "plot_predicted_vs_actual",
    "plot_error_distribution",
    "plot_cumulative_error",
    "plot_prediction_timeseries",
]
