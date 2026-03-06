"""
Hyperparameter tuning: Optuna-backed search with analytics and plotting.
"""
from src.rade_ml_pt.tuning.tuner import Tuner, TuningResult
from src.rade_ml_pt.tuning.plots import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice,
)

__all__ = [
    "Tuner",
    "TuningResult",
    "plot_optimization_history",
    "plot_param_importances",
    "plot_parallel_coordinate",
    "plot_contour",
    "plot_slice",
]
