"""
Table data builders — prepare row data for AG Grid metric tables.

Return ``(column_defs, row_data)`` tuples ready for ``metric_table()``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def percentile_table_data(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a percentile breakdown table comparing predicted vs target
    distributions — shows Predicted, Target, and Diff at each percentile.

    Returns
    -------
    tuple of (column_defs, row_data)
    """
    fmt = {"function": "d3.format('.4f')(params.value)"}
    column_defs = [
        {"field": "percentile", "headerName": "Percentile"},
        {"field": "predicted", "headerName": "Predicted", "valueFormatter": fmt},
        {"field": "target", "headerName": "Target", "valueFormatter": fmt},
        {"field": "diff", "headerName": "Diff", "valueFormatter": fmt},
        {"field": "abs_error", "headerName": "Abs Error", "valueFormatter": fmt},
    ]

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    row_data = []
    for p in percentiles:
        pred_p = float(np.percentile(predictions, p))
        targ_p = float(np.percentile(targets, p))
        row_data.append({
            "percentile": f"P{p}",
            "predicted": pred_p,
            "target": targ_p,
            "diff": pred_p - targ_p,
            "abs_error": float(np.percentile(np.abs(predictions - targets), p)),
        })

    row_data.append({
        "percentile": "Mean",
        "predicted": float(predictions.mean()),
        "target": float(targets.mean()),
        "diff": float((predictions - targets).mean()),
        "abs_error": float(np.abs(predictions - targets).mean()),
    })

    return column_defs, row_data


def worst_scenarios_data(
    predictions: np.ndarray,
    targets: np.ndarray,
    top_n: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a table of the worst (highest absolute error) scenarios.

    Returns
    -------
    tuple of (column_defs, row_data)
    """
    residuals = predictions - targets
    abs_errors = np.abs(residuals)
    worst_idx = np.argsort(abs_errors)[::-1][:top_n]

    column_defs = [
        {"field": "rank", "headerName": "#", "width": 60},
        {"field": "scenario", "headerName": "Scenario"},
        {"field": "target", "headerName": "Target", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        {"field": "prediction", "headerName": "Prediction", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        {"field": "abs_error", "headerName": "Abs Error", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
    ]

    row_data = [
        {
            "rank": rank + 1,
            "scenario": int(idx),
            "target": float(targets[idx]),
            "prediction": float(predictions[idx]),
            "abs_error": float(abs_errors[idx]),
        }
        for rank, idx in enumerate(worst_idx)
    ]

    return column_defs, row_data
