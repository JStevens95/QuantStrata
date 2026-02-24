"""
Professional training report generation for rade ML framework.

Produces model-independent Markdown reports with full training details, config,
data summary, and loss curves. Designed for audit trails and experiment documentation.

Plot generation is delegated to training.plots.
"""
from __future__ import annotations

import hashlib
import json
import logging
from io import StringIO
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from src.rade_ml.core.types import TrainingResult
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.training.plots import save_training_plots

logger = logging.getLogger(__name__)


def generate_training_report(
    result: TrainingResult,
    config: PipelineConfig,
    save_dir: Union[str, Path],
    data_result: Optional[Any] = None,
    model: Optional[Any] = None,
    run_name: Optional[str] = None,
    include_loss_plot: bool = True,
    format: str = "markdown",
) -> Path:
    """
    Generate a professional training report and save to disk.

    The report is model-independent and works with any TrainingResult produced by
    Trainer.fit(). Optionally enriches with data summary and model info when provided.

    :param result: TrainingResult from Trainer.fit().
    :param config: PipelineConfig used for the run.
    :param save_dir: directory to save the report and any figures.
    :param data_result: optional DataBuildResult with metadata (train/val/test sizes).
    :param model: optional Keras model for additional summary (params, layer count).
    :param run_name: optional run identifier for the header.
    :param include_loss_plot: whether to generate and embed the loss curve figure.
    :param format: output format — "markdown" (default) or "json".
    :return: path to the saved report file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_name or config.metadata.get("run_name", "train")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dc = config.data_config
    batch_size = (dc.get("batch_size") if isinstance(dc, dict) else
                  getattr(dc, "batch_size", None) if dc else None)

    # --- Build data summary (model-independent) ---
    data_summary = _extract_data_summary(data_result, batch_size=batch_size)

    # --- Build model summary ---
    model_summary = dict(result.model_summary or {})
    if model is not None:
        try:
            import tensorflow as tf

            def _param_count(weights):
                total = 0
                for w in weights:
                    try:
                        total += int(tf.reduce_prod(w.shape))
                    except (TypeError, ValueError):
                        pass
                return total

            model_summary["name"] = getattr(model, "name", "model")
            if hasattr(model, "trainable_weights"):
                model_summary["trainable_params"] = _param_count(model.trainable_weights)
            if hasattr(model, "non_trainable_weights"):
                model_summary["non_trainable_params"] = _param_count(model.non_trainable_weights)
            if hasattr(model, "layers"):
                model_summary["layers"] = len(model.layers)
        except Exception as e:
            logger.debug("Could not enrich model_summary from model: %s", e)

    # --- Keras model.summary() ---
    keras_summary_str: Optional[str] = None
    if model is not None and hasattr(model, "summary"):
        try:
            buffer = StringIO()
            model.summary(print_fn=lambda x: buffer.write(x + "\n"))
            keras_summary_str = buffer.getvalue().strip()
        except Exception as e:
            logger.debug("Could not capture Keras model.summary(): %s", e)

    # --- Multi-panel training plots ---
    plot_path = None
    if include_loss_plot and result.history:
        plot_path = save_training_plots(result, save_dir)

    # --- Config hash for reproducibility ---
    config_json = json.dumps(_serialise_config(config), sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    # --- Generate report ---
    if format == "json":
        report_path = save_dir / "training_report.json"
        _write_json_report(
            result=result,
            data_summary=data_summary,
            model_summary=model_summary,
            keras_summary=keras_summary_str,
            config_hash=config_hash,
            run_name=run_name,
            timestamp=timestamp,
            plot_path=str(plot_path) if plot_path else None,
            report_path=report_path,
        )
    else:
        report_path = save_dir / "training_report.md"
        _write_markdown_report(
            result=result,
            data_summary=data_summary,
            model_summary=model_summary,
            keras_summary=keras_summary_str,
            config_hash=config_hash,
            run_name=run_name,
            timestamp=timestamp,
            plot_path=plot_path,
            report_path=report_path,
        )

    logger.info(f"Training report saved to {report_path}")
    return report_path


def _extract_data_summary(
    data_result: Optional[Any],
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract data split sizes from DataBuildResult metadata."""
    summary: Dict[str, Any] = {}
    if data_result is None:
        return summary

    metadata = getattr(data_result, "metadata", None)
    if not isinstance(metadata, dict):
        return summary

    for key, count_key in [
        ("train_ends", "train_samples"),
        ("val_ends", "val_samples"),
        ("test_ends", "test_samples"),
    ]:
        if key in metadata:
            arr = metadata[key]
            summary[count_key] = len(arr) if hasattr(arr, "__len__") else None

    if batch_size is not None:
        if summary.get("train_samples") is not None:
            summary["train_batches"] = int(np.ceil(summary["train_samples"] / batch_size))
        if summary.get("val_samples") is not None:
            summary["val_batches"] = int(np.ceil(summary["val_samples"] / batch_size))
        summary["batch_size"] = batch_size

    return summary


def _serialise_config(config: PipelineConfig) -> Dict[str, Any]:
    """Convert PipelineConfig to a JSON-serialisable dict."""
    return {
        "training_config": config.training_config,
        "data_config": config.data_config,
        "model_config": config.model_config,
        "registry_dir": config.registry_dir,
        "tracking_dir": config.tracking_dir,
        "artifacts_dir": config.artifacts_dir,
        "metadata": config.metadata,
    }


def _write_markdown_report(
    result: TrainingResult,
    data_summary: Dict[str, Any],
    model_summary: Dict[str, Any],
    keras_summary: Optional[str],
    config_hash: str,
    run_name: str,
    timestamp: str,
    plot_path: Optional[Path],
    report_path: Path,
) -> None:
    """Write Markdown training report."""
    lines = [
        "# Training Report",
        "",
        f"**Run:** {run_name}  ",
        f"**Generated:** {timestamp}  ",
        f"**Config hash:** `{config_hash}`",
        "",
        "---",
        "",
        "## Executive summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Best epoch | {result.best_epoch} |",
        f"| Final epoch | {result.final_epoch} |",
        f"| Best train loss | {result.best_train_loss:.6f} |",
        f"| Best val loss | {result.best_val_loss or 'N/A'} |",
        f"| Training time | {result.training_time_seconds:.1f}s |",
        f"| Early stopped | {'Yes' if result.stopped_early else 'No'} |",
        "",
        "---",
        "",
        "## Data summary",
        "",
    ]

    if data_summary:
        for k, v in sorted(data_summary.items()):
            lines.append(f"- **{k}:** {v}")
        lines.append("")
    else:
        lines.append("*Data summary not available.*  ")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Model summary",
        "",
    ])

    if model_summary:
        for k, v in sorted(model_summary.items()):
            lines.append(f"- **{k}:** {v}")
        lines.append("")
    else:
        lines.append("*Model summary not available.*  ")
        lines.append("")

    if keras_summary:
        lines.extend([
            "### Model architecture (Keras summary)",
            "",
            "```text",
            keras_summary,
            "```",
            "",
        ])

    if plot_path and plot_path.exists():
        rel_path = plot_path.name
        lines.extend([
            "---",
            "",
            "## Training dynamics",
            "",
            f"![Training dynamics]({rel_path})",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Training configuration",
        "",
        "```json",
        json.dumps(result.config or {}, indent=2, default=str),
        "```",
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        f"Config hash (SHA256 prefix): `{config_hash}`  ",
        "Use this to verify identical configuration across runs.",
        "",
    ])

    if result.checkpoints:
        lines.extend([
            "### Checkpoints",
            "",
        ])
        for cp in result.checkpoints:
            lines.append(f"- Epoch {cp.epoch}: `{cp.path}` (val_loss={cp.val_loss})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_json_report(
    result: TrainingResult,
    data_summary: Dict[str, Any],
    model_summary: Dict[str, Any],
    keras_summary: Optional[str],
    config_hash: str,
    run_name: str,
    timestamp: str,
    plot_path: Optional[str],
    report_path: Path,
) -> None:
    """Write JSON-structured training report."""
    doc = {
        "run_name": run_name,
        "timestamp": timestamp,
        "config_hash": config_hash,
        "keras_model_summary": keras_summary,
        "executive_summary": {
            "best_epoch": result.best_epoch,
            "final_epoch": result.final_epoch,
            "best_train_loss": result.best_train_loss,
            "best_val_loss": result.best_val_loss,
            "training_time_seconds": result.training_time_seconds,
            "stopped_early": result.stopped_early,
        },
        "data_summary": data_summary,
        "model_summary": model_summary,
        "training_config": result.config,
        "history": result.history,
        "checkpoints": [c.to_dict() if hasattr(c, "to_dict") else vars(c) for c in result.checkpoints],
        "loss_plot_path": plot_path,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=str)
