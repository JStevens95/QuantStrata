#!/usr/bin/env python3
"""
Visualize Interpretable Weights from a Trained Hybrid GNN-RNN Model
===================================================================

Extracts and plots key learned weights in a professional, interpretable format:

  1. **Projection baseline kernels** – per-target linear weights over attention
     dimensions (how much each attention feature contributes to each target).
  2. **Projection baseline biases** – per-target additive offset.
  3. **Fusion gate weights** – relative importance of GNN vs RNN pathways (gate
     input = [fusion_features || rnn_features]; we decompose the gate weights).
  4. **GNN input projection** – first-layer weights from trade features to GNN
     hidden (optional; shows which input dimensions the GNN prioritizes).

Usage:
  # After running 01 or 02, visualize the latest registered model:
  python examples/rade_ml/hybrid_gnn_rnn/03_visualize_model_weights.py \\
    --registry_dir /path/to/artifacts/registry \\
    --output weights_report.png

  # Or load from a direct path:
  python examples/rade_ml/hybrid_gnn_rnn/03_visualize_model_weights.py \\
    --model_path /path/to/model.keras \\
    --output weights_report.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_model(registry_dir: Path | None = None, model_path: Path | None = None, version: str = "latest"):
    """Load HybridGnnRnn from registry or file. Registers custom layers for Keras."""
    import tensorflow as tf
    import src.rade_ml.models.hybrid_gnn_rnn.model  # noqa: F401 – registers custom classes

    if model_path is not None:
        return tf.keras.models.load_model(str(model_path))
    if registry_dir is not None:
        from src.rade_ml.registry.store import ModelRegistry
        registry = ModelRegistry(str(registry_dir))
        model, _ = registry.load(version_or_tag=version)
        return model
    raise ValueError("Provide either registry_dir or model_path")


def extract_weights(model) -> dict:
    """Extract interpretable weights from a HybridGnnRnn model."""
    out = {}

    proj = getattr(model, "projection_layer", None)
    if proj is not None and hasattr(proj, "_baseline_kernels"):
        k = proj._baseline_kernels
        b = proj._baseline_biases
        out["baseline_kernels"] = k.numpy() if hasattr(k, "numpy") else np.array(k)
        out["baseline_biases"] = b.numpy() if hasattr(b, "numpy") else np.array(b)

    fusion = getattr(model, "fusion_layer", None)
    if fusion is not None and hasattr(fusion, "gate_dense") and fusion.gate_dense is not None:
        w = fusion.gate_dense.kernel
        w_np = w.numpy() if hasattr(w, "numpy") else np.array(w)
        fusion_dim = w_np.shape[0] // 2
        out["gate_fusion_weights"] = w_np[:fusion_dim, 0]
        out["gate_rnn_weights"] = w_np[fusion_dim:, 0]
        out["gate_bias"] = (
            fusion.gate_dense.bias.numpy()
            if fusion.gate_dense.bias is not None and hasattr(fusion.gate_dense.bias, "numpy")
            else 0.0
        )

    gnn = getattr(model, "gnn_block", None)
    if gnn is not None and hasattr(gnn, "input_projection") and gnn.input_projection is not None:
        w = gnn.input_projection.kernel
        out["gnn_input_projection"] = w.numpy() if hasattr(w, "numpy") else np.array(w)
    elif gnn is not None and hasattr(gnn, "gnn_layers") and len(gnn.gnn_layers) > 0:
        first = gnn.gnn_layers[0]
        if hasattr(first, "dense_self") and first.dense_self is not None:
            w = first.dense_self.kernel
            out["gnn_first_layer"] = w.numpy() if hasattr(w, "numpy") else np.array(w)

    return out


def create_weight_figures(weights: dict, target_names: list[str] | None = None) -> "matplotlib.figure.Figure":
    """Create a multi-panel figure for weight visualization."""
    import matplotlib.pyplot as plt

    n_plots = 0
    if "baseline_kernels" in weights:
        n_plots += 2
    if "gate_fusion_weights" in weights:
        n_plots += 1
    if "gnn_input_projection" in weights or "gnn_first_layer" in weights:
        n_plots += 1

    n_plots = max(1, n_plots)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    idx = 0

    tgt_names = target_names or [f"Target {i+1}" for i in range(weights.get("baseline_kernels", np.zeros((1, 1))).shape[0])]

    if "baseline_kernels" in weights:
        K = weights["baseline_kernels"]
        ax = axes_flat[idx]
        im = ax.imshow(K, aspect="auto", cmap="RdBu_r", vmin=-np.abs(K).max(), vmax=np.abs(K).max())
        ax.set_xlabel("Attention dimension")
        ax.set_ylabel("Target trade")
        ax.set_yticks(range(K.shape[0]))
        ax.set_yticklabels(tgt_names[: K.shape[0]], fontsize=9)
        ax.set_title("Projection baseline kernels\n(attn → target)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        idx += 1

    if "baseline_biases" in weights:
        b = weights["baseline_biases"]
        ax = axes_flat[idx]
        ax.bar(range(len(b)), b, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Target trade")
        ax.set_ylabel("Bias")
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels(tgt_names[: len(b)], fontsize=9)
        ax.set_title("Projection baseline biases")
        idx += 1

    if "gate_fusion_weights" in weights:
        ax = axes_flat[idx]
        w_f = np.abs(weights["gate_fusion_weights"])
        w_r = np.abs(weights["gate_rnn_weights"])
        x = np.arange(len(w_f))
        width = 0.35
        ax.bar(x - width / 2, w_f, width, label="GNN (fusion)", color="coral", alpha=0.9)
        ax.bar(x + width / 2, w_r, width, label="RNN", color="seagreen", alpha=0.9)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("|Weight|")
        ax.set_title("Fusion gate input importance\n(GNN vs RNN pathway)")
        ax.legend(loc="upper right", fontsize=8)
        idx += 1

    if "gnn_input_projection" in weights or "gnn_first_layer" in weights:
        ax = axes_flat[idx]
        W = weights.get("gnn_input_projection", weights.get("gnn_first_layer"))
        W_imp = np.abs(W).mean(axis=1)
        ax.barh(range(len(W_imp)), W_imp, color="mediumpurple", alpha=0.8)
        ax.set_ylabel("Input feature index")
        ax.set_xlabel("Mean |weight|")
        ax.set_title("GNN input projection\n(feature importance)")
        idx += 1

    for j in range(idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Hybrid GNN-RNN learned weights", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize Hybrid GNN-RNN model weights")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--registry_dir", type=Path, help="Path to model registry (e.g. artifacts/registry)")
    g.add_argument("--model_path", type=Path, help="Path to model.keras file")
    parser.add_argument("--version", default="latest", help="Registry version/tag (default: latest)")
    parser.add_argument("--output", type=Path, default=None, help="Save figure to path (default: display)")
    parser.add_argument("--target_names", nargs="+", default=None, help="Target trade names for axis labels")
    args = parser.parse_args()

    model = load_model(
        registry_dir=args.registry_dir if args.registry_dir else None,
        model_path=args.model_path,
        version=args.version,
    )

    weights = extract_weights(model)
    if not weights:
        print("No interpretable weights found. Is this a HybridGnnRnn model?")
        return 1

    fig = create_weight_figures(weights, target_names=args.target_names)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
