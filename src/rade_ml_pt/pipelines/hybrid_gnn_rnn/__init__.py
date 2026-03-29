"""rade_ml_pt.pipelines.hybrid_gnn_rnn -- Hybrid GNN-RNN pipeline implementations."""
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval import HybridGnnRnnEvalPipeline
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.tune import HybridGnnRnnTunePipeline


def __getattr__(name: str):
    if name == "HybridGnnRnnInferencePipeline":
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import HybridGnnRnnInferencePipeline
        return HybridGnnRnnInferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HybridGnnRnnTrainPipeline",
    "HybridGnnRnnEvalPipeline",
    "HybridGnnRnnInferencePipeline",
    "HybridGnnRnnTunePipeline",
]
