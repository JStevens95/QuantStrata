"""
Pipeline wiring for the Hybrid GNN-RNN model.
"""
from src.rade_ml.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
from src.rade_ml.pipelines.hybrid_gnn_rnn.eval import HybridGnnRnnEvalPipeline
from src.rade_ml.pipelines.hybrid_gnn_rnn.infer import HybridGnnRnnInferencePipeline

__all__ = [
    "HybridGnnRnnTrainPipeline",
    "HybridGnnRnnEvalPipeline",
    "HybridGnnRnnInferencePipeline",
]
