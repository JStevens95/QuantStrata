"""
Ensemble model wrapping N trained members with trade routing and aggregation.

``EnsembleModel`` is **not** an ``nn.Module`` itself — it is a lightweight
orchestrator that holds references to N member modules, delegates each
trade to the correct member via ``TradeRouter``, and combines outputs
via the configured aggregation strategy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.rade_ml_pt.ensemble.router import TradeRouter
from src.rade_ml_pt.ensemble.aggregation import get_aggregation_fn

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Orchestrate N member ``nn.Module`` models under a single predict() API.

    Parameters
    ----------
    members : dict
        ``{cluster_id: nn.Module}`` — loaded and eval-ready member models.
    router : TradeRouter
        Trade-to-cluster routing logic.
    aggregation : str
        Name of the aggregation strategy (``"concat"`` or ``"weighted_mean"``).
    weights : dict or None
        ``{cluster_id: float}`` — member weights for weighted aggregation.
    cluster_trade_indices : dict or None
        ``{cluster_id: [global_col_indices]}`` — column positions of each
        cluster's targets in the full output array.  Required for ``"concat"``.
    n_total_targets : int or None
        Total number of target columns in the combined output.
        Required for ``"concat"``.
    """

    def __init__(
        self,
        members: Dict[str, nn.Module],
        router: TradeRouter,
        aggregation: str = "concat",
        weights: Optional[Dict[str, float]] = None,
        cluster_trade_indices: Optional[Dict[str, List[int]]] = None,
        n_total_targets: Optional[int] = None,
    ) -> None:
        self.members = members
        self.router = router
        self.aggregation = aggregation
        self.weights = weights or {}
        self.cluster_trade_indices = cluster_trade_indices or {}
        self.n_total_targets = n_total_targets
        self._aggregate_fn = get_aggregation_fn(aggregation)

        for cid, model in members.items():
            model.eval()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        member_inputs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Run each member on its inputs and aggregate.

        Parameters
        ----------
        member_inputs : dict
            ``{cluster_id: model_input_dict}`` — pre-routed inputs per member.

        Returns
        -------
        np.ndarray
            Combined predictions, shape ``[n_scenarios, n_total_targets]``.
        """
        member_preds: Dict[str, np.ndarray] = {}

        for cid, inputs in member_inputs.items():
            if cid not in self.members:
                logger.warning("No member model for cluster '%s'; skipping.", cid)
                continue
            member_preds[cid] = self.predict_member(cid, inputs)

        if not member_preds:
            raise RuntimeError("No member produced predictions.")

        return self._combine(member_preds)

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        """Return the device the model parameters live on."""
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _to_device(obj: Any, device: torch.device) -> Any:
        """Recursively move tensors / dicts / lists to *device*."""
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            return torch.as_tensor(obj).to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: EnsembleModel._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(EnsembleModel._to_device(v, device) for v in obj)
        return obj

    def predict_member(
        self,
        cluster_id: str,
        inputs: Any,
    ) -> np.ndarray:
        """
        Run a single member's forward pass.

        Inputs are automatically moved to the member model's device before
        inference, so callers can pass CPU tensors or numpy arrays regardless
        of where the model lives.

        Parameters
        ----------
        cluster_id : str
        inputs : dict or tensor
            Model-ready inputs for this member.

        Returns
        -------
        np.ndarray
        """
        model = self.members[cluster_id]
        model.eval()
        device = self._model_device(model)

        prepared = self._to_device(inputs, device)

        with torch.no_grad():
            output = model(prepared)

        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        return np.asarray(output)

    # ------------------------------------------------------------------
    # Metadata for UI / analytics
    # ------------------------------------------------------------------

    def get_member_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Return per-member metadata for the UI dashboard.

        Includes trade count, parameter count, and model class name
        for each cluster.
        """
        meta: Dict[str, Dict[str, Any]] = {}
        for cid, model in self.members.items():
            n_params = sum(p.numel() for p in model.parameters())
            meta[cid] = {
                "model_class": type(model).__name__,
                "n_parameters": n_params,
                "n_trades": len(self.router.get_trades_for_cluster(cid)),
            }
        return meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _combine(self, member_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply the configured aggregation strategy."""
        if self.aggregation == "concat":
            return self._aggregate_fn(
                member_preds,
                cluster_trade_indices=self.cluster_trade_indices,
                n_total_targets=self.n_total_targets,
            )
        elif self.aggregation == "weighted_mean":
            return self._aggregate_fn(member_preds, weights=self.weights)
        else:
            return self._aggregate_fn(member_preds)
