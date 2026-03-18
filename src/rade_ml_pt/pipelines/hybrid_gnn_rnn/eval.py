"""
Evaluation pipeline for the Hybrid GNN-RNN model (PyTorch).

Loads a registered model, builds test data, and runs evaluation through the
generic Evaluator. The post_eval hook adds GNN-RNN-specific diagnostics
and persists evaluation analytics for the UI dashboard.

When datasets were saved during training (registry/{version}/datasets/),
build_data() loads them directly instead of re-running the full data pipeline.
This enables true cold-start evaluation from the registry alone.

In addition to test-set evaluation, this pipeline collects predicted vs actual
PnL for all available splits (train, val, test) so the UI can show how model
output compares to actual PnL across the full timeline.
"""
from __future__ import annotations

import os
import json
import torch
import logging

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING, Optional, Tuple, List

from src.rade_ml_pt.evaluation.plots import save_evaluation_plots
from src.rade_ml_pt.pipelines.base import EvalPipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig

from src.rade_ml_pt.data.dataset import _collate_dict_batch
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import build_dataset, HybridGnnRnnResult
from src.rade_ml_pt.data.hybrid_gnn_rnn.plots import plot_portfolio_pnl
from tests.rade_ml_pt.pipelines.ensemble.conftest import cluster_mapping

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import EvaluationResult
    from src.rade_ml_pt.data.result import DataBuildResult

# define module level logging.
logger = logging.getLogger(__name__)


class HybridGnnRnnEvalPipeline(EvalPipeline):
    """
    Concrete evaluation pipeline for Hybrid GNN-RNN.

    Overrides the base ``run()`` to collect predictions on all splits
    (train, val, test) -- not just test -- so the UI can show model PnL
    vs actual PnL across the full training/evaluation timeline.

    If cached datasets exist in the registry version directory, they are loaded
    directly -- no data pipeline is re-run.
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        cached = self._load_cached_data()
        if cached is not None:
            logger.info("EvalPipeline: loaded cached datasets from registry (skipped data build)")
            return cached

        logger.info("EvalPipeline: no cached datasets found, running full data build")
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        job = config.metadata.get("job", {})
        return build_dataset(config=data_config, job=job)

    def run(self) -> Dict[str, "EvaluationResult"]:
        """
        Execute evaluation on the test set, then collect predictions on
        all available splits (train, val, test) for PnL comparison analytics.

        The per-split predictions are saved alongside the standard evaluation
        artifacts so the UI can render predicted vs actual PnL timelines
        across the full data period.
        """
        from src.rade_ml_pt.evaluation.evaluator import Evaluator
        logger.info("EvalPipeline: starting")

        # 1. load trained model.
        model, entry = self.load_model(self.config)
        self._loaded_entry = entry
        logger.info(f"EvalPipeline: loaded model version '{entry.version}'")

        # 2. build data
        data_result = self.build_data(self.config)
        logger.info("EvalPipeline: data built")

        # 3. load additional data to help with analytics.
        add_data = self.load_additional_data(data_result)

        # 4. initiate model evaluator.
        evaluator = Evaluator(model=model)
        output = {}

        # 5.1. evaluate training period data + post eval hook.
        train_eval_result = evaluator.run(data_result.train_ds)
        post_eval_train = self.post_eval(train_eval_result, self.config, data_result, **add_data)
        self.post_eval_plots(post_eval_train, data_result, "train")

        # update output with training results.
        output.update({"training_results": train_eval_result})
        print(" ------ Training Dataset ------")
        print(train_eval_result.summary())
        logger.info("EvalPipeline: training data evaluation complete.")

        # 5.2. evaluate validation period data + post eval hook (if availabe).
        if data_result.data_config["validation_split"] != 0.0:
            val_eval_result = evaluator.run(data_result.val_ds)
            post_eval_eval = self.post_eval(val_eval_result, self.config, data_result, **add_data)
            self.post_eval_plots(post_eval_eval, data_result, "val")

            # update output with validation results.
            output.update({"validation_results": val_eval_result})
            print(" ------ Validation Dataset ------")
            print(val_eval_result.summary())
            logger.info("EvalPipeline: validation dataset evaluation complete.")

        # 5.3. evaluate testing period data + post eval hook (if available).
        if data_result.data_config["test_split"] != 0.0:
            test_eval_result = evaluator.run(data_result.test_ds)
            post_eval_eval = self.post_eval(test_eval_result, self.config, data_result, **add_data)
            self.post_eval_plots(post_eval_eval, data_result, "test")

            # update output with testing result.
            output.update({"test_results": test_eval_result})
            print(" ------ Test Dataset ------")
            print(test_eval_result.summary())
            logger.error("EvalPipeline: test dataset evaluation complete.")

        logger.info("EvalPipeline: done")
        return output

    def post_eval(
            self, eval_result: "EvaluationResult", config: "PipelineConfig",
            data_result: Optional["DataBuildResult"] = None, **kwargs
    ) -> "EvaluationResult":
        """Post evaluation analytics for HybridGnnRnn model."""
        # default: register model and log to tracker.
        super().post_eval(eval_result, config, data_result)

        # 0. validate evaluation result.
        self._validate_inputs(eval_result, data_result, **kwargs)

        # extract predictions, targets and transformer from ``eval_result``
        predictions = eval_result.predictions
        targets = eval_result.targets
        transformer = self.get_target_scaler(data_result)

        # 1. invert standardisation of targets.
        predictions_unsclaed, targets_unsclaed = self._inverse_target_transforms(predictions, targets, transformer)

        # 2. aggregate predictions / target pnl across trades.
        portfolio_pnl = self._aggregate_trade_pnl(predictions, targets)
        portfolio_pnl_unscaled = self._aggregate_trade_pnl(predictions_unsclaed, targets_unsclaed)

        # 3. re-build full target trade portfolio from linear independent basis model was calibrated on.
        # TODO: implement scaling to full tagrte portfolio + original notional.

        # amend extra information / data to EvaluationResult.
        eval_result.metadata.update({
            "predictions_unscaled": predictions_unsclaed, "targets_unsclaed": targets_unsclaed,
            "portfolio_pnl_unscaled": portfolio_pnl_unscaled, "portfolio_pnl": portfolio_pnl,
        })
        return eval_result

    def post_eval_plots(
            self, result: "EvaluationResult", data_result: "DataBuildResult", period: str = "training"
    ) -> None:
        """Run GNN-RNN specific plots for evaluation."""
        # define evaluation artifacts path.
        save_path = Path(self.config.artifacts_dir, "evaluation", self._loaded_entry.version, period)
        save_path.mkdir(parents=True, exist_ok=True)

        # 0. run standard evaulation plots.
        save_evaluation_plots(result, save_path, data_result.metadata["target_ids"])

        # 1. plot portfolio period analytics -- scaled predictions/targets.
        plot_portfolio_pnl(
            pnl_df=result.metadata["portfolio_pnl"], save_path=save_path, period=period, scale_type="scaled"
        )
        # plot portfolio period analytics -- unsclaed predictions/targets
        plot_portfolio_pnl(
            pnl_df=result.metadata["portfolio_pnl_unscaled"], save_path=save_path, period=period, scale_type="unscaled"
        )

        # 2. plot full portfolio period analytics -- restore full population and revert to original notional
        # TODO: implement full portfolio period analytics

    def get_target_scaler(self, data_result: "DataBuildResult") -> Optional[Any]:
        """Return target pnl scaler for inverse transform to original unit space."""
        return data_result.metadata.get("target_pnl_transformer")

    @staticmethod
    def _aggregate_trade_pnl(
            predictions: np.ndarray, targets: np.ndarray
    ) -> pd.DataFrame:
        """Aggregate target portfolio across trades to give portfolio pnl for predictions and targets."""
        assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets pnl."

        # aggregate pnls across trades for each scenario.
        preds_agg, tgts_agg = predictions.sum(axis=1), targets.sum(axis=1)
        diff_agg = np.abs(tgts_agg - preds_agg)

        # create portfolio pnl dataframe.
        portfolio_pnl = pd.DataFrame({
            "scenario": np.arange(len(preds_agg)), "predicted_pnl": preds_agg, "target_pnl": tgts_agg,
            "abs_diff": diff_agg
        })
        return portfolio_pnl

    @staticmethod
    def _inverse_target_transforms(
            predictions: np.ndarray, targets: np.ndarray, transformer: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Invert standardisation of target trades."""
        # 0. reversion of scale transformation (z-space -> p-space).
        if transformer:
            predictions_unscaled = transformer.inverse_transform(predictions)
            targets_unsclaed = transformer.inverse_transform(targets)
        else:
            predictions_unscaled = predictions
            targets_unsclaed = targets
        return predictions_unscaled, targets_unsclaed

    @staticmethod
    def _inverse_target_notional(
            target_pnl: pd.DataFrame, target_attributes: Dict[str, Any],
            req_cols: List[str] = ["TradeKey", "NotionalSign"]
    ) -> pd.DataFrame:
        """Invert target trade notional scaling."""
        pass

    @staticmethod
    def _rebuild_target_portfolio(
            predictions: np.ndarray, m1: pd.DataFrame, m2: pd.DataFrame,
    ) -> pd.DataFrame:
        """Reconstruct the original target trade portfolio from linearly independent basis (M! and M2)."""
        inter_df = np.matmul(predictions, m1.T)
        inter_df.columns = m1.index
        inter_df.index = m2.index
        reconstructed_df = inter_df + m2
        return reconstructed_df

    def _load_cached_data(self) -> Optional[HybridGnnRnnResult]:
        """
        Attempt to load datasets and artifacts from the registry version directory.

        Returns a fully populated HybridGnnRnnResult if cached datasets exists otherwise None (triggering fallback
        to full data build)
        """
        import joblib
        from torch.utils.data import DataLoader

        # check if entry is None.
        if self._loaded_entry is None:
            return None

        # create version directory for artifact loading.
        version_dir = Path(self._loaded_entry.model_dir)
        ds_dir = Path(version_dir, "datasets")

        # build metadata component - pnl transformers.
        metadata = {}
        if (version_dir / "target_scaler.pkl").exists():
            metadata["target_pnl_transformer"] = joblib.load(version_dir / "target_scaler.pkl")
            logger.info(f"Loaded target_scaler.pkl from {version_dir}")
        if (version_dir / "elementary_scaler.pkl").exists():
            metadata["elementary_pnl_transformer"] = joblib.load(version_dir / "elementary_scaler.pkl")
            logger.info(f"Loaded elementary_pnl_transformer.pkl from {version_dir}")

        # build metadata component - elementary & target indices.
        if os.path.exists(version_dir / "trade_universe.json"):
            with open(version_dir / "trade_universe.json") as f:
                metadata.update(json.load(f))
            logger.info(f"Loaded trade_universe.json from {version_dir}")

        # load elementary and target pnl.
        target_pnl, elementary_pnl = None, None
        if (version_dir / "target_pnl.parquet").exists():
            target_pnl = pd.read_parquet(version_dir / "target_pnl.parquet")
            logger.info(f"Loaded target_pnl.parquet from {version_dir}")
        if (version_dir / "elementary_pnl.parquet").exists():
            elementary_pnl = pd.read_parquet(version_dir / "elementary_pnl.parquet")
            logger.info(f"Loaded elementary_pnl.parquet from {version_dir}")

        # load elementary and target attributes.
        target_attributes, elementary_attributes = None, None
        if os.path.exists(version_dir / "target_attributes.json"):
            with open(version_dir / "target_attributes.json") as f:
                target_attributes = json.load(f)
            logger.info(f"Loaded target_attributes.json from {version_dir}")
        if os.path.exists(version_dir / "elementary_attributes.json"):
            with open(version_dir / "elementary_attributes.json") as f:
                elementary_attributes = json.load(f)
            logger.info(f"Loaded elementary_attributes.json from {version_dir}")

        # load data configuration.
        data_config = None
        if os.path.exists(version_dir / "data_config.json"):
            with open(version_dir / "data_config.json") as f:
                data_config = json.load(f)

        # load cluster information.
        cluster_info = None
        if os.path.exists(version_dir / "cluster_info.json"):
            cluster_info = joblib.load(version_dir / "cluster_info.json")

        # reconstruct DataLoaders from saved datasets - train.
        train_ds = None
        if (ds_dir / "train.pt").exists():
            train_dataset = torch.load(str(ds_dir / "train.pt"), weights_only=False)
            train_ds = DataLoader(train_dataset, batch_size=data_config["batch_size"], collate_fn=_collate_dict_batch)

        # reconstruct DataLoaders from saved datasets - val.
        val_ds = None
        if (ds_dir / "val.pt").exists():
            val_dataset = torch.load(str(ds_dir / "val.pt"), weights_only=False)
            val_ds = DataLoader(val_dataset, batch_size=data_config["batch_size"], collate_fn=_collate_dict_batch)

        # reconstruct DataLoaders from saved datasets - test
        test_ds = None
        if (ds_dir / "test.pt").exists():
            test_dataset = torch.load(str(ds_dir / "test.pt"), weights_only=False)
            test_ds = DataLoader(test_dataset, batch_size=data_config["batch_size"], collate_fn=_collate_dict_batch)

        return HybridGnnRnnResult(
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, data_config=data_config, cluster_info=cluster_info,
            metadata=metadata, target_pnl=target_pnl, elementary_pnl=elementary_pnl, target_attributes=target_attributes,
            elementary_attributes=elementary_attributes
        )

    @staticmethod
    def load_additional_data(config: "DataBuildResult") -> Dict[str, Any]:
        """Load additional input data required for some evaluation analytics."""
        # extract cluster path for additional data loading.
        cluster_path = config.cluster_info["cluster_path"]

        # load m1 and m2 files.
        target_m1 = pd.read_csv(os.path.join(cluster_path, "target_m1.csv"), index_col=0)
        target_m2 = pd.read_csv(os.path.join(cluster_path, "target_m2.csv"), index_col=0)

        # load full target trade portfolio pnl time-series.
        target_pnl_full = pd.read_csv(os.path.join(cluster_path, "target_pnl_notional.csv"), index_col=0)
        return {
            "target_m1": target_m1, "target_m2": target_m2, "target_pnl_full": target_pnl_full,
        }

    @staticmethod
    def _validate_inputs(eval_result: "EvaluationResult", data_result: "DataBuildResult", **kwargs) -> None:
        """Validate input data and evalidation results."""

        # convert predictions and targets to numpy and validate shapes.
        preds, tgts = eval_result.predictions, eval_result.targets
        if preds.shape != tgts.shape:
            raise AssertionError(f"Shape mismatch between predictions {preds.shape} != targets {tgts.shape}")

        # validate scenario indices.
        scenario_indices = np.asarray(data_result.metadata["scenario_idx"])
        if not np.issubdtype(scenario_indices.dtype, np.integer):
            raise ValueError(
                "scenario_indices, must be integer positional indices for iloc indexing of m2 / target_pnl_full"
            )

        # validate additional data contains target m1/m2 and full target trade portfolio.
        assert "target_m1" in kwargs, "Kwargs missing required target m1 data."
        assert "target_m2" in kwargs, "Kwargs missing required target m2 data."
        assert "target_pnl_full" in kwargs, "Kwargs missing required target pnl data."

        assert isinstance(kwargs["target_m1"], pd.DataFrame), "target_m1 should be pd.Dataframe"
        assert isinstance(kwargs["target_m2"], pd.DataFrame), "target_m2 should be pd.Dataframe"
        assert isinstance(kwargs["target_pnl_full"], pd.DataFrame), "target_pnl_full should be pd.DataFrame"
