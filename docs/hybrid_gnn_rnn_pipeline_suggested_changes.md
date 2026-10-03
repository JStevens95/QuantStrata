# Suggested Changes for Hybrid GNN-RNN Pipelines

Each item below states the **reason** for the change and the **suggested fix**. Apply these in train, eval, and infer as needed.

---

## 1. train.py

### 1.1 _post_train_plots: data_config dict access (lines 262, 270)

**Reason:** `self.config.data_config` may be a `HybridGnnRnnDataConfig` dataclass (e.g. when the pipeline is called with `PipelineConfig(data_config=HybridGnnRnnDataConfig(...))`). Dataclasses do not support `["key"]` subscripting; that raises `TypeError: 'HybridGnnRnnDataConfig' object is not subscriptable`. Using `getattr` allows both dict and dataclass usage.

**Suggested change:**

```python
# Before:
if self.config.data_config["plot_pnl_distribution"] and self.config.artifacts_dir:
    ...
if self.config.data_config["plot_trade_graph"]:

# After:
dc = self.config.data_config
plot_pnl = getattr(dc, "plot_pnl_distribution", False) if hasattr(dc, "plot_pnl_distribution") else (dc.get("plot_pnl_distribution", False) if isinstance(dc, dict) else False)
if plot_pnl and self.config.artifacts_dir:
    plot_pnl_distribution(...)

plot_graph = getattr(dc, "plot_trade_graph", False) if hasattr(dc, "plot_trade_graph") else (dc.get("plot_trade_graph", False) if isinstance(dc, dict) else False)
if plot_graph:
    plot_trade_graph(...)
```

Or more simply, if you always use a dataclass from `build_data` and only ever pass dict from external callers, you can use:

```python
dc = self.config.data_config
plot_pnl = getattr(dc, "plot_pnl_distribution", False) if not isinstance(dc, dict) else dc.get("plot_pnl_distribution", False)
plot_graph = getattr(dc, "plot_trade_graph", False) if not isinstance(dc, dict) else dc.get("plot_trade_graph", False)
if plot_pnl and self.config.artifacts_dir:
    ...
if plot_graph:
    ...
```

---

### 1.2 trade_universe: sequence_length default (line 163)

**Reason:** `data_result.metadata.get("sequence_length", int)` uses the type `int` as the default value. When this is written to JSON, it is not serialisable (or becomes a string like `"<class 'int'>"`). The default should be a concrete integer so that cached eval loading and any consumer of `trade_universe.json` get a valid number.

**Suggested change:**

```python
# Before:
"sequence_length": data_result.metadata.get("sequence_length", int),

# After:
"sequence_length": data_result.metadata.get("sequence_length", 1),
```

---

## 2. eval.py

### 2.1 Remove production import from tests (line 38)

**Reason:** Importing `cluster_mapping` from `tests.rade_ml_pt.pipelines.ensemble.conftest` ties production code to the test package. The symbol is not used anywhere in eval.py. In packaged or deployed environments the `tests` package may be excluded, causing `ImportError`, or tests may run in an order that affects production behaviour. Production modules must not depend on test code.

**Suggested change:**

```python
# Before:
from tests.rade_ml_pt.pipelines.ensemble.conftest import cluster_mapping

# After:
# (delete this line entirely)
```

---

### 2.2 cluster_info loading: filename and existence check (lines 312–314)

**Reason:** Train saves job entries as `{key}.joblib` (e.g. `cluster_info.joblib`), not `cluster_info.json`. So `version_dir / "cluster_info.json"` never exists after a normal train run, and `cluster_info` is always `None`. Eval then fails in `load_additional_data` when it does `config.cluster_info["cluster_path"]` (AttributeError or KeyError). Loading from the same filename train uses (and guarding when the file or key is missing) keeps eval consistent with train and avoids crashes when cluster data is absent.

**Suggested change:**

```python
# Before:
cluster_info = None
if os.path.exists(version_dir / "cluster_info.json"):
    cluster_info = joblib.load(version_dir / "cluster_info.json")

# After:
cluster_info = None
cluster_info_path = version_dir / "cluster_info.joblib"
if cluster_info_path.exists():
    cluster_info = joblib.load(cluster_info_path)
    logger.info(f"Loaded cluster_info from {version_dir}")
```

---

### 2.3 load_additional_data: guard when cluster_info or path is missing (lines 339–352)

**Reason:** When running in the personal env or when a run did not save `cluster_info` (e.g. job had no such key), `data_result.cluster_info` is None and `config.cluster_info["cluster_path"]` raises. The same applies if the path or CSVs are missing. Returning an empty dict and skipping the extra analytics (or making post_eval tolerate missing keys) allows eval to complete and only run full portfolio analytics when the work-env files exist.

**Suggested change:**

```python
# Before:
@staticmethod
def load_additional_data(config: "DataBuildResult") -> Dict[str, Any]:
    """Load additional input data required for some evaluation analytics."""
    cluster_path = config.cluster_info["cluster_path"]
    target_m1 = pd.read_csv(os.path.join(cluster_path, "target_m1.csv"), index_col=0)
    ...

# After:
@staticmethod
def load_additional_data(config: "DataBuildResult") -> Dict[str, Any]:
    """Load additional input data required for some evaluation analytics.
    Returns empty dict if cluster_info or cluster_path or required CSVs are missing.
    """
    if config.cluster_info is None:
        logger.warning("cluster_info not in registry; skipping additional analytics (m1/m2/full PnL).")
        return {}
    cluster_path = config.cluster_info.get("cluster_path")
    if not cluster_path:
        logger.warning("cluster_info has no 'cluster_path'; skipping additional analytics.")
        return {}
    required = {"target_m1.csv", "target_m2.csv", "target_pnl_notional.csv"}
    base = os.path.join(cluster_path, "")
    if not all(os.path.exists(base + f) for f in required):
        logger.warning("One or more of target_m1.csv, target_m2.csv, target_pnl_notional.csv missing; skipping additional analytics.")
        return {}
    target_m1 = pd.read_csv(os.path.join(cluster_path, "target_m1.csv"), index_col=0)
    target_m2 = pd.read_csv(os.path.join(cluster_path, "target_m2.csv"), index_col=0)
    target_pnl_full = pd.read_csv(os.path.join(cluster_path, "target_pnl_notional.csv"), index_col=0)
    return {"target_m1": target_m1, "target_m2": target_m2, "target_pnl_full": target_pnl_full}
```

Then in `run()`, only call `post_eval(..., **add_data)` and use `add_data` in `_validate_inputs` only when the extra data is present (see 2.5).

---

### 2.4 Log level for test completion (line 136)

**Reason:** Completing the test evaluation is normal flow, not an error. Using `logger.error` mislabels it and can trigger alerting or log filters. Use `logger.info` for normal completion.

**Suggested change:**

```python
# Before:
logger.error("EvalPipeline: test dataset evaluation complete.")

# After:
logger.info("EvalPipeline: test dataset evaluation complete.")
```

---

### 2.5 _validate_inputs: allow missing additional data when not loaded (lines 356–374)

**Reason:** When `load_additional_data` returns `{}` (e.g. personal env or missing cluster path), `post_eval` still passes `**add_data` and `_validate_inputs` asserts that `target_m1`, `target_m2`, `target_pnl_full` are in kwargs. Those asserts then fail. Validation should require the extra data only when it was actually requested and loaded (e.g. when cluster_info and path and files exist). When `add_data` is empty, skip the additional-data checks.

**Suggested change:**

```python
# Before:
assert "target_m1" in kwargs, "Kwargs missing required target m1 data."
assert "target_m2" in kwargs, ...
assert "target_pnl_full" in kwargs, ...

# After:
if kwargs:
    assert "target_m1" in kwargs, "Kwargs missing required target m1 data."
    assert "target_m2" in kwargs, "Kwargs missing required target m2 data."
    assert "target_pnl_full" in kwargs, "Kwargs missing required target pnl data."
    assert isinstance(kwargs["target_m1"], pd.DataFrame), "target_m1 should be pd.DataFrame"
    assert isinstance(kwargs["target_m2"], pd.DataFrame), "target_m2 should be pd.DataFrame"
    assert isinstance(kwargs["target_pnl_full"], pd.DataFrame), "target_pnl_full should be pd.DataFrame"
# When kwargs is empty (load_additional_data returned {}), skip these checks.
```

---

### 2.6 data_config access for validation_split and test_split (lines 116, 128)

**Reason:** When data comes from `build_dataset`, `data_result.data_config` is a `HybridGnnRnnDataConfig` dataclass; subscripting it with `["validation_split"]` or `["test_split"]` raises TypeError. When data comes from cache, `data_config` is a dict from JSON. Supporting both avoids crashes when eval is run immediately after train without going through cache.

**Suggested change:**

```python
# Before:
if data_result.data_config["validation_split"] != 0.0:
...
if data_result.data_config["test_split"] != 0.0:

# After:
def _get_data_config_value(dc, key: str, default=None):
    if dc is None:
        return default
    if isinstance(dc, dict):
        return dc.get(key, default)
    return getattr(dc, key, default)

if _get_data_config_value(data_result.data_config, "validation_split", 0.0) != 0.0:
    ...
if _get_data_config_value(data_result.data_config, "test_split", 0.0) != 0.0:
    ...
```

You can add `_get_data_config_value` as a static method on the pipeline class or a module-level helper.

---

### 2.7 _load_cached_data: DataLoader batch_size when data_config is None (lines 318–330)

**Reason:** If `data_config.json` is missing (e.g. old run or partial registry), `data_config` stays None and `data_config["batch_size"]` in the DataLoader construction raises. Using a safe default allows cached datasets to still be loaded when data_config is missing.

**Suggested change:**

```python
# Before:
train_ds = DataLoader(train_dataset, batch_size=data_config["batch_size"], collate_fn=_collate_dict_batch)

# After:
batch_size = (data_config or {}).get("batch_size", 32) if isinstance(data_config, dict) else getattr(data_config, "batch_size", 32) if data_config is not None else 32
train_ds = DataLoader(train_dataset, batch_size=batch_size, collate_fn=_collate_dict_batch)
```

Repeat for `val_ds` and `test_ds` (reuse the same `batch_size` variable).

---

## 3. infer.py

### 3.1 load_inference_context: fix file-loading logic (lines 221–242)

**Reason:** The condition `if p.exists() and str(p).endswith(".joblib") or str(p).endswith(".pkl")` is parsed as `(p.exists() and ...) or str(p).endswith(".pkl")`. For a path like `data_config.json`, the second part is False, but for any `.pkl` path the `.pkl` part is True, so you can run `joblib.load` on a non-joblib file. Also, each of the three blocks (joblib, json, parquet) can overwrite the same key if multiple conditions are true for one path. Loading should branch on the actual file extension and run exactly one loader per path.

**Suggested change:**

```python
# Before:
for name, fname in [...]:
    p = version_dir / fname
    if p.exists() and str(p).endswith(".joblib") or str(p).endswith(".pkl"):
        context[name] = joblib.load(str(p))
    if p.exists() and str(p).endswith(".json"):
        with open(p) as f:
            context[name] = json.load(f)
    if p.exists() and str(p).endswith(".parquet"):
        context[name] = pd.read_parquet(path=str(p))

# After:
for name, fname in [
    ("data_config", "data_config.json"),
    ("graph_results", "graph_results.joblib"),
    ("encoder_results", "encoder_results.joblib"),
    ("elementary_pnl", "elementary_pnl.parquet"),
    ("elementary_attribs", "elementary_attributes.json"),
    ("elementary_scaler", "elementary_scaler.pkl"),
    ("target_attribs", "target_attributes.json"),
    ("trade_universe", "trade_universe.json"),
    ("cluster_info", "cluster_info.joblib"),
    ("cluster_assets", "cluster_assets.joblib"),
    ("cluster_elem_trades", "cluster_elem_trades.joblib"),
]:
    p = version_dir / fname
    if not p.exists():
        continue
    suf = p.suffix.lower()
    if suf == ".json":
        with open(p) as f:
            context[name] = json.load(f)
    elif suf in (".pkl", ".joblib"):
        context[name] = joblib.load(str(p))
    elif suf == ".parquet":
        context[name] = pd.read_parquet(path=str(p))
```

---

### 3.2 _prepare_new_scenarios_inputs: return real model inputs and support pre-built pnl_history (lines 277–324)

**Reason:** The method currently builds an `InferenceInputData` and fills `inputs.elementary_pnl` but never assembles the 7-key dict expected by `HybridGnnRnn.forward()`, and it returns `output = {}`. The runner therefore receives no `"inputs"` and inference fails. In addition, the only supported path uses ShockHistory/PnlFactory and cluster_assets/cluster_elem_trades, which are not available in the personal env. Supporting a path where the user supplies pre-built `pnl_history` in `config.metadata["inference"]` allows testing and production use without rade_sr.

**Suggested change (conceptual):**

1. **Branch on pre-built pnl_history**  
   At the start of `_prepare_new_scenarios_inputs`, if `config.metadata.get("inference", {}).get("pnl_history")` is not None:
   - Use it as the elementary PnL array (convert to numpy, ensure 3D `[B, seq_len, n_elementary]`).
   - Call `_inject_unchanged_inputs(context, "new_scenarios")` to get static parts (trade_features, adjacency_*, elementary/target indices).
   - Build the 7-key model input dict (trade_features, pnl_history, adjacency_indices, adjacency_values, adjacency_dense_shape, elementary_indices, target_indices) as tensors or numpy consistent with the model.
   - Return `{"inputs": <that dict>, "sample_ids": context.trade_universe["target_ids"], "metadata": {...}}`.

2. **Otherwise (raw scenario dir path)**  
   Keep the existing logic (load shocks, update assets, calculate_elementary_pnl, etc.). After building `inputs.elementary_pnl`, add:
   - Build `pnl_history` from `inputs.elementary_pnl` (e.g. shape `[n_scenarios, seq_len, n_elementary]` from data_config or 1).
   - Build the same 7-key dict from `inputs` (trade_features, adjacency_*, indices) and this `pnl_history`.
   - Return `{"inputs": ..., "sample_ids": ..., "metadata": ...}`.

3. **Shared helper**  
   Introduce a helper, e.g. `_build_forward_input_dict(static_inputs: InferenceInputData, pnl_history: np.ndarray) -> Dict[str, Any]`, that produces the 7-key structure expected by the model so both branches use the same contract.

---

### 3.3 _standardise_pnl: support Pipeline scalers (lines 335–343)

**Reason:** The code assumes `scaler.feature_names_in_` and `scaler.mean_`, which is true for a single `StandardScaler` but not for a `Pipeline` (e.g. signed_log_standard). For a Pipeline, the last step may be the scaler, or you may need to run the full pipeline. Documenting the assumption or branching on Pipeline avoids AttributeError and wrong behaviour when the training pipeline uses a composite transform.

**Suggested change (defensive):**

```python
@staticmethod
def _standardise_pnl(pnl_unscaled: pd.DataFrame, scaler: Any) -> np.ndarray:
    """Transform elementary pnl into scaled space, consistent with training.
    Supports a single sklearn scaler (with feature_names_in_ and mean_) or
    a Pipeline whose final step is a scaler.
    """
    if hasattr(scaler, "steps") and len(scaler.steps):
        # Pipeline: transform with full pipeline (e.g. signed_log then standardize).
        return scaler.transform(pnl_unscaled)
    # Single scaler
    feat_index = pd.Index(scaler.feature_names_in_).get_indexer(pnl_unscaled.columns.tolist()).tolist()
    padded = np.zeros((pnl_unscaled.shape[0], len(scaler.mean_)))
    padded[:, feat_index] = pnl_unscaled.to_numpy()
    pnl_scaled = scaler.transform(padded)
    return pnl_scaled[:, feat_index]
```

If the training pipeline never uses a Pipeline for elementary PnL, you can instead add a short docstring stating that only a single scaler with `feature_names_in_` and `mean_` is supported.

---

### 3.4 post_infer: inverse scaling (lines 328–329)

**Reason:** Predictions are in the same scaled space as the training targets. For interpretability and downstream use, they should be converted back to original PnL units using the same `target_scaler` saved at train time. Without this, metrics or UI that assume real currency units will be wrong.

**Suggested change:**

```python
def post_infer(self, result: InferenceResult, config: PipelineConfig) -> None:
    """Inference-specific analytics after predictions; inverse-scale to original PnL units when scaler is available."""
    try:
        ctx = self.load_inference_context()
        scaler = ctx.get("target_scaler")
        if result.predictions is not None and scaler is not None:
            result.metadata["predictions_scaled"] = result.predictions.copy()
            result.predictions = scaler.inverse_transform(result.predictions)
    except Exception as e:
        logger.debug("Could not apply target_scaler inverse transform: %s", e)
```

(Use a cached context if you already have it to avoid reloading; the above is minimal.)

---

## 4. Summary table

| File     | Location        | Reason | Action |
|----------|-----------------|--------|--------|
| train.py | _post_train_plots | data_config may be dataclass | Use getattr / dict.get for plot_pnl_distribution, plot_trade_graph |
| train.py | trade_universe    | sequence_length default `int` not JSON-serialisable | Use default `1` |
| eval.py  | imports           | Production must not import from tests | Remove cluster_mapping import |
| eval.py  | _load_cached_data  | cluster_info saved as .joblib not .json | Load from cluster_info.joblib |
| eval.py  | load_additional_data | Crashes when cluster_info/path/files missing | Guard and return {} when absent |
| eval.py  | run()              | Wrong log level for test complete | Use logger.info |
| eval.py  | _validate_inputs   | Asserts fail when add_data is empty | Only assert m1/m2/pnl when kwargs non-empty |
| eval.py  | run()              | data_config may be dataclass | Use helper for validation_split, test_split |
| eval.py  | _load_cached_data  | data_config can be None | Use safe default batch_size when building DataLoaders |
| infer.py | load_inference_context | Wrong condition and overwrites | Branch on p.suffix; load each file once |
| infer.py | _prepare_new_scenarios_inputs | Returns empty dict; no 7-key input | Build and return inputs + sample_ids + metadata; support pre-built pnl_history |
| infer.py | _standardise_pnl   | Pipeline scaler has no feature_names_in_/mean_ | Support Pipeline or document assumption |
| infer.py | post_infer         | Predictions stay in scaled space | Apply target_scaler.inverse_transform when available |

Implementing these in the order above (train → eval → infer) will align the pipelines with the intended behaviour and avoid the crashes and inconsistencies described.
