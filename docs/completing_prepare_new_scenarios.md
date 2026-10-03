# Completing `_prepare_new_scenarios_inputs()` — Summary

## Current State

Steps 1–8 are implemented and do the right thing:

| Step | What it does | Status |
|------|-------------|--------|
| 1 | Load new risk-factor shock CSVs from `new_scenario_dir` | Done |
| 2 | Inject unchanged static inputs (trade_features, adjacency, indices) into `InferenceInputData` | Done |
| 3 | Deep-copy asset portfolio with new shocks injected | Done |
| 4 | Filter elementary trades to match the cluster's reduced population | Done |
| 5 | Calculate elementary PnL for new scenarios via `ShockHistory` / `PnlFactory` | Done |
| 6 | Concatenate elementary PnL across assets, reorder to training column order | Done |
| 7 | Standardise elementary PnL using saved scaler | Done (but has a bug — see Issue A) |
| 8 | Store scaled PnL DataFrame on `InferenceInputData` | Done |
| 9 | Build inference input from sequences | **Empty** |
| 10 | Assemble and return final dict | **Returns `{}` — broken** |

Steps 9–10 are where the function fails. It returns an empty dict, but the base class
(`InferencePipeline.run()`) expects:

```python
prepared = self.prepare_inputs(self.config)
inputs = prepared["inputs"]          # 7-key dict fed to model()
sample_ids = prepared.get("sample_ids")
metadata = prepared.get("metadata")
```

And `InferenceRunner.predict()` does a **single** `model(inputs)` call — no DataLoader,
no batching. So `prepare_inputs()` must return the fully assembled model input as a plain
dict of numpy arrays.

---

## Issues to Fix

### Issue A — `_standardise_pnl` assumes sklearn StandardScaler

**Current code (line 336–345):**
```python
@staticmethod
def _standardise_pnl(pnl_unscaled: pd.DataFrame, scaler: Any) -> np.ndarray:
    feat_index = pd.Index(scaler.feature_names_in_).get_indexer(pnl_unscaled.columns.tolist()).tolist()
    padded = np.zeros((pnl_unscaled.shape[0], len(scaler.mean_)))  # <-- BUG: scaler.mean_
    padded[:, feat_index] = pnl_unscaled.to_numpy()
    pnl_scaled = scaler.transform(padded)
    return pnl_scaled[:, feat_index]
```

**Problem:** Uses `scaler.mean_` to determine the number of features for the padded array.
This only works for sklearn `StandardScaler`. If the training pipeline used
`signed_log_standard` (a `Pipeline([SignedLogTransformer, StandardScaler])`), the outer
object has no `mean_` attribute — it lives on `scaler.named_steps["scaler"].mean_`.

**Fix:** Use `scaler.feature_names_in_` (which we explicitly set after fit) to determine
width, and don't reference `mean_` at all:

```python
@staticmethod
def _standardise_pnl(pnl_unscaled: pd.DataFrame, scaler: Any) -> np.ndarray:
    scaler_features = list(scaler.feature_names_in_)
    feat_index = [scaler_features.index(c) for c in pnl_unscaled.columns.tolist()]
    n_features = len(scaler_features)
    padded = np.zeros((pnl_unscaled.shape[0], n_features), dtype=np.float32)
    padded[:, feat_index] = pnl_unscaled.to_numpy()
    pnl_scaled = scaler.transform(padded)
    return pnl_scaled[:, feat_index]
```

This works for any scaler/pipeline that exposes `.feature_names_in_` and `.transform()`.

---

### Issue B — `load_inference_context` operator precedence bug

**Current code (line 236):**
```python
if p.exists() and str(p).endswith(".joblib") or str(p).endswith(".pkl"):
```

**Problem:** Python operator precedence makes this:
```python
(p.exists() and endswith(".joblib")) or endswith(".pkl")
```
So `.pkl` files are loaded even when `p.exists()` is False (which would then fail).

**Fix:** Add parentheses:
```python
if p.exists() and (str(p).endswith(".joblib") or str(p).endswith(".pkl")):
```

---

### Issue C — `load_inference_context` missing `target_scaler`

**What train.py saves:** `target_scaler.pkl` (line 140 of train.py).

**What `load_inference_context` loads:** No `target_scaler` entry in the file list.

The `InferenceContext` dataclass *has* a `target_scaler` field, and `build_inference_context`
maps `context.get("target_scaler")` to it. But the loading loop never loads the file.

**Fix:** Add `("target_scaler", "target_scaler.pkl")` to the file list in
`load_inference_context`.

---

### Issue D — `data_config` is a dict (loaded from JSON), not a dataclass

`load_inference_context` loads `data_config.json` via `json.load()`, so
`context.data_config` is a plain `dict`. Any code that accesses it must use
`context.data_config["seq_length"]`, not `context.data_config.seq_length`.

This is relevant for step 9 where we need `seq_length`.

---

## What Steps 9–10 Need to Do

### Step 9 — Build PnL sequences (same windowing as training)

At this point we have:
- `inputs.elementary_pnl`: DataFrame `[N_new_scenarios, n_elementary]` — scaled
- `context.data_config["seq_length"]`: the sequence length used in training

We need to window the elementary PnL into sequences of shape `[N_windows, seq_length, n_elementary]`
using the same `_build_pnl_sequences()` function that training uses.

`_build_pnl_sequences` requires a `target_pnl` array for index alignment (the target at
position `start + seq_length - 1`). At inference we have no target PnL for new scenarios,
but the function only uses it for indexing — the returned target array is discarded. A
zeros placeholder of the correct shape works.

```python
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import _build_pnl_sequences, window_starts_from_days

seq_length = context.data_config["seq_length"]
n_scenarios = inputs.elementary_pnl.shape[0]

# All new scenarios are contiguous: indices 0..N-1
inference_starts, _ = window_starts_from_days(
    scenario_idx=np.arange(n_scenarios),
    sequence_length=seq_length,
)

# Placeholder — _build_pnl_sequences uses target_pnl only for row indexing
target_placeholder = np.zeros(
    (n_scenarios, len(inputs.target_indices)), dtype=np.float32
)

elem_seq, _ = _build_pnl_sequences(
    elementary_pnl=inputs.elementary_pnl.to_numpy(),
    target_pnl=target_placeholder,
    period_starts=inference_starts,
    sequence_length=seq_length,
)
```

`elem_seq` is now `[N_windows, seq_length, n_elementary]` — identical shape and windowing
to what the model saw during training.

### Step 10 — Assemble the 7-key model input dict and return

The model's `forward()` expects exactly these keys:

```
trade_features          [n_trades, n_features]     — from encoder (static)
pnl_history             [N, seq_length, n_elem]    — from step 9 (variable)
adjacency_indices       [2, nnz]                   — from graph builder (static)
adjacency_values        [nnz]                      — from graph builder (static)
adjacency_dense_shape   [2]                        — from graph builder (static)
elementary_indices      [n_elementary]              — from trade universe (static)
target_indices          [n_targets]                 — from trade universe (static)
```

All the static pieces already live on the `inputs` object (populated by
`_inject_unchanged_inputs` at step 2). We just need to combine them with `elem_seq`:

```python
model_inputs = {
    "pnl_history":          elem_seq,
    "trade_features":       inputs.trade_features,
    "adjacency_indices":    inputs.adjacency_indices,
    "adjacency_values":     inputs.adjacency_values,
    "adjacency_dense_shape": np.array(inputs.adjacency_dense_shape, dtype=np.int64),
    "elementary_indices":   np.array(inputs.elementary_indices, dtype=np.int64),
    "target_indices":       np.array(inputs.target_indices, dtype=np.int64),
}

return {
    "inputs": model_inputs,
    "sample_ids": inputs.target_ids,
    "metadata": {
        "mode": "new_scenarios",
        "n_scenarios": elem_seq.shape[0],
        "seq_length": seq_length,
        "elementary_ids": inputs.elementary_ids,
        "target_ids": inputs.target_ids,
    },
}
```

`InferenceRunner.predict()` then:
1. Converts each numpy array to a tensor and moves it to the model device.
2. Calls `model(inputs)` once under `torch.no_grad()`.
3. Returns `InferenceResult` with `.predictions` as a numpy array `[N, n_targets]`.

---

## How `post_infer` Completes the Pipeline

After `runner.predict()` returns, the base class calls `post_infer(result, config)`.
This is currently empty. It needs to:

1. **Inverse-scale the predictions** using the saved `target_scaler`:

```python
def post_infer(self, result: InferenceResult, config: PipelineConfig) -> None:
    context = self.build_inference_context()

    if context.target_scaler is not None:
        raw = result.predictions                          # [N, n_targets]
        n_target_features = len(context.target_scaler.feature_names_in_)

        # Pad if scaler was fit on more features than the prediction width
        if raw.shape[1] < n_target_features:
            padded = np.zeros((raw.shape[0], n_target_features), dtype=np.float32)
            padded[:, :raw.shape[1]] = raw
            unscaled = context.target_scaler.inverse_transform(padded)
            result.predictions = unscaled[:, :raw.shape[1]]
        else:
            result.predictions = context.target_scaler.inverse_transform(raw)
```

2. Optionally attach scenario IDs or any reporting metadata.

Note: `build_inference_context()` is called twice (once in `prepare_inputs`, once in
`post_infer`). If this is a concern, the context can be cached on `self` during
`prepare_inputs` and reused.

---

## End-to-End Flow After Fixes

```
InferencePipeline.run()
│
├── 1. load_runner()
│       └── loads model from registry → InferenceRunner
│
├── 2. prepare_inputs(config)
│       └── _prepare_new_scenarios_inputs(config, context)
│           ├── steps 1-6:  load shocks → compute elementary PnL
│           ├── step 7:     _standardise_pnl(pnl, elementary_scaler)  [fix: no mean_ ref]
│           ├── step 8:     store scaled PnL on InferenceInputData
│           ├── step 9:     _build_pnl_sequences(elem_pnl, placeholder, starts, seq_length)
│           └── step 10:    return {"inputs": {7-key dict}, "sample_ids": [...], "metadata": {...}}
│
├── 3. runner.predict(inputs={7-key dict}, sample_ids=[...], metadata={...})
│       ├── _prepare_inputs():  numpy → tensor, move to device
│       ├── model(prepared):    single forward pass under torch.no_grad()
│       └── → InferenceResult(predictions=[N, n_targets], ...)
│
└── 4. post_infer(result, config)
        └── target_scaler.inverse_transform(result.predictions)
            → result.predictions now in original PnL units
```

---

## Changes Checklist

| # | File | Change | Why |
|---|------|--------|-----|
| 1 | `infer.py` `_standardise_pnl` | Replace `len(scaler.mean_)` with `len(scaler.feature_names_in_)` | Works with any scaler/pipeline, not just StandardScaler |
| 2 | `infer.py` `load_inference_context` | Fix operator precedence: add parentheses | `.pkl` files incorrectly loaded when file missing |
| 3 | `infer.py` `load_inference_context` | Add `("target_scaler", "target_scaler.pkl")` to file list | Needed for inverse-scaling predictions in `post_infer` |
| 4 | `infer.py` `_prepare_new_scenarios_inputs` | Implement steps 9–10 (sequence building + dict assembly) | Core: produces the model input that the base class and runner expect |
| 5 | `infer.py` `post_infer` | Implement inverse scaling via `target_scaler` | Returns predictions in original PnL units |
| 6 | `dataset.py` | Revert `RadeInferenceDataset`, `_collate_inference_batch`, `build_inference_dataloader` | Not needed — runner does single-pass, not DataLoader iteration |
| 7 | `build.py` | Revert `_make_inference_dataloader` and `torch` import | Not needed — same reason |
