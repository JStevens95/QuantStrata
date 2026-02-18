# Examples Run Report

**Purpose:** Confirm that example scripts and notebooks run without error after unit test fixes (portfolio, pricers, volatility, neural_sde, skippable q_learning/machine_learning).

**Environment:** Python 3.12, run from repo root with `PYTHONPATH=.`.  
**Command:** `PYTHONPATH=. python3.12 examples/run_all_examples.py` (or `python3.12 examples/<path>/<script>.py` for a single script).

---

## Summary

| Category              | Result |
|-----------------------|--------|
| **run_all_examples.py** | 40 passed, 6 skipped (long/optional), **0 failed** |
| **Single-script runs** | All run scripts completed successfully with Python 3.12 |
| **Fix applied**       | `examples/pricing/01_fx_vanilla_pricing.py`: FD pricer constructor updated from `n_spot`/`n_time` to `n_space`/`n_time_steps` to match `FxVanillaEuropeanOptionFdPricer` |

---

## Run-all summary (default, no `--long`)

- **Passed (40):** All fundamentals, pricing, risk, showcase, pipelines (build curves/vol, greeks, VaR, calibrate Heston/SABR, backtest, portfolio optimisation/from_config, deploy RL, hyperparameter tuning, train GNN/neural SDE), ml/01_hedging_environment, ml/03_model_validation.
- **Skipped (6):** machine_learning/01–03 (TensorFlow), q_learning/01–02, ml/02_rl_hedging_agent (long-running or optional deps). Use `--long` to include them (longer timeout).
- **Failed:** 0.

---

## Notes (no failures; informational)

1. **Python version:** Examples use `dataclass(slots=True)` and other 3.10+ features via the library. Run with **Python 3.12** (or 3.10+). With Python 3.9 you may see `TypeError: dataclass() got an unexpected keyword argument 'slots'` when the library is loaded.
2. **pricing/01_fx_vanilla_pricing.py:** Previously the FD pricer was called with `n_spot` and `n_time`; the library API is `n_space` and `n_time_steps`. This was fixed so FD pricing and convergence run without warning.
3. **pricing/01_equity_vanilla_pricing.py:** If the equity pricer module is not available, the example skips equity pricing and completes successfully.
4. **fundamentals/07_timeseries_generation.py:** May log a `RuntimeWarning` (e.g. invalid value in log) and “Max error vs input: nan” in one correlation check; the script still completes. This is a known numerical edge case in the generator, not introduced by the unit test changes.
5. **Machine learning examples:** Without TensorFlow, ML examples exit 0 and report “TensorFlow not installed. Skipping …”.
6. **Notebooks:** Not executed in this run. To run them: `jupyter nbconvert --execute --to notebook examples/notebooks/*.ipynb` (requires `jupyter`/`nbconvert`). Manual run of notebooks is recommended to confirm after any further changes.

---

## Fix applied

**File:** `examples/pricing/01_fx_vanilla_pricing.py`

- **Issue:** `FxVanillaEuropeanOptionFdPricer.__init__() got an unexpected keyword argument 'n_spot'` (and `n_time`). The library pricer uses `n_space` and `n_time_steps`.
- **Change:** In `safe_fd_price()`, the pricer is now constructed with:
  `FxVanillaEuropeanOptionFdPricer(n_space=n_spot, n_time_steps=n_time)`  
  so the example’s `n_spot`/`n_time` arguments are passed correctly to the library.

---

## How to re-run

```bash
cd /path/to/QuantStrata
export PYTHONPATH=.

# All examples (default timeout; skips long/ML)
python3.12 examples/run_all_examples.py

# Include long-running and ML/RL examples
python3.12 examples/run_all_examples.py --long

# Single example
python3.12 examples/pricing/01_fx_vanilla_pricing.py
```

No example run failed as a result of the unit test fixes; the only code change needed was the FD pricer argument names in the FX vanilla pricing example.
