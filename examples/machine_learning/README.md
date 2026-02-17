# Machine Learning Examples (Neural Pricers, Calibration, Neural SDEs)

Production-oriented examples for ML in quantitative finance: neural option pricing, model calibration, and neural SDEs. Designed for hedge fund / front-office use.

## Examples

### 1. `01_neural_pricer.py` – Neural Network Option Pricer

**Learning objectives**
- Use the library's `MLPPricer` and training pipeline
- Generate training data from analytical BSM; train and evaluate
- Compare speed and accuracy vs analytical pricing

**Production context**
- Neural pricers enable fast portfolio pricing (10–1000× vs MC)
- Use `ZeroRateCurve` and `GridVolSurface` for real market data (no flat curves in production)
- Validate on out-of-sample data; log and save metrics/artifacts for audit

---

### 2. `02_calibration_ml.py` – ML-Accelerated Heston Calibration

**Learning objectives**
- Formulate calibration as an inverse problem (prices/vols → parameters)
- Train a network to map vol surface → Heston parameters
- Compare optimization-based vs neural calibration; hybrid (ML init + optimization) for production

**Production context**
- Calibration is required for marking, risk, and Greeks
- ML gives a fast warm start; refine with optimization for sign-off
- Use real market smiles and `GridVolSurface`; enforce Feller and parameter bounds

**Flags**
- `--fast`: reduced samples (~1–2 min)
- `--smoke`: minimal run for CI (~30–60 s)

---

### 3. `03_neural_sde.py` – Neural Stochastic Differential Equations

**Learning objectives**
- Use `NeuralSDEDynamics` with drift and diffusion networks
- Train on simulated (or historical) paths; match moments and pathwise behaviour
- Validate against known parametric models (e.g. GBM, Heston)

**Production context**
- Neural SDEs capture complex dynamics beyond parametric models
- Use historical returns and library `Market` (curves, vol surfaces) for realistic inputs
- Validate and enforce no-arbitrage before use in pricing/risk

---

## Running the examples

```bash
cd /path/to/QuantStrata

# Neural pricer (requires TensorFlow)
PYTHONPATH=. python examples/machine_learning/01_neural_pricer.py
PYTHONPATH=. python examples/machine_learning/01_neural_pricer.py --no-plot --seed 42 --output-dir output/ml_neural_pricer

# Calibration (optional --fast / --smoke)
PYTHONPATH=. python examples/machine_learning/02_calibration_ml.py --fast
PYTHONPATH=. python examples/machine_learning/02_calibration_ml.py --smoke --seed 42

# Neural SDE
PYTHONPATH=. python examples/machine_learning/03_neural_sde.py
PYTHONPATH=. python examples/machine_learning/03_neural_sde.py --no-plot --output-dir output/ml_neural_sde
```

## Production checklist (hedge fund use)

- **Reproducibility**: Use `--seed` and record it in logs/artifacts.
- **Audit trail**: Use `--output-dir` to save config, metrics, and plots for each run.
- **Market data**: In production, use `ZeroRateCurve` and `GridVolSurface` (no `FlatZeroRateCurve` / `FlatVolSurface`).
- **Validation**: Always validate against benchmark pricers and out-of-sample data; document error tolerances.
- **Dependencies**: TensorFlow 2.x for neural pricer and calibration; see `requirements.txt`.

## Related

- **ML (RL) examples**: `examples/ml/` – hedging environment and RL agent
- **Library modules**: `src.machine_learning`, `src.models.neural_sde`, `src.calibration`
- **Docs**: `docs/reference/machine_learning/`, `docs/reference/models/`
