# rade_xl — Documentation

A model-independent framework for machine learning and reinforcement learning
in quantitative finance. A model author supplies the architecture, the data
build and the fitted state; the framework supplies everything else —
training, evaluation, inference, tuning, bundles, reports, fan-out and
placement.

---

## Start here

| You want to | Read | Length |
| --- | --- | --- |
| Understand what this is and how it fits together | [`ARCHITECTURE.md`](ARCHITECTURE.md) | Long — the reference |
| Build the next piece of it | [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Medium — the plan |
| Know the standard to write to | [`CODING_STANDARDS.md`](CODING_STANDARDS.md) | Medium — the rules |
| Work on one specific phase | [`phases/`](phases/) | Long — the specifications |
| Know what belongs in a package | That package's `__init__.py` | Short — the charter |

**Picking up the build cold?** Read `ARCHITECTURE.md`, then
`CODING_STANDARDS.md`, then go to the phase map in `IMPLEMENTATION.md` §4 and
open the lowest phase not marked complete.

**Looking for one module?** Every package's `__init__.py` docstring lists the
modules it will hold and the phase that delivers each. That is the fastest
route from "where does X live?" to the right document.

---

## The phases

| Phase | Document | Delivers |
| :---: | --- | --- |
| 1 | [Core](phases/PHASE_1_CORE.md) | Specs, contracts, capabilities, runtime, storage, analysis bases |
| 2 | [Torch engine](phases/PHASE_2_TORCH_ENGINE.md) | PyTorch engine, dataset source, `TrainPipeline` — first model trains |
| 0+3 | [Baseline](phases/PHASE_0_BASELINE.md) + [Flagship](phases/PHASE_3_HYBRID_GNN_RNN.md) | Golden fixture and parity harness, then the hybrid graph-temporal network, single member, at parity |
| 4 | [Job sets](phases/PHASE_4_JOB_SETS.md) | Fan-out across many jobs; executors; P&L domain |
| 5 | [Evaluate · infer · tune](phases/PHASE_5_EVALUATE_INFER_TUNE.md) | The other three pipelines |
| 6 | [More engines](phases/PHASE_6_ADDITIONAL_ENGINES.md) | XGBoost, scikit-learn, baseline models |
| 7 | [Reinforcement learning](phases/PHASE_7_REINFORCEMENT_LEARNING.md) | Environments, learners, differentiable hedging |

---

## Checks

```bash
.venv/bin/python -m ruff check  src/rade_xl tests/rade_xl
.venv/bin/python -m ruff format --check src/rade_xl tests/rade_xl
.venv/bin/python -m pytest tests/rade_xl -q
```

All three must be clean for any phase to be considered complete. Configuration
is in [`../ruff.toml`](../ruff.toml).
