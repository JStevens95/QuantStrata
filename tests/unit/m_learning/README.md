# m_learning unit tests

Tests mirror the `src/m_learning` structure:

- **core/** — types (TrainingResult, EvaluationResult, TuningResult), config
- **data/** — dataset, pricing build, gnn_rnn_hybrid build, common
- **pipeline/** — training, evaluation, inference, tuning
- **evaluation/** — delta hedging backtest
- **models/pricing/** — config
- **models/gnn_rnn_hybrid/** — config

## Requirements

- **Python 3.12+** (required; conftest will exit with a clear message if an older Python is used)
- **TensorFlow >= 2.20** (see `requirements.txt`)

TensorFlow must be importable in the same environment you use to run pytest. There should be no import issue when using Python 3.12 and TensorFlow >= 2.20. If you see an "Aborted" crash, the test runner is likely using a different interpreter (e.g. Anaconda Python 3.9); run pytest explicitly with Python 3.12.

## Running tests

From the project root, using Python 3.12 and an environment where TensorFlow is installed:

```bash
python3.12 -m pytest tests/unit/m_learning/ -v
```

Or, if your default `pytest` is already Python 3.12:

```bash
pytest tests/unit/m_learning/ -v
```

To run a subset:

```bash
python3.12 -m pytest tests/unit/m_learning/core/ tests/unit/m_learning/pipeline/test_tuning.py -v
```

## If tests fail on TensorFlow import

1. Check the interpreter: `python --version` or `python3 --version` should be **3.12.x**.
2. Check TensorFlow: `python -c "import tensorflow as tf; print(tf.__version__)"` should succeed and print >= 2.20.
3. Run pytest with that same interpreter: `python3.12 -m pytest tests/unit/m_learning/ -v` (replace `python3.12` with your 3.12 binary if needed).

If your IDE or runner uses a different Python (e.g. Anaconda 3.9), point it at the Python 3.12 environment where TensorFlow is installed.
