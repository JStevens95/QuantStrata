# m_learning unit tests

Tests mirror the `src/m_learning` structure:

- **core/** — types (TrainingResult, EvaluationResult, TuningResult), config
- **data/** — dataset, pricing build, gnn_rnn_hybrid build, common
- **pipeline/** — training, evaluation, inference, tuning
- **evaluation/** — delta hedging backtest
- **models/pricing/** — config
- **models/gnn_rnn_hybrid/** — config

## Running tests

All tests require **TensorFlow** to be installed and importable. From the project root:

```bash
pytest tests/unit/m_learning/ -v
```

To run a subset:

```bash
pytest tests/unit/m_learning/core/ tests/unit/m_learning/pipeline/test_tuning.py -v
```

If TensorFlow fails to load (e.g. ABI mismatch on your platform), fix the TF installation or use a compatible environment; the tests themselves do not mock TF.
