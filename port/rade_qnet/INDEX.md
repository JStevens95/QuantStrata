# Porting `rade_qnet` through a markdown-only proxy

80 documents, 364 files, 99,460 lines, 3,741,009 bytes.

Each document below covers one directory: create the directory, then create each file in it from the block that carries it. Rebuild the source tree first, then the tests.

## Where it lands

This export is written for a vendored layout: the package imports as `tranql.models.rade.rade_qnet.rade_qnet` and its suite as `tranql.models.rade.rade_qnet.tests`. Every heading below already names its file's path in that layout, and every import already uses that name, so nothing needs editing after the paste.

The directories above the package must already be packages, which in an existing repository they normally are. If any of these is missing, create it empty:

- `tranql/__init__.py`
- `tranql/models/__init__.py`
- `tranql/models/rade/__init__.py`
- `tranql/models/rade/rade_qnet/__init__.py`

Run every command from the repository root -- the directory holding `tranql/` -- since the suite imports the package by its full name.

## What is not here, and what to expect because of it

**`pyproject.toml` does not travel**, because the host repository has its own and pasting this one over it would break it. What it would have contributed is the interpreter and dependency floor, so make sure the host environment has:

- Python `>=3.12`
- `pydantic>=2.6,<3`
- `PyYAML>=6.0`
- `numpy>=1.26`
- `matplotlib>=3.8`

And, for the engines and sources that need them -- tests of anything absent skip rather than fail:

- `torch>=2.2` (torch)
- `xgboost>=2.0` (xgboost)
- `scikit-learn>=1.4` (sklearn)
- `joblib>=1.3` (sklearn)
- `scipy>=1.11` (hybrid)
- `pandas>=2.0` (parquet)
- `pyarrow>=14.0` (parquet)
- `ruff>=0.6` (dev)
- `pytest>=8.0` (dev)

**The golden parity fixtures do not travel.** `tests/fixtures/rade_qnet/` holds `.npy` and `.npz` arrays — binary, and so impossible to carry as text. They guard numerical parity against a captured reference, so if that matters on the far side the arrays have to cross by some other route. The suite looks for them under `tests/fixtures/rade_qnet/golden` at the root the package is imported from; to keep them anywhere else, point the `RADE_QNET_GOLDEN_ROOT` environment variable at the directory holding `hybrid_gnn_rnn/`.

Every test that needs them skips cleanly, so **a correct paste is all-green** and any red at all means something did not land. From the repository root, run the suite and compare:

```
pytest tranql/models/rade/rade_qnet/tests
  -> 3070 passed, 101 skipped
```

The 66 extra skips relative to a full checkout are the parity tests: the ones that compare this implementation's numbers against the original's. Everything else runs, so the suite still proves the framework behaves — it just stops proving it reproduces the baseline's figures.

Nine of the fixtures are `.json` and could in principle travel as text. Copying only those gains nothing: the arrays beside them are what the tests read, so the same tests skip either way. It is all of them or none, and none is a perfectly good answer.

## A note on fence lengths

Almost every block below is fenced with three backticks. A file that spells out a fence of its own gets four, so that it cannot close its own block early. Copy whatever sits *between* the fence lines and the length never matters.

## Source: 39 documents, 177 files, 1,816,475 bytes

The package. Work down the list in order: a parent directory always appears before its children, so the tree is importable at every step.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 1 | [`_root.md`](_root.md) | `tranql/models/rade/rade_qnet/rade_qnet` | 3 | 39,765 |
| 2 | [`analysis.md`](analysis.md) | `tranql/models/rade/rade_qnet/rade_qnet/analysis` | 1 | 955 |
| 3 | [`analysis__metrics.md`](analysis__metrics.md) | `tranql/models/rade/rade_qnet/rade_qnet/analysis/metrics` | 4 | 36,690 |
| 4 | [`analysis__reports.md`](analysis__reports.md) | `tranql/models/rade/rade_qnet/rade_qnet/analysis/reports` | 6 | 48,453 |
| 5 | [`analysis__visuals.md`](analysis__visuals.md) | `tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals` | 9 | 87,465 |
| 6 | [`core.md`](core.md) | `tranql/models/rade/rade_qnet/rade_qnet/core` | 1 | 2,379 |
| 7 | [`core__authoring.md`](core__authoring.md) | `tranql/models/rade/rade_qnet/rade_qnet/core/authoring` | 5 | 46,573 |
| 8 | [`core__contract.md`](core__contract.md) | `tranql/models/rade/rade_qnet/rade_qnet/core/contract` | 9 | 93,124 |
| 9 | [`core__lifecycle.md`](core__lifecycle.md) | `tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle` | 7 | 54,940 |
| 10 | [`core__provenance.md`](core__provenance.md) | `tranql/models/rade/rade_qnet/rade_qnet/core/provenance` | 4 | 26,552 |
| 11 | [`core__spec.md`](core__spec.md) | `tranql/models/rade/rade_qnet/rade_qnet/core/spec` | 10 | 100,011 |
| 12 | [`engines.md`](engines.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines` | 3 | 39,790 |
| 13 | [`engines__sklearn.md`](engines__sklearn.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn` | 2 | 20,710 |
| 14 | [`engines__torch.md`](engines__torch.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/torch` | 5 | 67,949 |
| 15 | [`engines__torch__hardware.md`](engines__torch__hardware.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware` | 4 | 29,422 |
| 16 | [`engines__torch__learners.md`](engines__torch__learners.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/learners` | 3 | 24,183 |
| 17 | [`engines__torch__training.md`](engines__torch__training.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training` | 5 | 76,120 |
| 18 | [`engines__xgboost.md`](engines__xgboost.md) | `tranql/models/rade/rade_qnet/rade_qnet/engines/xgboost` | 2 | 29,615 |
| 19 | [`models.md`](models.md) | `tranql/models/rade/rade_qnet/rade_qnet/models` | 1 | 5,201 |
| 20 | [`models__hybrid_gnn_rnn.md`](models__hybrid_gnn_rnn.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn` | 8 | 101,948 |
| 21 | [`models__hybrid_gnn_rnn__features.md`](models__hybrid_gnn_rnn__features.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/features` | 4 | 55,030 |
| 22 | [`models__hybrid_gnn_rnn__layers.md`](models__hybrid_gnn_rnn__layers.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers` | 6 | 59,239 |
| 23 | [`models__hybrid_gnn_rnn__pipelines.md`](models__hybrid_gnn_rnn__pipelines.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines` | 4 | 24,917 |
| 24 | [`models__lstm_tabular.md`](models__lstm_tabular.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular` | 5 | 14,758 |
| 25 | [`models__ridge.md`](models__ridge.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/ridge` | 5 | 10,771 |
| 26 | [`models__xgb_tabular.md`](models__xgb_tabular.md) | `tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular` | 5 | 11,131 |
| 27 | [`orchestration.md`](orchestration.md) | `tranql/models/rade/rade_qnet/rade_qnet/orchestration` | 2 | 18,572 |
| 28 | [`orchestration__compute.md`](orchestration__compute.md) | `tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute` | 6 | 48,408 |
| 29 | [`orchestration__jobs.md`](orchestration__jobs.md) | `tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs` | 6 | 58,453 |
| 30 | [`orchestration__pipelines.md`](orchestration__pipelines.md) | `tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines` | 6 | 103,198 |
| 31 | [`orchestration__stages.md`](orchestration__stages.md) | `tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages` | 5 | 46,897 |
| 32 | [`sources.md`](sources.md) | `tranql/models/rade/rade_qnet/rade_qnet/sources` | 1 | 1,247 |
| 33 | [`sources__batching.md`](sources__batching.md) | `tranql/models/rade/rade_qnet/rade_qnet/sources/batching` | 3 | 34,972 |
| 34 | [`sources__dataset.md`](sources__dataset.md) | `tranql/models/rade/rade_qnet/rade_qnet/sources/dataset` | 7 | 97,065 |
| 35 | [`sources__dataset__transforms.md`](sources__dataset__transforms.md) | `tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms` | 6 | 69,429 |
| 36 | [`sources__environment.md`](sources__environment.md) | `tranql/models/rade/rade_qnet/rade_qnet/sources/environment` | 2 | 9,938 |
| 37 | [`storage.md`](storage.md) | `tranql/models/rade/rade_qnet/rade_qnet/storage` | 4 | 36,396 |
| 38 | [`storage__runs.md`](storage__runs.md) | `tranql/models/rade/rade_qnet/rade_qnet/storage/runs` | 4 | 56,728 |
| 39 | [`testkit.md`](testkit.md) | `tranql/models/rade/rade_qnet/rade_qnet/testkit` | 4 | 127,481 |

## Documentation: 2 documents, 14 files, 407,421 bytes

The prose, including `ARCHITECTURE.md`. These are already markdown and could be fetched directly, but they are carried here so they land in the manifest: a truncated paste then shows up as a digest mismatch rather than as a puzzling test failure. Four tests read these files and check the examples in them still parse, so the suite needs them present at these exact paths.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 40 | [`docs.md`](docs.md) | `tranql/models/rade/rade_qnet/rade_qnet/docs` | 6 | 197,086 |
| 41 | [`docs__phases.md`](docs__phases.md) | `tranql/models/rade/rade_qnet/rade_qnet/docs/phases` | 8 | 210,335 |

## Tests: 39 documents, 173 files, 1,517,113 bytes

The suite. Rebuild it after the source and run it as shown above -- that run is what turns a pasted tree into a verified one. Each document's name mirrors the source document it exercises: `tests__core__spec.md` tests `core__spec.md`.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 42 | [`tests.md`](tests.md) | `tranql/models/rade/rade_qnet/tests` | 8 | 82,511 |
| 43 | [`tests__analysis.md`](tests__analysis.md) | `tranql/models/rade/rade_qnet/tests/analysis` | 1 | 368 |
| 44 | [`tests__analysis__metrics.md`](tests__analysis__metrics.md) | `tranql/models/rade/rade_qnet/tests/analysis/metrics` | 4 | 37,818 |
| 45 | [`tests__analysis__reports.md`](tests__analysis__reports.md) | `tranql/models/rade/rade_qnet/tests/analysis/reports` | 6 | 57,218 |
| 46 | [`tests__analysis__visuals.md`](tests__analysis__visuals.md) | `tranql/models/rade/rade_qnet/tests/analysis/visuals` | 9 | 80,365 |
| 47 | [`tests__core.md`](tests__core.md) | `tranql/models/rade/rade_qnet/tests/core` | 1 | 359 |
| 48 | [`tests__core__authoring.md`](tests__core__authoring.md) | `tranql/models/rade/rade_qnet/tests/core/authoring` | 5 | 34,623 |
| 49 | [`tests__core__contract.md`](tests__core__contract.md) | `tranql/models/rade/rade_qnet/tests/core/contract` | 8 | 71,923 |
| 50 | [`tests__core__lifecycle.md`](tests__core__lifecycle.md) | `tranql/models/rade/rade_qnet/tests/core/lifecycle` | 7 | 47,848 |
| 51 | [`tests__core__provenance.md`](tests__core__provenance.md) | `tranql/models/rade/rade_qnet/tests/core/provenance` | 4 | 26,368 |
| 52 | [`tests__core__spec.md`](tests__core__spec.md) | `tranql/models/rade/rade_qnet/tests/core/spec` | 8 | 71,119 |
| 53 | [`tests__engines.md`](tests__engines.md) | `tranql/models/rade/rade_qnet/tests/engines` | 4 | 28,176 |
| 54 | [`tests__engines__sklearn.md`](tests__engines__sklearn.md) | `tranql/models/rade/rade_qnet/tests/engines/sklearn` | 2 | 10,956 |
| 55 | [`tests__engines__torch.md`](tests__engines__torch.md) | `tranql/models/rade/rade_qnet/tests/engines/torch` | 4 | 41,836 |
| 56 | [`tests__engines__torch__hardware.md`](tests__engines__torch__hardware.md) | `tranql/models/rade/rade_qnet/tests/engines/torch/hardware` | 4 | 27,180 |
| 57 | [`tests__engines__torch__learners.md`](tests__engines__torch__learners.md) | `tranql/models/rade/rade_qnet/tests/engines/torch/learners` | 3 | 25,517 |
| 58 | [`tests__engines__torch__training.md`](tests__engines__torch__training.md) | `tranql/models/rade/rade_qnet/tests/engines/torch/training` | 5 | 67,325 |
| 59 | [`tests__engines__xgboost.md`](tests__engines__xgboost.md) | `tranql/models/rade/rade_qnet/tests/engines/xgboost` | 2 | 16,777 |
| 60 | [`tests__models.md`](tests__models.md) | `tranql/models/rade/rade_qnet/tests/models` | 3 | 39,716 |
| 61 | [`tests__models__hybrid_gnn_rnn.md`](tests__models__hybrid_gnn_rnn.md) | `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn` | 9 | 78,431 |
| 62 | [`tests__models__hybrid_gnn_rnn__features.md`](tests__models__hybrid_gnn_rnn__features.md) | `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features` | 5 | 39,014 |
| 63 | [`tests__models__hybrid_gnn_rnn__layers.md`](tests__models__hybrid_gnn_rnn__layers.md) | `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers` | 7 | 43,175 |
| 64 | [`tests__models__hybrid_gnn_rnn__pipelines.md`](tests__models__hybrid_gnn_rnn__pipelines.md) | `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines` | 4 | 21,558 |
| 65 | [`tests__models__lstm_tabular.md`](tests__models__lstm_tabular.md) | `tranql/models/rade/rade_qnet/tests/models/lstm_tabular` | 2 | 4,268 |
| 66 | [`tests__models__ridge.md`](tests__models__ridge.md) | `tranql/models/rade/rade_qnet/tests/models/ridge` | 3 | 6,316 |
| 67 | [`tests__models__xgb_tabular.md`](tests__models__xgb_tabular.md) | `tranql/models/rade/rade_qnet/tests/models/xgb_tabular` | 2 | 3,382 |
| 68 | [`tests__orchestration.md`](tests__orchestration.md) | `tranql/models/rade/rade_qnet/tests/orchestration` | 2 | 16,255 |
| 69 | [`tests__orchestration__compute.md`](tests__orchestration__compute.md) | `tranql/models/rade/rade_qnet/tests/orchestration/compute` | 4 | 34,989 |
| 70 | [`tests__orchestration__jobs.md`](tests__orchestration__jobs.md) | `tranql/models/rade/rade_qnet/tests/orchestration/jobs` | 8 | 76,646 |
| 71 | [`tests__orchestration__pipelines.md`](tests__orchestration__pipelines.md) | `tranql/models/rade/rade_qnet/tests/orchestration/pipelines` | 7 | 119,443 |
| 72 | [`tests__orchestration__stages.md`](tests__orchestration__stages.md) | `tranql/models/rade/rade_qnet/tests/orchestration/stages` | 2 | 5,734 |
| 73 | [`tests__sources.md`](tests__sources.md) | `tranql/models/rade/rade_qnet/tests/sources` | 1 | 356 |
| 74 | [`tests__sources__batching.md`](tests__sources__batching.md) | `tranql/models/rade/rade_qnet/tests/sources/batching` | 3 | 32,042 |
| 75 | [`tests__sources__dataset.md`](tests__sources__dataset.md) | `tranql/models/rade/rade_qnet/tests/sources/dataset` | 5 | 52,151 |
| 76 | [`tests__sources__dataset__transforms.md`](tests__sources__dataset__transforms.md) | `tranql/models/rade/rade_qnet/tests/sources/dataset/transforms` | 6 | 49,869 |
| 77 | [`tests__sources__environment.md`](tests__sources__environment.md) | `tranql/models/rade/rade_qnet/tests/sources/environment` | 2 | 6,435 |
| 78 | [`tests__storage.md`](tests__storage.md) | `tranql/models/rade/rade_qnet/tests/storage` | 4 | 37,120 |
| 79 | [`tests__storage__runs.md`](tests__storage__runs.md) | `tranql/models/rade/rade_qnet/tests/storage/runs` | 4 | 45,204 |
| 80 | [`tests__testkit.md`](tests__testkit.md) | `tranql/models/rade/rade_qnet/tests/testkit` | 5 | 76,722 |

## Verifying the result

Every file above carries the first 16 hex characters of its SHA-256. Save the manifest at the end of this page as `MANIFEST.txt` in the repository root, then run the script below from the same place. It names every file that is missing or whose contents differ, which is the one check that turns a silent bad paste into a reported one.

```python
"""Check a ported tree against MANIFEST.txt. Run from the repository root."""

import hashlib
import pathlib
import sys

missing: list[str] = []
differs: list[str] = []
checked = 0

for line in pathlib.Path("MANIFEST.txt").read_text().splitlines():
    if not line.strip():
        continue
    expected, _, name = line.partition("  ")
    path = pathlib.Path(name)
    if not path.exists():
        missing.append(name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if actual != expected:
        differs.append(name)
    checked += 1

for name in missing:
    print(f"MISSING  {name}")
for name in differs:
    print(f"DIFFERS  {name}")
print(f"\nchecked {checked}, missing {len(missing)}, differs {len(differs)}")
sys.exit(1 if missing or differs else 0)
```

## Manifest

```
ed6532d66a586898  tranql/models/rade/rade_qnet/rade_qnet/__init__.py
97da3806b2c8875e  tranql/models/rade/rade_qnet/rade_qnet/api.py
1cfe14c3882b84ad  tranql/models/rade/rade_qnet/rade_qnet/ruff.toml
c610c7393d3bee5b  tranql/models/rade/rade_qnet/rade_qnet/analysis/__init__.py
6bc47bdcc1b578ae  tranql/models/rade/rade_qnet/rade_qnet/analysis/metrics/__init__.py
aba29d7c39dcf912  tranql/models/rade/rade_qnet/rade_qnet/analysis/metrics/drift.py
24c242fef6bec8fe  tranql/models/rade/rade_qnet/rade_qnet/analysis/metrics/quality.py
8d183bb0c63180c1  tranql/models/rade/rade_qnet/rade_qnet/analysis/metrics/regression.py
67191a456d8c17a9  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/__init__.py
700a581a6ca0be3b  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/base.py
1557a6382e2340f6  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/baselines.py
dc286333a6cdc775  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/curves.py
233491a91e32745d  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/quality.py
7632b593c2015988  tranql/models/rade/rade_qnet/rade_qnet/analysis/reports/summary.py
c99bfeb943fc679d  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/__init__.py
f707b3b1534e74a7  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/data.py
e527baf067f10b00  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/evaluation.py
99b281603eeebcac  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/export.py
63e4127807280361  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/figures.py
436a49c80ae60a51  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/jobset.py
f6b3228dede4416b  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/style.py
55ea82a6c0e8096a  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/training.py
98bdfe633894c955  tranql/models/rade/rade_qnet/rade_qnet/analysis/visuals/tuning.py
93489b048e7a52fd  tranql/models/rade/rade_qnet/rade_qnet/core/__init__.py
da89724859285c14  tranql/models/rade/rade_qnet/rade_qnet/core/authoring/__init__.py
438101cea82e2646  tranql/models/rade/rade_qnet/rade_qnet/core/authoring/capabilities.py
4c391938e3356d06  tranql/models/rade/rade_qnet/rade_qnet/core/authoring/definition.py
9bfa261bc0bb07fa  tranql/models/rade/rade_qnet/rade_qnet/core/authoring/policy.py
453ad8a21e99e6bd  tranql/models/rade/rade_qnet/rade_qnet/core/authoring/supervised.py
03e5e8f82e62f796  tranql/models/rade/rade_qnet/rade_qnet/core/contract/__init__.py
a1b63e99a6b56f4e  tranql/models/rade/rade_qnet/rade_qnet/core/contract/base.py
c81d47d6593c6bfb  tranql/models/rade/rade_qnet/rade_qnet/core/contract/bundle.py
73648a6e604ea98b  tranql/models/rade/rade_qnet/rade_qnet/core/contract/data.py
4dbd820c64ef1ce7  tranql/models/rade/rade_qnet/rade_qnet/core/contract/requirement.py
2d2f2ea95b63cfb9  tranql/models/rade/rade_qnet/rade_qnet/core/contract/result.py
64516ecd4c330424  tranql/models/rade/rade_qnet/rade_qnet/core/contract/signature.py
0f58535c7104056f  tranql/models/rade/rade_qnet/rade_qnet/core/contract/source.py
c9795030254a3196  tranql/models/rade/rade_qnet/rade_qnet/core/contract/state.py
104b6ad947ef13b3  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/__init__.py
5c564d38da6f6864  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/components.py
4626d883ff35448d  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/context.py
f13eb08c3452f71f  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/errors.py
b9db3f32f535c7ad  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/hooks.py
9e58f9499eb4a1e1  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/pipeline.py
131cf36657c86a5d  tranql/models/rade/rade_qnet/rade_qnet/core/lifecycle/registry.py
b556b6e4ac00495d  tranql/models/rade/rade_qnet/rade_qnet/core/provenance/__init__.py
f6d9f4a484edae26  tranql/models/rade/rade_qnet/rade_qnet/core/provenance/hashing.py
00e4746f8dd6ecf0  tranql/models/rade/rade_qnet/rade_qnet/core/provenance/logging.py
030643171325087b  tranql/models/rade/rade_qnet/rade_qnet/core/provenance/seeding.py
8d034c0a484bd379  tranql/models/rade/rade_qnet/rade_qnet/core/spec/__init__.py
0b868e88f5dd798a  tranql/models/rade/rade_qnet/rade_qnet/core/spec/base.py
7ed9e6228f3c4454  tranql/models/rade/rade_qnet/rade_qnet/core/spec/data.py
4386834de64e33d2  tranql/models/rade/rade_qnet/rade_qnet/core/spec/hardware.py
d4cb787f098fb7e0  tranql/models/rade/rade_qnet/rade_qnet/core/spec/jobs.py
89eeb600e0a48ae8  tranql/models/rade/rade_qnet/rade_qnet/core/spec/merge.py
c34d4853629d6b2d  tranql/models/rade/rade_qnet/rade_qnet/core/spec/reports.py
dd12485b60f37568  tranql/models/rade/rade_qnet/rade_qnet/core/spec/run.py
d652ff166c195ee9  tranql/models/rade/rade_qnet/rade_qnet/core/spec/training.py
c8be6dcfed974aa9  tranql/models/rade/rade_qnet/rade_qnet/core/spec/tune.py
bfa72bb67acebe57  tranql/models/rade/rade_qnet/rade_qnet/engines/__init__.py
373cda51c8e5f5b2  tranql/models/rade/rade_qnet/rade_qnet/engines/base.py
1e5d33305a6210f6  tranql/models/rade/rade_qnet/rade_qnet/engines/loaders.py
9687c52982db50ec  tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn/__init__.py
3da30705e4f82de9  tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn/engine.py
5bbf3471bf2e365e  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/__init__.py
7f8c07ab11295508  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/engine.py
3da70099294e818b  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/loaders.py
6947dc425a1e454a  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/materialise.py
d3bbea51d11a2eef  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/predictor.py
0f5c6d3c17c4d67f  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/__init__.py
0e3900bc3906f0c3  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/determinism.py
f5fbc3699eba7e1b  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/devices.py
9addadb978642a89  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/distributed.py
78173434fb32ab47  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/learners/__init__.py
ef1f1aa9f834ed2b  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/learners/random.py
7cccaac0f285b296  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/learners/supervised.py
1513f7834fd4c319  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training/__init__.py
4f364430b3ee327c  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training/callbacks.py
df134fb5e194563f  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training/checkpoint.py
4cb0b2dc45846113  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training/loops.py
e2df8fc6f1313596  tranql/models/rade/rade_qnet/rade_qnet/engines/torch/training/losses.py
d6ff5a7febff9ffa  tranql/models/rade/rade_qnet/rade_qnet/engines/xgboost/__init__.py
4e410e412fdc4a82  tranql/models/rade/rade_qnet/rade_qnet/engines/xgboost/engine.py
09411121b750cab6  tranql/models/rade/rade_qnet/rade_qnet/models/__init__.py
d99cf28ec67f4ec0  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/__init__.py
73f29f102ea6daaa  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/data.py
7d269cda90d695c5  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/model.py
589cfb05e6a8b23e  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/register.py
5308b7d914f3b947  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/reports.py
a177505b14867755  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/spec.py
3cfb158385bbfb3f  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/state.py
c033d0fcfc405089  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/visuals.py
011ecbecf59e5538  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/features/__init__.py
f0991d6c6cf24d24  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/features/basis.py
b1f1f4ff533ab65f  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/features/encoder.py
53720982a6bf987e  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/features/graph.py
f2731bbc54fbfd3c  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/__init__.py
8965de3a68e489f0  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/attention.py
f48de1d670a94a44  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/fusion.py
7b1e2fec9b208fe6  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/gnn.py
40485abc1ee3cd42  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/projection.py
f69273be0c78e06d  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/layers/rnn.py
2ee06142a0e2a995  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/__init__.py
ec3525c753752a11  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/eval.py
5e713339832a8ff7  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/train.py
c6982aae644e43c9  tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/tune.py
8ed63db8039871d8  tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular/__init__.py
78bb98d6c885d8aa  tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular/data.py
2db3a3716d364c54  tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular/model.py
1e29ed34e6a56eb1  tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular/register.py
684004c8b0263ebe  tranql/models/rade/rade_qnet/rade_qnet/models/lstm_tabular/spec.py
8e878ba14afd847c  tranql/models/rade/rade_qnet/rade_qnet/models/ridge/__init__.py
04a97203d1724ccf  tranql/models/rade/rade_qnet/rade_qnet/models/ridge/data.py
86653e260ade081f  tranql/models/rade/rade_qnet/rade_qnet/models/ridge/model.py
5ec21b14fc34eabe  tranql/models/rade/rade_qnet/rade_qnet/models/ridge/register.py
968365b42f941d08  tranql/models/rade/rade_qnet/rade_qnet/models/ridge/spec.py
8a5273793dc9fa45  tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular/__init__.py
cf2983dc854e17c0  tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular/data.py
c7cf7a7b0632cefd  tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular/model.py
292e398c993f0db6  tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular/register.py
446633bfa7a1d14b  tranql/models/rade/rade_qnet/rade_qnet/models/xgb_tabular/spec.py
41743506e557df79  tranql/models/rade/rade_qnet/rade_qnet/orchestration/__init__.py
c2ee35c80b1a71f6  tranql/models/rade/rade_qnet/rade_qnet/orchestration/serving.py
e51e01a50744d04c  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/__init__.py
4368fcdf46c5da38  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/base.py
518ba800f9e96a02  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/gpus.py
68231f2a0dd188a5  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/local.py
108efe6b640a3a0d  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/placement.py
3b073ef3c554bdce  tranql/models/rade/rade_qnet/rade_qnet/orchestration/compute/processes.py
38c0c4fe2f3800a8  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/__init__.py
9455837cb8ba0832  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/fanout.py
40c05ee56911e539  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/groups.py
80e4c231e0368046  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/manifest.py
6482b8dd9fd29aa8  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/set.py
4b4447c3711fbf4c  tranql/models/rade/rade_qnet/rade_qnet/orchestration/jobs/unit.py
67aa54d4ce6a7acf  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/__init__.py
f9f065ad810b76dc  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/evaluate.py
85df0b89051a7a26  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/infer.py
96b510ee66d5fd62  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/reinforce.py
f61cbb9d86175cd0  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/train.py
dd0d38b33938ebe7  tranql/models/rade/rade_qnet/rade_qnet/orchestration/pipelines/tune.py
d4555a56cb073632  tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages/__init__.py
fd1d9434a83b8a95  tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages/reload.py
52a0bf45099efed5  tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages/resolve.py
2c240ae5ec2ff681  tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages/scoring.py
f3ffe459960cd03e  tranql/models/rade/rade_qnet/rade_qnet/orchestration/stages/search.py
0933cff34ac5afc2  tranql/models/rade/rade_qnet/rade_qnet/sources/__init__.py
705aee492f36f269  tranql/models/rade/rade_qnet/rade_qnet/sources/batching/__init__.py
e9bf1a479fecbcb8  tranql/models/rade/rade_qnet/rade_qnet/sources/batching/dataset.py
a23e2fda7fa4f1bb  tranql/models/rade/rade_qnet/rade_qnet/sources/batching/rollout.py
bf7f3fb0f0814d67  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/__init__.py
a0eab4ed54f687d8  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/cache.py
1336d3633e002262  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/module.py
aa447d0e624c00f6  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/rebuild.py
bcb1ada5e215aa5b  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/splits.py
4aecedb17e4a109a  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/tables.py
329f5b278aac36a3  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/tabular.py
6d3de29fb11b2a0c  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/__init__.py
f47510e9282e3cc7  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/composite.py
904b518e82471413  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/encoding.py
b938b45b7953e380  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/reduction.py
a1932b200ec1be9d  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/scaling.py
f6955b509b3329f8  tranql/models/rade/rade_qnet/rade_qnet/sources/dataset/transforms/sequence.py
8d8eb3deaf1ea40f  tranql/models/rade/rade_qnet/rade_qnet/sources/environment/__init__.py
b6c88ab23fb9cf73  tranql/models/rade/rade_qnet/rade_qnet/sources/environment/protocol.py
5ba06be2a276cadf  tranql/models/rade/rade_qnet/rade_qnet/storage/__init__.py
fcaf136ba9d57f84  tranql/models/rade/rade_qnet/rade_qnet/storage/bundle.py
a6dcb2a486d49d91  tranql/models/rade/rade_qnet/rade_qnet/storage/locking.py
593c50e7d6217c29  tranql/models/rade/rade_qnet/rade_qnet/storage/manifest.py
48dcf2a1299e07de  tranql/models/rade/rade_qnet/rade_qnet/storage/runs/__init__.py
9c922f96d5c8cd98  tranql/models/rade/rade_qnet/rade_qnet/storage/runs/catalog.py
8dcf796a1a0050c8  tranql/models/rade/rade_qnet/rade_qnet/storage/runs/registry.py
d7c5799fb1f03913  tranql/models/rade/rade_qnet/rade_qnet/storage/runs/tracker.py
be147ade92895e66  tranql/models/rade/rade_qnet/rade_qnet/testkit/__init__.py
f34d858b31058be2  tranql/models/rade/rade_qnet/rade_qnet/testkit/conformance.py
a10dd886a237dbfa  tranql/models/rade/rade_qnet/rade_qnet/testkit/fixtures.py
8ce5f0502f170513  tranql/models/rade/rade_qnet/rade_qnet/testkit/parity.py
a3fb518abd400722  tranql/models/rade/rade_qnet/rade_qnet/docs/ARCHITECTURE.md
36ccc834b3a6e610  tranql/models/rade/rade_qnet/rade_qnet/docs/CODING_STANDARDS.md
c82a3b4e2f97e0a3  tranql/models/rade/rade_qnet/rade_qnet/docs/GUIDE.md
ab00d7895cb5df61  tranql/models/rade/rade_qnet/rade_qnet/docs/IMPLEMENTATION.md
94ad0612437a98ee  tranql/models/rade/rade_qnet/rade_qnet/docs/MODEL_IMPLEMENTATION.md
ef24e3568f6a7890  tranql/models/rade/rade_qnet/rade_qnet/docs/README.md
3abb766333e3b5af  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_0_BASELINE.md
3f25eee14d83e7ea  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_1_CORE.md
5b736cdce98c30c9  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_2_TORCH_ENGINE.md
daeb3bc283ff45e1  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_3_HYBRID_GNN_RNN.md
dda53aadcb5305e0  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_4_JOB_SETS.md
f75ec71330030c3b  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_5_EVALUATE_INFER_TUNE.md
d56f0aa8d78b857a  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_6_ADDITIONAL_ENGINES.md
24dfc1d96d0e9ffe  tranql/models/rade/rade_qnet/rade_qnet/docs/phases/PHASE_7_REINFORCEMENT_LEARNING.md
01e950fac0ad5abe  tranql/models/rade/rade_qnet/tests/__init__.py
81d71ab5fafa422c  tranql/models/rade/rade_qnet/tests/conftest.py
59c4e2876f572acb  tranql/models/rade/rade_qnet/tests/locations.py
81fd794fd26156fb  tranql/models/rade/rade_qnet/tests/ruff.toml
bdd9c331304761db  tranql/models/rade/rade_qnet/tests/test_api.py
a4784275d63c97e5  tranql/models/rade/rade_qnet/tests/test_documentation.py
09b30ee0351cac76  tranql/models/rade/rade_qnet/tests/test_extensibility.py
6f2109cbb0ec4c83  tranql/models/rade/rade_qnet/tests/test_scaffold.py
8d2d3c04dfdfe110  tranql/models/rade/rade_qnet/tests/analysis/__init__.py
815690298ea5fb56  tranql/models/rade/rade_qnet/tests/analysis/metrics/__init__.py
2b6c0fa30b125934  tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_drift.py
0c70ea96252aee07  tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_quality.py
c512c2990d0fee32  tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_regression.py
243b3f95c99bd0f3  tranql/models/rade/rade_qnet/tests/analysis/reports/__init__.py
2c6e7e5b832b2fd4  tranql/models/rade/rade_qnet/tests/analysis/reports/test_reports_base.py
c478eeeac9f13769  tranql/models/rade/rade_qnet/tests/analysis/reports/test_reports_baselines.py
dd09c9423042765a  tranql/models/rade/rade_qnet/tests/analysis/reports/test_reports_curves.py
85cf623cb6e1442c  tranql/models/rade/rade_qnet/tests/analysis/reports/test_reports_quality.py
9e74c6fb255b8c37  tranql/models/rade/rade_qnet/tests/analysis/reports/test_reports_summary.py
abc60d222b39299f  tranql/models/rade/rade_qnet/tests/analysis/visuals/__init__.py
39ac02c601c05b1d  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_data.py
592ab8e77ae80b3c  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_evaluation.py
b8cb0504d089af21  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_export.py
f79683c084efecd0  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_figures.py
eacbb8c9dba02b37  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_jobset.py
e8b0a271d7788be6  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_style.py
c2fbaac2f558c796  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_training.py
f49e426812e3f11f  tranql/models/rade/rade_qnet/tests/analysis/visuals/test_visuals_tuning.py
a33185e148d60465  tranql/models/rade/rade_qnet/tests/core/__init__.py
5fa0437358e278ef  tranql/models/rade/rade_qnet/tests/core/authoring/__init__.py
d2243e7ba9bd5bc1  tranql/models/rade/rade_qnet/tests/core/authoring/test_authoring_capabilities.py
0d6fc7f5663edc2b  tranql/models/rade/rade_qnet/tests/core/authoring/test_authoring_definition.py
37f2b43dc508da7d  tranql/models/rade/rade_qnet/tests/core/authoring/test_authoring_policy.py
71f521468b68fc50  tranql/models/rade/rade_qnet/tests/core/authoring/test_authoring_supervised.py
455461a7d762debd  tranql/models/rade/rade_qnet/tests/core/contract/__init__.py
2f6bca7bfc6aac68  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_bundle.py
4b6c56618ee6b8a9  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_data.py
5eb299b8a2a8a99f  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_requirement.py
eebf5dcffee6a44d  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_result.py
6b5862db80e17356  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_signature.py
9119a7440c1661e1  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_source.py
6cf4980731811d54  tranql/models/rade/rade_qnet/tests/core/contract/test_contract_state.py
41150f68fedf6616  tranql/models/rade/rade_qnet/tests/core/lifecycle/__init__.py
4a55edbca998f486  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_components.py
1274a89bf84228fd  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_context.py
6f08e3310be93c4a  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_errors.py
56a3f611e6cd74c6  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_hooks.py
d872d49afa6b1807  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_pipeline.py
169401fc2259d15c  tranql/models/rade/rade_qnet/tests/core/lifecycle/test_lifecycle_registry.py
17f398962e74a13b  tranql/models/rade/rade_qnet/tests/core/provenance/__init__.py
9d2d6f169633a478  tranql/models/rade/rade_qnet/tests/core/provenance/test_provenance_hashing.py
527238a773e1545c  tranql/models/rade/rade_qnet/tests/core/provenance/test_provenance_logging.py
736b32b4683d4832  tranql/models/rade/rade_qnet/tests/core/provenance/test_provenance_seeding.py
4cf0eee3c800d080  tranql/models/rade/rade_qnet/tests/core/spec/__init__.py
25bc288cac73d8ef  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_data.py
b24058dee3eacc8b  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_hardware.py
69029ff7f70932b0  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_jobs.py
debd5485278b451b  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_merge.py
d65a989f17e7791e  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_reports.py
7bb254c2d6ff8f88  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_run.py
cdcc067c7514717c  tranql/models/rade/rade_qnet/tests/core/spec/test_spec_training.py
17c5d2fc92ec9189  tranql/models/rade/rade_qnet/tests/engines/__init__.py
bb1750b7e2f905e4  tranql/models/rade/rade_qnet/tests/engines/test_engine_layout.py
951f4a6254bd4f2b  tranql/models/rade/rade_qnet/tests/engines/test_engines_base.py
367fbd5e980a80e5  tranql/models/rade/rade_qnet/tests/engines/test_engines_loaders.py
b0a1ae83d8b2a9fc  tranql/models/rade/rade_qnet/tests/engines/sklearn/__init__.py
d92376e80569861b  tranql/models/rade/rade_qnet/tests/engines/sklearn/test_sklearn_engine.py
ce84b941ee165fc1  tranql/models/rade/rade_qnet/tests/engines/torch/__init__.py
f3327525ec28b6c7  tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_engine.py
caf4130a6e746d13  tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_loaders.py
f7fbb31f07001372  tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_materialise.py
7860806c8e2de23b  tranql/models/rade/rade_qnet/tests/engines/torch/hardware/__init__.py
8d84ca2045db57b1  tranql/models/rade/rade_qnet/tests/engines/torch/hardware/test_hardware_determinism.py
4eb01b9957fdad4a  tranql/models/rade/rade_qnet/tests/engines/torch/hardware/test_hardware_devices.py
b32a07774b2a40d1  tranql/models/rade/rade_qnet/tests/engines/torch/hardware/test_hardware_distributed.py
347c5ed4a780e803  tranql/models/rade/rade_qnet/tests/engines/torch/learners/__init__.py
2a44ac1a284abfd1  tranql/models/rade/rade_qnet/tests/engines/torch/learners/test_learners_random.py
7ea49e9ffc1d244b  tranql/models/rade/rade_qnet/tests/engines/torch/learners/test_learners_supervised.py
ea056fce628eee7f  tranql/models/rade/rade_qnet/tests/engines/torch/training/__init__.py
ebe7d55b5f351cfa  tranql/models/rade/rade_qnet/tests/engines/torch/training/test_training_callbacks.py
b67c93de9d9c5ce0  tranql/models/rade/rade_qnet/tests/engines/torch/training/test_training_checkpoint.py
7de83861839e99d3  tranql/models/rade/rade_qnet/tests/engines/torch/training/test_training_loops.py
f24828d41fedd51f  tranql/models/rade/rade_qnet/tests/engines/torch/training/test_training_losses.py
d35aed8beca12275  tranql/models/rade/rade_qnet/tests/engines/xgboost/__init__.py
b3d47617564b7f18  tranql/models/rade/rade_qnet/tests/engines/xgboost/test_xgboost_engine.py
aec3631b58e0e12c  tranql/models/rade/rade_qnet/tests/models/__init__.py
99514716aeae0118  tranql/models/rade/rade_qnet/tests/models/test_model_layout.py
f4a8bac2488f354c  tranql/models/rade/rade_qnet/tests/models/test_reference_models.py
5383ca55a0c3650b  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/__init__.py
e630fd59a0014469  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_data.py
1a0289dc02f4c4e4  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_model.py
0937aea9a32f8f33  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_parity.py
b193105cb59734a7  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_register.py
0a0031459e0895d1  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_reports.py
501cd328cd45e06b  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_state.py
9cb3857df194e430  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_universe.py
358c1bc5be3d1ee1  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_visuals.py
7cd332b12fc0f4a5  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/__init__.py
c03a6d739771af29  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/conftest.py
8591510cb75cc9e8  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_basis.py
b9189e00a51b762a  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_encoder.py
23f7649ad5faa458  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/features/test_graph.py
74ed9a32e29b7cd6  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/__init__.py
22199b76430ed4c2  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/conftest.py
ccff859641dcee9d  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_attention.py
fb100899a532acb1  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_fusion.py
c101232729666724  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_gnn.py
0576d467a97225b6  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_projection.py
e6647b6a8262d8f7  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_rnn.py
460b7d1c67ae53f9  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/__init__.py
89d17d7ac49b0874  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_eval.py
188d734e5e26bca9  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_train.py
5bd05794926fdea4  tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_tune.py
e19dbdeabb5b7af6  tranql/models/rade/rade_qnet/tests/models/lstm_tabular/__init__.py
902ae0ea45c6da22  tranql/models/rade/rade_qnet/tests/models/lstm_tabular/test_model.py
f92a7a31ce5f7c72  tranql/models/rade/rade_qnet/tests/models/ridge/__init__.py
f5e2c06ad8fa4475  tranql/models/rade/rade_qnet/tests/models/ridge/test_model.py
dfb1bf05f9a86720  tranql/models/rade/rade_qnet/tests/models/ridge/test_register.py
dfae360f563b4a2f  tranql/models/rade/rade_qnet/tests/models/xgb_tabular/__init__.py
4ca20b5de78e0b35  tranql/models/rade/rade_qnet/tests/models/xgb_tabular/test_model.py
af8945dcb1b57f28  tranql/models/rade/rade_qnet/tests/orchestration/__init__.py
8a62bccf4e4bc4f8  tranql/models/rade/rade_qnet/tests/orchestration/test_orchestration_serving.py
31dd72a61b33b7be  tranql/models/rade/rade_qnet/tests/orchestration/compute/__init__.py
0d8b5b6ec5992161  tranql/models/rade/rade_qnet/tests/orchestration/compute/test_compute_executors.py
fe7229d72769ca53  tranql/models/rade/rade_qnet/tests/orchestration/compute/test_compute_placement.py
84d601ed094a99f4  tranql/models/rade/rade_qnet/tests/orchestration/compute/workers.py
db64606cfe41ff84  tranql/models/rade/rade_qnet/tests/orchestration/jobs/__init__.py
14c04f379adc0598  tranql/models/rade/rade_qnet/tests/orchestration/jobs/support.py
2d148b117a1194f3  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_fanout.py
d4c5e354583927fe  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_groups.py
fe895af0683815e5  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_manifest.py
b229e68eea5e23a7  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_parity.py
932ff6835e38e445  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_set.py
411c6c12bb2d0a17  tranql/models/rade/rade_qnet/tests/orchestration/jobs/test_jobs_unit.py
d46d1a478e03562f  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/__init__.py
14b9f51c1df34a8e  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/support.py
28d1e9e79950a7e1  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_evaluate.py
be43cdcafbbe59a0  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_infer.py
b78854a4531fbd00  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_reinforce.py
09056bdfc76bb41e  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_train.py
e67eee2015ed3f2e  tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_tune.py
861fa175ff9e24b5  tranql/models/rade/rade_qnet/tests/orchestration/stages/__init__.py
18687c4a9a7abf6c  tranql/models/rade/rade_qnet/tests/orchestration/stages/test_stages_resolve.py
edc2c57fa62a87d4  tranql/models/rade/rade_qnet/tests/sources/__init__.py
75d38714d92fee51  tranql/models/rade/rade_qnet/tests/sources/batching/__init__.py
6dd0c4578f5be157  tranql/models/rade/rade_qnet/tests/sources/batching/test_batching_dataset.py
105e8d47ec66ea68  tranql/models/rade/rade_qnet/tests/sources/batching/test_batching_rollout.py
aca25b3ce8fb419d  tranql/models/rade/rade_qnet/tests/sources/dataset/__init__.py
aaa06cc28e137895  tranql/models/rade/rade_qnet/tests/sources/dataset/test_dataset_cache.py
ca08c25f1016ea36  tranql/models/rade/rade_qnet/tests/sources/dataset/test_dataset_module.py
eabbe927b17d99a9  tranql/models/rade/rade_qnet/tests/sources/dataset/test_dataset_splits.py
da37954b5775424f  tranql/models/rade/rade_qnet/tests/sources/dataset/test_dataset_tables.py
2d244a830faebd42  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/__init__.py
32678fa78e49e640  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_composite.py
77ec92acc59bbfc4  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_encoding.py
efdc6746d5981dde  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_reduction.py
3ef1d50bfe25cc83  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_scaling.py
a72496bc7acf552b  tranql/models/rade/rade_qnet/tests/sources/dataset/transforms/test_transforms_sequence.py
ff225df9a8ee5c59  tranql/models/rade/rade_qnet/tests/sources/environment/__init__.py
19062baa2f665154  tranql/models/rade/rade_qnet/tests/sources/environment/test_environment_protocol.py
3ccdb89a57a83475  tranql/models/rade/rade_qnet/tests/storage/__init__.py
30d310a70aa3f3ab  tranql/models/rade/rade_qnet/tests/storage/test_storage_bundle.py
b73beb8968131ac2  tranql/models/rade/rade_qnet/tests/storage/test_storage_locking.py
c78f2996c4c37a29  tranql/models/rade/rade_qnet/tests/storage/test_storage_manifest.py
3c6b979e4a3d6f08  tranql/models/rade/rade_qnet/tests/storage/runs/__init__.py
bd5a886c9deefc16  tranql/models/rade/rade_qnet/tests/storage/runs/test_runs_catalog.py
94b5ac8ed244d8b3  tranql/models/rade/rade_qnet/tests/storage/runs/test_runs_registry.py
eb53007fc6451b77  tranql/models/rade/rade_qnet/tests/storage/runs/test_runs_tracker.py
4dd05188c31c1115  tranql/models/rade/rade_qnet/tests/testkit/__init__.py
87652ac5eefc6681  tranql/models/rade/rade_qnet/tests/testkit/test_testkit_conformance.py
d1dffce957e4dd7c  tranql/models/rade/rade_qnet/tests/testkit/test_testkit_fixtures.py
645ba597e80df7a6  tranql/models/rade/rade_qnet/tests/testkit/test_testkit_golden_fixture.py
fc8b37eef922fcfc  tranql/models/rade/rade_qnet/tests/testkit/test_testkit_parity.py
```

