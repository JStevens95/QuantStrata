# Porting `rade_qnet` through a markdown-only proxy

81 documents, 364 files, 98,780 lines, 3,702,676 bytes.

Each document below covers one directory: create the directory, then create each file in it from the block that carries it. Rebuild the source tree first, then the tests.

## What is not here, and what to expect because of it

Everything needed to install and run is carried, including `pyproject.toml` (the first document, so the rebuilt tree installs with `pip install -e .`) and the documentation.

**The golden parity fixtures do not travel.** `tests/fixtures/rade_qnet/` holds `.npy` and `.npz` arrays — binary, and so impossible to carry as text. They guard numerical parity against a captured reference, so if that matters on the far side the arrays have to cross by some other route.

Most tests that need them skip cleanly. **Twenty-six do not** — they fail or error on the missing file instead. That is a gap in those tests rather than in this port, but it means a correct paste is *not* all-green. Run the suite and compare against the expected result below; anything else means something did not land.

```
pytest tests/rade_qnet
  -> 7 failed, 3072 passed, 70 skipped, 19 errors
```

**Do not copy across a subset of the fixtures.** The nine `.json` files among them are text and look portable, but supplying those without the arrays is worse than supplying none: the loader then finds the directory, the tests stop skipping, and the failure count rises to 22. It is all of them or none.

## A note on fence lengths

Almost every block below is fenced with three backticks. A file that spells out a fence of its own gets four, so that it cannot close its own block early. Copy whatever sits *between* the fence lines and the length never matters.

## Source: 40 documents, 178 files, 1,820,630 bytes

The package. Work down the list in order: a parent directory always appears before its children, so the tree is importable at every step.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 1 | [`_repository.md`](_repository.md) | `.` | 1 | 5,974 |
| 2 | [`_root.md`](_root.md) | `src/rade_qnet` | 3 | 39,605 |
| 3 | [`analysis.md`](analysis.md) | `src/rade_qnet/analysis` | 1 | 955 |
| 4 | [`analysis__metrics.md`](analysis__metrics.md) | `src/rade_qnet/analysis/metrics` | 4 | 36,690 |
| 5 | [`analysis__reports.md`](analysis__reports.md) | `src/rade_qnet/analysis/reports` | 6 | 48,453 |
| 6 | [`analysis__visuals.md`](analysis__visuals.md) | `src/rade_qnet/analysis/visuals` | 9 | 87,465 |
| 7 | [`core.md`](core.md) | `src/rade_qnet/core` | 1 | 2,379 |
| 8 | [`core__authoring.md`](core__authoring.md) | `src/rade_qnet/core/authoring` | 5 | 46,573 |
| 9 | [`core__contract.md`](core__contract.md) | `src/rade_qnet/core/contract` | 9 | 93,124 |
| 10 | [`core__lifecycle.md`](core__lifecycle.md) | `src/rade_qnet/core/lifecycle` | 7 | 54,940 |
| 11 | [`core__provenance.md`](core__provenance.md) | `src/rade_qnet/core/provenance` | 4 | 25,656 |
| 12 | [`core__spec.md`](core__spec.md) | `src/rade_qnet/core/spec` | 10 | 100,011 |
| 13 | [`engines.md`](engines.md) | `src/rade_qnet/engines` | 3 | 39,771 |
| 14 | [`engines__sklearn.md`](engines__sklearn.md) | `src/rade_qnet/engines/sklearn` | 2 | 20,710 |
| 15 | [`engines__torch.md`](engines__torch.md) | `src/rade_qnet/engines/torch` | 5 | 67,930 |
| 16 | [`engines__torch__hardware.md`](engines__torch__hardware.md) | `src/rade_qnet/engines/torch/hardware` | 4 | 29,422 |
| 17 | [`engines__torch__learners.md`](engines__torch__learners.md) | `src/rade_qnet/engines/torch/learners` | 3 | 24,183 |
| 18 | [`engines__torch__training.md`](engines__torch__training.md) | `src/rade_qnet/engines/torch/training` | 5 | 76,120 |
| 19 | [`engines__xgboost.md`](engines__xgboost.md) | `src/rade_qnet/engines/xgboost` | 2 | 29,615 |
| 20 | [`models.md`](models.md) | `src/rade_qnet/models` | 1 | 5,182 |
| 21 | [`models__hybrid_gnn_rnn.md`](models__hybrid_gnn_rnn.md) | `src/rade_qnet/models/hybrid_gnn_rnn` | 8 | 101,948 |
| 22 | [`models__hybrid_gnn_rnn__features.md`](models__hybrid_gnn_rnn__features.md) | `src/rade_qnet/models/hybrid_gnn_rnn/features` | 4 | 55,030 |
| 23 | [`models__hybrid_gnn_rnn__layers.md`](models__hybrid_gnn_rnn__layers.md) | `src/rade_qnet/models/hybrid_gnn_rnn/layers` | 6 | 59,239 |
| 24 | [`models__hybrid_gnn_rnn__pipelines.md`](models__hybrid_gnn_rnn__pipelines.md) | `src/rade_qnet/models/hybrid_gnn_rnn/pipelines` | 4 | 24,917 |
| 25 | [`models__lstm_tabular.md`](models__lstm_tabular.md) | `src/rade_qnet/models/lstm_tabular` | 5 | 14,758 |
| 26 | [`models__ridge.md`](models__ridge.md) | `src/rade_qnet/models/ridge` | 5 | 10,771 |
| 27 | [`models__xgb_tabular.md`](models__xgb_tabular.md) | `src/rade_qnet/models/xgb_tabular` | 5 | 11,131 |
| 28 | [`orchestration.md`](orchestration.md) | `src/rade_qnet/orchestration` | 2 | 18,572 |
| 29 | [`orchestration__compute.md`](orchestration__compute.md) | `src/rade_qnet/orchestration/compute` | 6 | 48,408 |
| 30 | [`orchestration__jobs.md`](orchestration__jobs.md) | `src/rade_qnet/orchestration/jobs` | 6 | 58,453 |
| 31 | [`orchestration__pipelines.md`](orchestration__pipelines.md) | `src/rade_qnet/orchestration/pipelines` | 6 | 103,198 |
| 32 | [`orchestration__stages.md`](orchestration__stages.md) | `src/rade_qnet/orchestration/stages` | 5 | 46,897 |
| 33 | [`sources.md`](sources.md) | `src/rade_qnet/sources` | 1 | 1,247 |
| 34 | [`sources__batching.md`](sources__batching.md) | `src/rade_qnet/sources/batching` | 3 | 34,972 |
| 35 | [`sources__dataset.md`](sources__dataset.md) | `src/rade_qnet/sources/dataset` | 7 | 97,065 |
| 36 | [`sources__dataset__transforms.md`](sources__dataset__transforms.md) | `src/rade_qnet/sources/dataset/transforms` | 6 | 69,429 |
| 37 | [`sources__environment.md`](sources__environment.md) | `src/rade_qnet/sources/environment` | 2 | 9,938 |
| 38 | [`storage.md`](storage.md) | `src/rade_qnet/storage` | 4 | 36,396 |
| 39 | [`storage__runs.md`](storage__runs.md) | `src/rade_qnet/storage/runs` | 4 | 56,728 |
| 40 | [`testkit.md`](testkit.md) | `src/rade_qnet/testkit` | 4 | 126,775 |

## Documentation: 2 documents, 14 files, 389,291 bytes

The prose, including `ARCHITECTURE.md`. These are already markdown and could be fetched directly, but they are carried here so they land in the manifest: a truncated paste then shows up as a digest mismatch rather than as a puzzling test failure. Four tests read these files and check the examples in them still parse, so the suite needs them present at these exact paths.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 41 | [`docs.md`](docs.md) | `src/rade_qnet/docs` | 6 | 179,171 |
| 42 | [`docs__phases.md`](docs__phases.md) | `src/rade_qnet/docs/phases` | 8 | 210,120 |

## Tests: 39 documents, 172 files, 1,492,755 bytes

The suite. Rebuild it after the source and run `pytest tests/rade_qnet` -- that run is what turns a pasted tree into a verified one. Each document's name mirrors the source document it exercises: `tests__core__spec.md` tests `core__spec.md`.

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 43 | [`tests.md`](tests.md) | `tests/rade_qnet` | 7 | 75,009 |
| 44 | [`tests__analysis.md`](tests__analysis.md) | `tests/rade_qnet/analysis` | 1 | 368 |
| 45 | [`tests__analysis__metrics.md`](tests__analysis__metrics.md) | `tests/rade_qnet/analysis/metrics` | 4 | 37,693 |
| 46 | [`tests__analysis__reports.md`](tests__analysis__reports.md) | `tests/rade_qnet/analysis/reports` | 6 | 56,538 |
| 47 | [`tests__analysis__visuals.md`](tests__analysis__visuals.md) | `tests/rade_qnet/analysis/visuals` | 9 | 79,935 |
| 48 | [`tests__core.md`](tests__core.md) | `tests/rade_qnet/core` | 1 | 359 |
| 49 | [`tests__core__authoring.md`](tests__core__authoring.md) | `tests/rade_qnet/core/authoring` | 5 | 34,192 |
| 50 | [`tests__core__contract.md`](tests__core__contract.md) | `tests/rade_qnet/core/contract` | 8 | 71,377 |
| 51 | [`tests__core__lifecycle.md`](tests__core__lifecycle.md) | `tests/rade_qnet/core/lifecycle` | 7 | 47,210 |
| 52 | [`tests__core__provenance.md`](tests__core__provenance.md) | `tests/rade_qnet/core/provenance` | 4 | 25,253 |
| 53 | [`tests__core__spec.md`](tests__core__spec.md) | `tests/rade_qnet/core/spec` | 8 | 70,869 |
| 54 | [`tests__engines.md`](tests__engines.md) | `tests/rade_qnet/engines` | 4 | 28,041 |
| 55 | [`tests__engines__sklearn.md`](tests__engines__sklearn.md) | `tests/rade_qnet/engines/sklearn` | 2 | 10,768 |
| 56 | [`tests__engines__torch.md`](tests__engines__torch.md) | `tests/rade_qnet/engines/torch` | 4 | 41,380 |
| 57 | [`tests__engines__torch__hardware.md`](tests__engines__torch__hardware.md) | `tests/rade_qnet/engines/torch/hardware` | 4 | 26,967 |
| 58 | [`tests__engines__torch__learners.md`](tests__engines__torch__learners.md) | `tests/rade_qnet/engines/torch/learners` | 3 | 25,229 |
| 59 | [`tests__engines__torch__training.md`](tests__engines__torch__training.md) | `tests/rade_qnet/engines/torch/training` | 5 | 66,874 |
| 60 | [`tests__engines__xgboost.md`](tests__engines__xgboost.md) | `tests/rade_qnet/engines/xgboost` | 2 | 16,521 |
| 61 | [`tests__models.md`](tests__models.md) | `tests/rade_qnet/models` | 3 | 39,386 |
| 62 | [`tests__models__hybrid_gnn_rnn.md`](tests__models__hybrid_gnn_rnn.md) | `tests/rade_qnet/models/hybrid_gnn_rnn` | 9 | 76,879 |
| 63 | [`tests__models__hybrid_gnn_rnn__features.md`](tests__models__hybrid_gnn_rnn__features.md) | `tests/rade_qnet/models/hybrid_gnn_rnn/features` | 5 | 38,729 |
| 64 | [`tests__models__hybrid_gnn_rnn__layers.md`](tests__models__hybrid_gnn_rnn__layers.md) | `tests/rade_qnet/models/hybrid_gnn_rnn/layers` | 7 | 42,782 |
| 65 | [`tests__models__hybrid_gnn_rnn__pipelines.md`](tests__models__hybrid_gnn_rnn__pipelines.md) | `tests/rade_qnet/models/hybrid_gnn_rnn/pipelines` | 4 | 21,283 |
| 66 | [`tests__models__lstm_tabular.md`](tests__models__lstm_tabular.md) | `tests/rade_qnet/models/lstm_tabular` | 2 | 4,180 |
| 67 | [`tests__models__ridge.md`](tests__models__ridge.md) | `tests/rade_qnet/models/ridge` | 3 | 6,141 |
| 68 | [`tests__models__xgb_tabular.md`](tests__models__xgb_tabular.md) | `tests/rade_qnet/models/xgb_tabular` | 2 | 3,332 |
| 69 | [`tests__orchestration.md`](tests__orchestration.md) | `tests/rade_qnet/orchestration` | 2 | 15,861 |
| 70 | [`tests__orchestration__compute.md`](tests__orchestration__compute.md) | `tests/rade_qnet/orchestration/compute` | 4 | 34,688 |
| 71 | [`tests__orchestration__jobs.md`](tests__orchestration__jobs.md) | `tests/rade_qnet/orchestration/jobs` | 8 | 75,579 |
| 72 | [`tests__orchestration__pipelines.md`](tests__orchestration__pipelines.md) | `tests/rade_qnet/orchestration/pipelines` | 7 | 117,649 |
| 73 | [`tests__orchestration__stages.md`](tests__orchestration__stages.md) | `tests/rade_qnet/orchestration/stages` | 2 | 5,469 |
| 74 | [`tests__sources.md`](tests__sources.md) | `tests/rade_qnet/sources` | 1 | 356 |
| 75 | [`tests__sources__batching.md`](tests__sources__batching.md) | `tests/rade_qnet/sources/batching` | 3 | 31,712 |
| 76 | [`tests__sources__dataset.md`](tests__sources__dataset.md) | `tests/rade_qnet/sources/dataset` | 5 | 51,674 |
| 77 | [`tests__sources__dataset__transforms.md`](tests__sources__dataset__transforms.md) | `tests/rade_qnet/sources/dataset/transforms` | 6 | 49,484 |
| 78 | [`tests__sources__environment.md`](tests__sources__environment.md) | `tests/rade_qnet/sources/environment` | 2 | 6,385 |
| 79 | [`tests__storage.md`](tests__storage.md) | `tests/rade_qnet/storage` | 4 | 36,849 |
| 80 | [`tests__storage__runs.md`](tests__storage__runs.md) | `tests/rade_qnet/storage/runs` | 4 | 44,979 |
| 81 | [`tests__testkit.md`](tests__testkit.md) | `tests/rade_qnet/testkit` | 5 | 74,775 |

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
fba21c243bfdeb98  pyproject.toml
f0aa795153b0a74e  src/rade_qnet/__init__.py
97da3806b2c8875e  src/rade_qnet/api.py
f84c1e95253049a7  src/rade_qnet/ruff.toml
c610c7393d3bee5b  src/rade_qnet/analysis/__init__.py
6bc47bdcc1b578ae  src/rade_qnet/analysis/metrics/__init__.py
aba29d7c39dcf912  src/rade_qnet/analysis/metrics/drift.py
24c242fef6bec8fe  src/rade_qnet/analysis/metrics/quality.py
8d183bb0c63180c1  src/rade_qnet/analysis/metrics/regression.py
67191a456d8c17a9  src/rade_qnet/analysis/reports/__init__.py
700a581a6ca0be3b  src/rade_qnet/analysis/reports/base.py
1557a6382e2340f6  src/rade_qnet/analysis/reports/baselines.py
dc286333a6cdc775  src/rade_qnet/analysis/reports/curves.py
233491a91e32745d  src/rade_qnet/analysis/reports/quality.py
7632b593c2015988  src/rade_qnet/analysis/reports/summary.py
c99bfeb943fc679d  src/rade_qnet/analysis/visuals/__init__.py
f707b3b1534e74a7  src/rade_qnet/analysis/visuals/data.py
e527baf067f10b00  src/rade_qnet/analysis/visuals/evaluation.py
99b281603eeebcac  src/rade_qnet/analysis/visuals/export.py
63e4127807280361  src/rade_qnet/analysis/visuals/figures.py
436a49c80ae60a51  src/rade_qnet/analysis/visuals/jobset.py
f6b3228dede4416b  src/rade_qnet/analysis/visuals/style.py
55ea82a6c0e8096a  src/rade_qnet/analysis/visuals/training.py
98bdfe633894c955  src/rade_qnet/analysis/visuals/tuning.py
93489b048e7a52fd  src/rade_qnet/core/__init__.py
da89724859285c14  src/rade_qnet/core/authoring/__init__.py
438101cea82e2646  src/rade_qnet/core/authoring/capabilities.py
4c391938e3356d06  src/rade_qnet/core/authoring/definition.py
9bfa261bc0bb07fa  src/rade_qnet/core/authoring/policy.py
453ad8a21e99e6bd  src/rade_qnet/core/authoring/supervised.py
03e5e8f82e62f796  src/rade_qnet/core/contract/__init__.py
a1b63e99a6b56f4e  src/rade_qnet/core/contract/base.py
c81d47d6593c6bfb  src/rade_qnet/core/contract/bundle.py
73648a6e604ea98b  src/rade_qnet/core/contract/data.py
4dbd820c64ef1ce7  src/rade_qnet/core/contract/requirement.py
2d2f2ea95b63cfb9  src/rade_qnet/core/contract/result.py
64516ecd4c330424  src/rade_qnet/core/contract/signature.py
0f58535c7104056f  src/rade_qnet/core/contract/source.py
c9795030254a3196  src/rade_qnet/core/contract/state.py
104b6ad947ef13b3  src/rade_qnet/core/lifecycle/__init__.py
5c564d38da6f6864  src/rade_qnet/core/lifecycle/components.py
4626d883ff35448d  src/rade_qnet/core/lifecycle/context.py
f13eb08c3452f71f  src/rade_qnet/core/lifecycle/errors.py
b9db3f32f535c7ad  src/rade_qnet/core/lifecycle/hooks.py
9e58f9499eb4a1e1  src/rade_qnet/core/lifecycle/pipeline.py
131cf36657c86a5d  src/rade_qnet/core/lifecycle/registry.py
b556b6e4ac00495d  src/rade_qnet/core/provenance/__init__.py
f6d9f4a484edae26  src/rade_qnet/core/provenance/hashing.py
f107778d8a86e24c  src/rade_qnet/core/provenance/logging.py
030643171325087b  src/rade_qnet/core/provenance/seeding.py
8d034c0a484bd379  src/rade_qnet/core/spec/__init__.py
0b868e88f5dd798a  src/rade_qnet/core/spec/base.py
7ed9e6228f3c4454  src/rade_qnet/core/spec/data.py
4386834de64e33d2  src/rade_qnet/core/spec/hardware.py
d4cb787f098fb7e0  src/rade_qnet/core/spec/jobs.py
89eeb600e0a48ae8  src/rade_qnet/core/spec/merge.py
c34d4853629d6b2d  src/rade_qnet/core/spec/reports.py
dd12485b60f37568  src/rade_qnet/core/spec/run.py
d652ff166c195ee9  src/rade_qnet/core/spec/training.py
c8be6dcfed974aa9  src/rade_qnet/core/spec/tune.py
93c9685349d6ad01  src/rade_qnet/engines/__init__.py
373cda51c8e5f5b2  src/rade_qnet/engines/base.py
1e5d33305a6210f6  src/rade_qnet/engines/loaders.py
9687c52982db50ec  src/rade_qnet/engines/sklearn/__init__.py
3da30705e4f82de9  src/rade_qnet/engines/sklearn/engine.py
67bf31fb1ffd7356  src/rade_qnet/engines/torch/__init__.py
7f8c07ab11295508  src/rade_qnet/engines/torch/engine.py
3da70099294e818b  src/rade_qnet/engines/torch/loaders.py
6947dc425a1e454a  src/rade_qnet/engines/torch/materialise.py
d3bbea51d11a2eef  src/rade_qnet/engines/torch/predictor.py
0f5c6d3c17c4d67f  src/rade_qnet/engines/torch/hardware/__init__.py
0e3900bc3906f0c3  src/rade_qnet/engines/torch/hardware/determinism.py
f5fbc3699eba7e1b  src/rade_qnet/engines/torch/hardware/devices.py
9addadb978642a89  src/rade_qnet/engines/torch/hardware/distributed.py
78173434fb32ab47  src/rade_qnet/engines/torch/learners/__init__.py
ef1f1aa9f834ed2b  src/rade_qnet/engines/torch/learners/random.py
7cccaac0f285b296  src/rade_qnet/engines/torch/learners/supervised.py
1513f7834fd4c319  src/rade_qnet/engines/torch/training/__init__.py
4f364430b3ee327c  src/rade_qnet/engines/torch/training/callbacks.py
df134fb5e194563f  src/rade_qnet/engines/torch/training/checkpoint.py
4cb0b2dc45846113  src/rade_qnet/engines/torch/training/loops.py
e2df8fc6f1313596  src/rade_qnet/engines/torch/training/losses.py
d6ff5a7febff9ffa  src/rade_qnet/engines/xgboost/__init__.py
4e410e412fdc4a82  src/rade_qnet/engines/xgboost/engine.py
40e2314b287e5a4a  src/rade_qnet/models/__init__.py
d99cf28ec67f4ec0  src/rade_qnet/models/hybrid_gnn_rnn/__init__.py
73f29f102ea6daaa  src/rade_qnet/models/hybrid_gnn_rnn/data.py
7d269cda90d695c5  src/rade_qnet/models/hybrid_gnn_rnn/model.py
589cfb05e6a8b23e  src/rade_qnet/models/hybrid_gnn_rnn/register.py
5308b7d914f3b947  src/rade_qnet/models/hybrid_gnn_rnn/reports.py
a177505b14867755  src/rade_qnet/models/hybrid_gnn_rnn/spec.py
3cfb158385bbfb3f  src/rade_qnet/models/hybrid_gnn_rnn/state.py
c033d0fcfc405089  src/rade_qnet/models/hybrid_gnn_rnn/visuals.py
011ecbecf59e5538  src/rade_qnet/models/hybrid_gnn_rnn/features/__init__.py
f0991d6c6cf24d24  src/rade_qnet/models/hybrid_gnn_rnn/features/basis.py
b1f1f4ff533ab65f  src/rade_qnet/models/hybrid_gnn_rnn/features/encoder.py
53720982a6bf987e  src/rade_qnet/models/hybrid_gnn_rnn/features/graph.py
f2731bbc54fbfd3c  src/rade_qnet/models/hybrid_gnn_rnn/layers/__init__.py
8965de3a68e489f0  src/rade_qnet/models/hybrid_gnn_rnn/layers/attention.py
f48de1d670a94a44  src/rade_qnet/models/hybrid_gnn_rnn/layers/fusion.py
7b1e2fec9b208fe6  src/rade_qnet/models/hybrid_gnn_rnn/layers/gnn.py
40485abc1ee3cd42  src/rade_qnet/models/hybrid_gnn_rnn/layers/projection.py
f69273be0c78e06d  src/rade_qnet/models/hybrid_gnn_rnn/layers/rnn.py
2ee06142a0e2a995  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/__init__.py
ec3525c753752a11  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/eval.py
5e713339832a8ff7  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/train.py
c6982aae644e43c9  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/tune.py
8ed63db8039871d8  src/rade_qnet/models/lstm_tabular/__init__.py
78bb98d6c885d8aa  src/rade_qnet/models/lstm_tabular/data.py
2db3a3716d364c54  src/rade_qnet/models/lstm_tabular/model.py
1e29ed34e6a56eb1  src/rade_qnet/models/lstm_tabular/register.py
684004c8b0263ebe  src/rade_qnet/models/lstm_tabular/spec.py
8e878ba14afd847c  src/rade_qnet/models/ridge/__init__.py
04a97203d1724ccf  src/rade_qnet/models/ridge/data.py
86653e260ade081f  src/rade_qnet/models/ridge/model.py
5ec21b14fc34eabe  src/rade_qnet/models/ridge/register.py
968365b42f941d08  src/rade_qnet/models/ridge/spec.py
8a5273793dc9fa45  src/rade_qnet/models/xgb_tabular/__init__.py
cf2983dc854e17c0  src/rade_qnet/models/xgb_tabular/data.py
c7cf7a7b0632cefd  src/rade_qnet/models/xgb_tabular/model.py
292e398c993f0db6  src/rade_qnet/models/xgb_tabular/register.py
446633bfa7a1d14b  src/rade_qnet/models/xgb_tabular/spec.py
41743506e557df79  src/rade_qnet/orchestration/__init__.py
c2ee35c80b1a71f6  src/rade_qnet/orchestration/serving.py
e51e01a50744d04c  src/rade_qnet/orchestration/compute/__init__.py
4368fcdf46c5da38  src/rade_qnet/orchestration/compute/base.py
518ba800f9e96a02  src/rade_qnet/orchestration/compute/gpus.py
68231f2a0dd188a5  src/rade_qnet/orchestration/compute/local.py
108efe6b640a3a0d  src/rade_qnet/orchestration/compute/placement.py
3b073ef3c554bdce  src/rade_qnet/orchestration/compute/processes.py
38c0c4fe2f3800a8  src/rade_qnet/orchestration/jobs/__init__.py
9455837cb8ba0832  src/rade_qnet/orchestration/jobs/fanout.py
40c05ee56911e539  src/rade_qnet/orchestration/jobs/groups.py
80e4c231e0368046  src/rade_qnet/orchestration/jobs/manifest.py
6482b8dd9fd29aa8  src/rade_qnet/orchestration/jobs/set.py
4b4447c3711fbf4c  src/rade_qnet/orchestration/jobs/unit.py
67aa54d4ce6a7acf  src/rade_qnet/orchestration/pipelines/__init__.py
f9f065ad810b76dc  src/rade_qnet/orchestration/pipelines/evaluate.py
85df0b89051a7a26  src/rade_qnet/orchestration/pipelines/infer.py
96b510ee66d5fd62  src/rade_qnet/orchestration/pipelines/reinforce.py
f61cbb9d86175cd0  src/rade_qnet/orchestration/pipelines/train.py
dd0d38b33938ebe7  src/rade_qnet/orchestration/pipelines/tune.py
d4555a56cb073632  src/rade_qnet/orchestration/stages/__init__.py
fd1d9434a83b8a95  src/rade_qnet/orchestration/stages/reload.py
52a0bf45099efed5  src/rade_qnet/orchestration/stages/resolve.py
2c240ae5ec2ff681  src/rade_qnet/orchestration/stages/scoring.py
f3ffe459960cd03e  src/rade_qnet/orchestration/stages/search.py
0933cff34ac5afc2  src/rade_qnet/sources/__init__.py
705aee492f36f269  src/rade_qnet/sources/batching/__init__.py
e9bf1a479fecbcb8  src/rade_qnet/sources/batching/dataset.py
a23e2fda7fa4f1bb  src/rade_qnet/sources/batching/rollout.py
bf7f3fb0f0814d67  src/rade_qnet/sources/dataset/__init__.py
a0eab4ed54f687d8  src/rade_qnet/sources/dataset/cache.py
1336d3633e002262  src/rade_qnet/sources/dataset/module.py
aa447d0e624c00f6  src/rade_qnet/sources/dataset/rebuild.py
bcb1ada5e215aa5b  src/rade_qnet/sources/dataset/splits.py
4aecedb17e4a109a  src/rade_qnet/sources/dataset/tables.py
329f5b278aac36a3  src/rade_qnet/sources/dataset/tabular.py
6d3de29fb11b2a0c  src/rade_qnet/sources/dataset/transforms/__init__.py
f47510e9282e3cc7  src/rade_qnet/sources/dataset/transforms/composite.py
904b518e82471413  src/rade_qnet/sources/dataset/transforms/encoding.py
b938b45b7953e380  src/rade_qnet/sources/dataset/transforms/reduction.py
a1932b200ec1be9d  src/rade_qnet/sources/dataset/transforms/scaling.py
f6955b509b3329f8  src/rade_qnet/sources/dataset/transforms/sequence.py
8d8eb3deaf1ea40f  src/rade_qnet/sources/environment/__init__.py
b6c88ab23fb9cf73  src/rade_qnet/sources/environment/protocol.py
5ba06be2a276cadf  src/rade_qnet/storage/__init__.py
fcaf136ba9d57f84  src/rade_qnet/storage/bundle.py
a6dcb2a486d49d91  src/rade_qnet/storage/locking.py
593c50e7d6217c29  src/rade_qnet/storage/manifest.py
48dcf2a1299e07de  src/rade_qnet/storage/runs/__init__.py
9c922f96d5c8cd98  src/rade_qnet/storage/runs/catalog.py
2a91c5e65bf839b9  src/rade_qnet/storage/runs/registry.py
d7c5799fb1f03913  src/rade_qnet/storage/runs/tracker.py
be147ade92895e66  src/rade_qnet/testkit/__init__.py
f34d858b31058be2  src/rade_qnet/testkit/conformance.py
a10dd886a237dbfa  src/rade_qnet/testkit/fixtures.py
366dcea95f2cd3ac  src/rade_qnet/testkit/parity.py
7abe69d46e761658  src/rade_qnet/docs/ARCHITECTURE.md
ec30f7f0c0938f7a  src/rade_qnet/docs/CODING_STANDARDS.md
3d856f8f56c6f0ee  src/rade_qnet/docs/GUIDE.md
457874a795d914f9  src/rade_qnet/docs/IMPLEMENTATION.md
a82b8b93fd0d7a64  src/rade_qnet/docs/MODEL_IMPLEMENTATION.md
26205bc095821e81  src/rade_qnet/docs/README.md
f83b21a468bc96b2  src/rade_qnet/docs/phases/PHASE_0_BASELINE.md
fa59c7242c850390  src/rade_qnet/docs/phases/PHASE_1_CORE.md
5a4d2dcc551f2911  src/rade_qnet/docs/phases/PHASE_2_TORCH_ENGINE.md
5c38c8ee1e87f928  src/rade_qnet/docs/phases/PHASE_3_HYBRID_GNN_RNN.md
dda53aadcb5305e0  src/rade_qnet/docs/phases/PHASE_4_JOB_SETS.md
f75ec71330030c3b  src/rade_qnet/docs/phases/PHASE_5_EVALUATE_INFER_TUNE.md
a8c4ca3da3c99b0c  src/rade_qnet/docs/phases/PHASE_6_ADDITIONAL_ENGINES.md
24dfc1d96d0e9ffe  src/rade_qnet/docs/phases/PHASE_7_REINFORCEMENT_LEARNING.md
9b77412d766c78e0  tests/rade_qnet/__init__.py
9552f767b1faaf66  tests/rade_qnet/conftest.py
4f63ed66941eacb9  tests/rade_qnet/ruff.toml
1a5abd1e0ce7b27a  tests/rade_qnet/test_api.py
82debfd68b1f8421  tests/rade_qnet/test_documentation.py
dbbe2e645c40c432  tests/rade_qnet/test_extensibility.py
36449874bb76d417  tests/rade_qnet/test_scaffold.py
8d2d3c04dfdfe110  tests/rade_qnet/analysis/__init__.py
815690298ea5fb56  tests/rade_qnet/analysis/metrics/__init__.py
894875d5d4e44445  tests/rade_qnet/analysis/metrics/test_metrics_drift.py
1ce48491f16e5376  tests/rade_qnet/analysis/metrics/test_metrics_quality.py
fb38f3bc008c07b8  tests/rade_qnet/analysis/metrics/test_metrics_regression.py
243b3f95c99bd0f3  tests/rade_qnet/analysis/reports/__init__.py
131b98b13897f88f  tests/rade_qnet/analysis/reports/test_reports_base.py
9b12b9557f8e4ea2  tests/rade_qnet/analysis/reports/test_reports_baselines.py
95fe830b23029c76  tests/rade_qnet/analysis/reports/test_reports_curves.py
c4848a2f442042b8  tests/rade_qnet/analysis/reports/test_reports_quality.py
f2f29c7816c8684e  tests/rade_qnet/analysis/reports/test_reports_summary.py
abc60d222b39299f  tests/rade_qnet/analysis/visuals/__init__.py
7a56dedb933736a8  tests/rade_qnet/analysis/visuals/test_visuals_data.py
ccfa5fe9d0da520e  tests/rade_qnet/analysis/visuals/test_visuals_evaluation.py
692334725d62e0f8  tests/rade_qnet/analysis/visuals/test_visuals_export.py
715c04c3860dc524  tests/rade_qnet/analysis/visuals/test_visuals_figures.py
f128864e10da8635  tests/rade_qnet/analysis/visuals/test_visuals_jobset.py
072bcbbeaf080d05  tests/rade_qnet/analysis/visuals/test_visuals_style.py
df167c0059d128ce  tests/rade_qnet/analysis/visuals/test_visuals_training.py
52554c0b2f02024a  tests/rade_qnet/analysis/visuals/test_visuals_tuning.py
a33185e148d60465  tests/rade_qnet/core/__init__.py
5fa0437358e278ef  tests/rade_qnet/core/authoring/__init__.py
0a7e4938fcbfb1b8  tests/rade_qnet/core/authoring/test_authoring_capabilities.py
deb2cc813fc5215f  tests/rade_qnet/core/authoring/test_authoring_definition.py
fe80eb7d05705468  tests/rade_qnet/core/authoring/test_authoring_policy.py
996747fa07aa4b1c  tests/rade_qnet/core/authoring/test_authoring_supervised.py
455461a7d762debd  tests/rade_qnet/core/contract/__init__.py
e7f0fc8dc09c7503  tests/rade_qnet/core/contract/test_contract_bundle.py
73b19d1e89316134  tests/rade_qnet/core/contract/test_contract_data.py
6017b8f7ead178ed  tests/rade_qnet/core/contract/test_contract_requirement.py
7a2525781c1cdef6  tests/rade_qnet/core/contract/test_contract_result.py
2ac93fb372973aef  tests/rade_qnet/core/contract/test_contract_signature.py
d6c4f0f4a3b93cb1  tests/rade_qnet/core/contract/test_contract_source.py
85aeb1274d4d54a1  tests/rade_qnet/core/contract/test_contract_state.py
41150f68fedf6616  tests/rade_qnet/core/lifecycle/__init__.py
7e93e52fcab13c55  tests/rade_qnet/core/lifecycle/test_lifecycle_components.py
c187dac84db6bf6d  tests/rade_qnet/core/lifecycle/test_lifecycle_context.py
58c2af643b46fd7a  tests/rade_qnet/core/lifecycle/test_lifecycle_errors.py
7732a3679d7e16f5  tests/rade_qnet/core/lifecycle/test_lifecycle_hooks.py
d07c5906200a2d10  tests/rade_qnet/core/lifecycle/test_lifecycle_pipeline.py
801deef65d258f62  tests/rade_qnet/core/lifecycle/test_lifecycle_registry.py
17f398962e74a13b  tests/rade_qnet/core/provenance/__init__.py
550397124d3379c4  tests/rade_qnet/core/provenance/test_provenance_hashing.py
0beb7c90ec33c592  tests/rade_qnet/core/provenance/test_provenance_logging.py
164ef7aa60f3bcac  tests/rade_qnet/core/provenance/test_provenance_seeding.py
4cf0eee3c800d080  tests/rade_qnet/core/spec/__init__.py
cf59fa4befee87a3  tests/rade_qnet/core/spec/test_spec_data.py
7beadfe09ad37b73  tests/rade_qnet/core/spec/test_spec_hardware.py
73e15a3dde448b19  tests/rade_qnet/core/spec/test_spec_jobs.py
4f2d2383798d852e  tests/rade_qnet/core/spec/test_spec_merge.py
6abb6cbdee9201df  tests/rade_qnet/core/spec/test_spec_reports.py
950ffd0f989e98d5  tests/rade_qnet/core/spec/test_spec_run.py
0f30bc07b387cda5  tests/rade_qnet/core/spec/test_spec_training.py
17c5d2fc92ec9189  tests/rade_qnet/engines/__init__.py
ccec6580fbb253cf  tests/rade_qnet/engines/test_engine_layout.py
99dda0aeda3cf4ad  tests/rade_qnet/engines/test_engines_base.py
69e2fdf28f2f1d6e  tests/rade_qnet/engines/test_engines_loaders.py
b0a1ae83d8b2a9fc  tests/rade_qnet/engines/sklearn/__init__.py
182cdf432d2207de  tests/rade_qnet/engines/sklearn/test_sklearn_engine.py
ce84b941ee165fc1  tests/rade_qnet/engines/torch/__init__.py
9f1f7da522b13c93  tests/rade_qnet/engines/torch/test_torch_engine.py
0dcf87126c969172  tests/rade_qnet/engines/torch/test_torch_loaders.py
cc197e973028ec2a  tests/rade_qnet/engines/torch/test_torch_materialise.py
7860806c8e2de23b  tests/rade_qnet/engines/torch/hardware/__init__.py
47f5af4fd3eae9b5  tests/rade_qnet/engines/torch/hardware/test_hardware_determinism.py
2ce882982db68491  tests/rade_qnet/engines/torch/hardware/test_hardware_devices.py
217493ce9169d482  tests/rade_qnet/engines/torch/hardware/test_hardware_distributed.py
347c5ed4a780e803  tests/rade_qnet/engines/torch/learners/__init__.py
65cfc15b18ea7679  tests/rade_qnet/engines/torch/learners/test_learners_random.py
4304123ce66061d2  tests/rade_qnet/engines/torch/learners/test_learners_supervised.py
ea056fce628eee7f  tests/rade_qnet/engines/torch/training/__init__.py
97c783702a4278fc  tests/rade_qnet/engines/torch/training/test_training_callbacks.py
07ea0986a0ceb441  tests/rade_qnet/engines/torch/training/test_training_checkpoint.py
eee06920985be093  tests/rade_qnet/engines/torch/training/test_training_loops.py
ad6de8c53529ae7d  tests/rade_qnet/engines/torch/training/test_training_losses.py
d35aed8beca12275  tests/rade_qnet/engines/xgboost/__init__.py
7cfb06ea22f62526  tests/rade_qnet/engines/xgboost/test_xgboost_engine.py
e11173949276a7e1  tests/rade_qnet/models/__init__.py
3a1a074e153fe65b  tests/rade_qnet/models/test_model_layout.py
804c5e8b4c66f0c6  tests/rade_qnet/models/test_reference_models.py
5383ca55a0c3650b  tests/rade_qnet/models/hybrid_gnn_rnn/__init__.py
33266791ef4ea6e6  tests/rade_qnet/models/hybrid_gnn_rnn/test_data.py
c57796835c402eda  tests/rade_qnet/models/hybrid_gnn_rnn/test_model.py
daef5e3ddff87425  tests/rade_qnet/models/hybrid_gnn_rnn/test_parity.py
9342b6e363699dd7  tests/rade_qnet/models/hybrid_gnn_rnn/test_register.py
294a0163bea578b4  tests/rade_qnet/models/hybrid_gnn_rnn/test_reports.py
2e526e1e4d6ddd1f  tests/rade_qnet/models/hybrid_gnn_rnn/test_state.py
fb508cf76c889e00  tests/rade_qnet/models/hybrid_gnn_rnn/test_universe.py
899e3486d7ebd733  tests/rade_qnet/models/hybrid_gnn_rnn/test_visuals.py
7cd332b12fc0f4a5  tests/rade_qnet/models/hybrid_gnn_rnn/features/__init__.py
c03a6d739771af29  tests/rade_qnet/models/hybrid_gnn_rnn/features/conftest.py
0b76004077b2312f  tests/rade_qnet/models/hybrid_gnn_rnn/features/test_basis.py
1d82585a0612a1bf  tests/rade_qnet/models/hybrid_gnn_rnn/features/test_encoder.py
a4dd3f259a1f2a62  tests/rade_qnet/models/hybrid_gnn_rnn/features/test_graph.py
74ed9a32e29b7cd6  tests/rade_qnet/models/hybrid_gnn_rnn/layers/__init__.py
22199b76430ed4c2  tests/rade_qnet/models/hybrid_gnn_rnn/layers/conftest.py
e37200c730a5a76c  tests/rade_qnet/models/hybrid_gnn_rnn/layers/test_attention.py
f62954d39045e6fb  tests/rade_qnet/models/hybrid_gnn_rnn/layers/test_fusion.py
db86619a1b40adb7  tests/rade_qnet/models/hybrid_gnn_rnn/layers/test_gnn.py
1447669b64965f56  tests/rade_qnet/models/hybrid_gnn_rnn/layers/test_projection.py
b3fb84a31c2521dc  tests/rade_qnet/models/hybrid_gnn_rnn/layers/test_rnn.py
460b7d1c67ae53f9  tests/rade_qnet/models/hybrid_gnn_rnn/pipelines/__init__.py
387ee1e438d8a501  tests/rade_qnet/models/hybrid_gnn_rnn/pipelines/test_pipelines_eval.py
4c34861fc5eedc96  tests/rade_qnet/models/hybrid_gnn_rnn/pipelines/test_pipelines_train.py
e1f2bf7610d37101  tests/rade_qnet/models/hybrid_gnn_rnn/pipelines/test_pipelines_tune.py
e19dbdeabb5b7af6  tests/rade_qnet/models/lstm_tabular/__init__.py
d020a24094b900d1  tests/rade_qnet/models/lstm_tabular/test_model.py
f92a7a31ce5f7c72  tests/rade_qnet/models/ridge/__init__.py
b4bbdc31a30aaafd  tests/rade_qnet/models/ridge/test_model.py
ce59e428145e17ed  tests/rade_qnet/models/ridge/test_register.py
dfae360f563b4a2f  tests/rade_qnet/models/xgb_tabular/__init__.py
bd478f1286a8a145  tests/rade_qnet/models/xgb_tabular/test_model.py
af8945dcb1b57f28  tests/rade_qnet/orchestration/__init__.py
0e78199808bff149  tests/rade_qnet/orchestration/test_orchestration_serving.py
31dd72a61b33b7be  tests/rade_qnet/orchestration/compute/__init__.py
61decdea64072ade  tests/rade_qnet/orchestration/compute/test_compute_executors.py
1200246b161141ec  tests/rade_qnet/orchestration/compute/test_compute_placement.py
84a957b1c982a938  tests/rade_qnet/orchestration/compute/workers.py
db64606cfe41ff84  tests/rade_qnet/orchestration/jobs/__init__.py
57bc8a00c87d9c9f  tests/rade_qnet/orchestration/jobs/support.py
1bb24859e5431a8f  tests/rade_qnet/orchestration/jobs/test_jobs_fanout.py
caf6e448ee6e970d  tests/rade_qnet/orchestration/jobs/test_jobs_groups.py
98dc6b714ca46aec  tests/rade_qnet/orchestration/jobs/test_jobs_manifest.py
60545903949860a3  tests/rade_qnet/orchestration/jobs/test_jobs_parity.py
e62c9a827ce52b21  tests/rade_qnet/orchestration/jobs/test_jobs_set.py
e3cb40ca51ccddfe  tests/rade_qnet/orchestration/jobs/test_jobs_unit.py
d46d1a478e03562f  tests/rade_qnet/orchestration/pipelines/__init__.py
f560486697b8c17f  tests/rade_qnet/orchestration/pipelines/support.py
cd32177ccc2ce794  tests/rade_qnet/orchestration/pipelines/test_pipelines_evaluate.py
28ac163fb7ba64a5  tests/rade_qnet/orchestration/pipelines/test_pipelines_infer.py
f94d426ee710e0c8  tests/rade_qnet/orchestration/pipelines/test_pipelines_reinforce.py
e037ad1947a9cd5b  tests/rade_qnet/orchestration/pipelines/test_pipelines_train.py
554c80819471773d  tests/rade_qnet/orchestration/pipelines/test_pipelines_tune.py
861fa175ff9e24b5  tests/rade_qnet/orchestration/stages/__init__.py
9dd2fb4dcaea65dd  tests/rade_qnet/orchestration/stages/test_stages_resolve.py
edc2c57fa62a87d4  tests/rade_qnet/sources/__init__.py
75d38714d92fee51  tests/rade_qnet/sources/batching/__init__.py
cf9c39982bb47805  tests/rade_qnet/sources/batching/test_batching_dataset.py
c78fa44a1e733357  tests/rade_qnet/sources/batching/test_batching_rollout.py
aca25b3ce8fb419d  tests/rade_qnet/sources/dataset/__init__.py
219df6062c0637eb  tests/rade_qnet/sources/dataset/test_dataset_cache.py
6a34cddc7ef22200  tests/rade_qnet/sources/dataset/test_dataset_module.py
34d44469c31c40c1  tests/rade_qnet/sources/dataset/test_dataset_splits.py
4645567f20904486  tests/rade_qnet/sources/dataset/test_dataset_tables.py
2d244a830faebd42  tests/rade_qnet/sources/dataset/transforms/__init__.py
12a48b6a3e946c2a  tests/rade_qnet/sources/dataset/transforms/test_transforms_composite.py
12982d53c655169c  tests/rade_qnet/sources/dataset/transforms/test_transforms_encoding.py
a4c587d5962eeb86  tests/rade_qnet/sources/dataset/transforms/test_transforms_reduction.py
65d7e3db21ea29f0  tests/rade_qnet/sources/dataset/transforms/test_transforms_scaling.py
8dbf39b6809d2186  tests/rade_qnet/sources/dataset/transforms/test_transforms_sequence.py
ff225df9a8ee5c59  tests/rade_qnet/sources/environment/__init__.py
92c969f3d0c70188  tests/rade_qnet/sources/environment/test_environment_protocol.py
3ccdb89a57a83475  tests/rade_qnet/storage/__init__.py
235c3977d0ba70ca  tests/rade_qnet/storage/test_storage_bundle.py
d2ca71c71c19d047  tests/rade_qnet/storage/test_storage_locking.py
b655369174dc00c0  tests/rade_qnet/storage/test_storage_manifest.py
3c6b979e4a3d6f08  tests/rade_qnet/storage/runs/__init__.py
973463f5d4c635b2  tests/rade_qnet/storage/runs/test_runs_catalog.py
25025909f204c9e9  tests/rade_qnet/storage/runs/test_runs_registry.py
259a9fe4ee5c9963  tests/rade_qnet/storage/runs/test_runs_tracker.py
4dd05188c31c1115  tests/rade_qnet/testkit/__init__.py
e2871cbd5e4fa5f0  tests/rade_qnet/testkit/test_testkit_conformance.py
aa867e4786c01c80  tests/rade_qnet/testkit/test_testkit_fixtures.py
640a0603e61fb4ec  tests/rade_qnet/testkit/test_testkit_golden_fixture.py
081eb6c29a54e0ff  tests/rade_qnet/testkit/test_testkit_parity.py
```

