# Porting `rade_qnet` through a markdown-only proxy

40 documents, 177 files, 50,294 lines, 1,791,267 bytes.

Each document below covers one directory. Work down the list in order: a parent directory always appears before its children, so the tree is importable at every step.

## What is not here

`src/rade_qnet/docs/` is already markdown, so it crosses the proxy unchanged — fetch those files directly rather than through this set. `pyproject.toml` is included, as the first document, so the rebuilt tree installs with `pip install -e .`. The test suite, fixtures and examples are excluded by scope; the golden parity fixtures under `tests/fixtures/rade_qnet/` are binary `.npy` files and cannot travel as text at all.

## Directories, in creation order

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 1 | [`_repository.md`](_repository.md) | `.` | 1 | 5,974 |
| 2 | [`_root.md`](_root.md) | `src/rade_qnet` | 3 | 34,521 |
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
| 28 | [`orchestration.md`](orchestration.md) | `src/rade_qnet/orchestration` | 1 | 1,227 |
| 29 | [`orchestration__compute.md`](orchestration__compute.md) | `src/rade_qnet/orchestration/compute` | 6 | 48,408 |
| 30 | [`orchestration__jobs.md`](orchestration__jobs.md) | `src/rade_qnet/orchestration/jobs` | 6 | 58,453 |
| 31 | [`orchestration__pipelines.md`](orchestration__pipelines.md) | `src/rade_qnet/orchestration/pipelines` | 6 | 102,254 |
| 32 | [`orchestration__stages.md`](orchestration__stages.md) | `src/rade_qnet/orchestration/stages` | 5 | 40,907 |
| 33 | [`sources.md`](sources.md) | `src/rade_qnet/sources` | 1 | 1,247 |
| 34 | [`sources__batching.md`](sources__batching.md) | `src/rade_qnet/sources/batching` | 3 | 34,972 |
| 35 | [`sources__dataset.md`](sources__dataset.md) | `src/rade_qnet/sources/dataset` | 7 | 97,065 |
| 36 | [`sources__dataset__transforms.md`](sources__dataset__transforms.md) | `src/rade_qnet/sources/dataset/transforms` | 6 | 69,429 |
| 37 | [`sources__environment.md`](sources__environment.md) | `src/rade_qnet/sources/environment` | 2 | 9,938 |
| 38 | [`storage.md`](storage.md) | `src/rade_qnet/storage` | 4 | 36,396 |
| 39 | [`storage__runs.md`](storage__runs.md) | `src/rade_qnet/storage/runs` | 4 | 56,728 |
| 40 | [`testkit.md`](testkit.md) | `src/rade_qnet/testkit` | 4 | 126,775 |

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
561f825b25dd9412  src/rade_qnet/api.py
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
f53febd64a1946b7  src/rade_qnet/orchestration/pipelines/infer.py
96b510ee66d5fd62  src/rade_qnet/orchestration/pipelines/reinforce.py
f61cbb9d86175cd0  src/rade_qnet/orchestration/pipelines/train.py
dd0d38b33938ebe7  src/rade_qnet/orchestration/pipelines/tune.py
d4555a56cb073632  src/rade_qnet/orchestration/stages/__init__.py
625a003bdd93daa9  src/rade_qnet/orchestration/stages/reload.py
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
```

