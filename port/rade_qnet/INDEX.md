# Porting `rade_qnet` through a markdown-only proxy

36 documents, 165 files, 46,780 lines, 1,651,733 bytes.

Each document below covers one directory. Work down the list in order: a parent directory always appears before its children, so the tree is importable at every step.

## What is not here

`src/rade_qnet/docs/` is already markdown, so it crosses the proxy unchanged — fetch those files directly rather than through this set. `pyproject.toml` is included, as the first document, so the rebuilt tree installs with `pip install -e .`. The test suite, fixtures and examples are excluded by scope; the golden parity fixtures under `tests/fixtures/rade_qnet/` are binary `.npy` files and cannot travel as text at all.

## Directories, in creation order

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 1 | [`_repository.md`](_repository.md) | `.` | 1 | 5,974 |
| 2 | [`_root.md`](_root.md) | `src/rade_qnet` | 3 | 32,958 |
| 3 | [`analysis.md`](analysis.md) | `src/rade_qnet/analysis` | 1 | 955 |
| 4 | [`analysis__metrics.md`](analysis__metrics.md) | `src/rade_qnet/analysis/metrics` | 4 | 36,684 |
| 5 | [`analysis__reports.md`](analysis__reports.md) | `src/rade_qnet/analysis/reports` | 6 | 48,444 |
| 6 | [`analysis__visuals.md`](analysis__visuals.md) | `src/rade_qnet/analysis/visuals` | 9 | 87,463 |
| 7 | [`core.md`](core.md) | `src/rade_qnet/core` | 1 | 1,277 |
| 8 | [`core__capability.md`](core__capability.md) | `src/rade_qnet/core/capability` | 4 | 39,365 |
| 9 | [`core__contract.md`](core__contract.md) | `src/rade_qnet/core/contract` | 9 | 91,726 |
| 10 | [`core__runtime.md`](core__runtime.md) | `src/rade_qnet/core/runtime` | 9 | 76,283 |
| 11 | [`core__spec.md`](core__spec.md) | `src/rade_qnet/core/spec` | 10 | 99,461 |
| 12 | [`engines.md`](engines.md) | `src/rade_qnet/engines` | 2 | 19,516 |
| 13 | [`engines__sklearn.md`](engines__sklearn.md) | `src/rade_qnet/engines/sklearn` | 3 | 33,462 |
| 14 | [`engines__torch.md`](engines__torch.md) | `src/rade_qnet/engines/torch` | 12 | 136,640 |
| 15 | [`engines__torch__learners.md`](engines__torch__learners.md) | `src/rade_qnet/engines/torch/learners` | 2 | 11,016 |
| 16 | [`engines__xgboost.md`](engines__xgboost.md) | `src/rade_qnet/engines/xgboost` | 2 | 29,626 |
| 17 | [`models.md`](models.md) | `src/rade_qnet/models` | 1 | 4,459 |
| 18 | [`models__hybrid_gnn_rnn.md`](models__hybrid_gnn_rnn.md) | `src/rade_qnet/models/hybrid_gnn_rnn` | 8 | 101,934 |
| 19 | [`models__hybrid_gnn_rnn__features.md`](models__hybrid_gnn_rnn__features.md) | `src/rade_qnet/models/hybrid_gnn_rnn/features` | 4 | 55,018 |
| 20 | [`models__hybrid_gnn_rnn__layers.md`](models__hybrid_gnn_rnn__layers.md) | `src/rade_qnet/models/hybrid_gnn_rnn/layers` | 6 | 59,229 |
| 21 | [`models__hybrid_gnn_rnn__pipelines.md`](models__hybrid_gnn_rnn__pipelines.md) | `src/rade_qnet/models/hybrid_gnn_rnn/pipelines` | 4 | 24,920 |
| 22 | [`models__lstm_tabular.md`](models__lstm_tabular.md) | `src/rade_qnet/models/lstm_tabular` | 5 | 14,753 |
| 23 | [`models__ridge.md`](models__ridge.md) | `src/rade_qnet/models/ridge` | 5 | 10,767 |
| 24 | [`models__xgb_tabular.md`](models__xgb_tabular.md) | `src/rade_qnet/models/xgb_tabular` | 5 | 11,127 |
| 25 | [`orchestration.md`](orchestration.md) | `src/rade_qnet/orchestration` | 1 | 1,227 |
| 26 | [`orchestration__compute.md`](orchestration__compute.md) | `src/rade_qnet/orchestration/compute` | 6 | 47,783 |
| 27 | [`orchestration__jobs.md`](orchestration__jobs.md) | `src/rade_qnet/orchestration/jobs` | 6 | 58,409 |
| 28 | [`orchestration__pipelines.md`](orchestration__pipelines.md) | `src/rade_qnet/orchestration/pipelines` | 9 | 116,395 |
| 29 | [`sources.md`](sources.md) | `src/rade_qnet/sources` | 1 | 1,247 |
| 30 | [`sources__batching.md`](sources__batching.md) | `src/rade_qnet/sources/batching` | 2 | 17,008 |
| 31 | [`sources__dataset.md`](sources__dataset.md) | `src/rade_qnet/sources/dataset` | 5 | 92,968 |
| 32 | [`sources__dataset__transforms.md`](sources__dataset__transforms.md) | `src/rade_qnet/sources/dataset/transforms` | 6 | 69,409 |
| 33 | [`sources__environment.md`](sources__environment.md) | `src/rade_qnet/sources/environment` | 1 | 1,563 |
| 34 | [`sources__environment__adapters.md`](sources__environment__adapters.md) | `src/rade_qnet/sources/environment/adapters` | 1 | 580 |
| 35 | [`storage.md`](storage.md) | `src/rade_qnet/storage` | 7 | 89,242 |
| 36 | [`testkit.md`](testkit.md) | `src/rade_qnet/testkit` | 4 | 122,845 |

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
d1c1797478f192a4  src/rade_qnet/api.py
f84c1e95253049a7  src/rade_qnet/ruff.toml
c610c7393d3bee5b  src/rade_qnet/analysis/__init__.py
6bc47bdcc1b578ae  src/rade_qnet/analysis/metrics/__init__.py
a6c640555cfab43e  src/rade_qnet/analysis/metrics/drift.py
0378d66d5bfb2ed7  src/rade_qnet/analysis/metrics/quality.py
2e7f22aa6facf8dd  src/rade_qnet/analysis/metrics/regression.py
67191a456d8c17a9  src/rade_qnet/analysis/reports/__init__.py
25d759f42d87b0a2  src/rade_qnet/analysis/reports/base.py
fb7a6c6ed77475da  src/rade_qnet/analysis/reports/baselines.py
9db36b4306e4f1d9  src/rade_qnet/analysis/reports/curves.py
233491a91e32745d  src/rade_qnet/analysis/reports/quality.py
225ff04319798e23  src/rade_qnet/analysis/reports/summary.py
3f467edddb1956a3  src/rade_qnet/analysis/visuals/__init__.py
e760737083820f0f  src/rade_qnet/analysis/visuals/data.py
e2b5e6e1d13d1d12  src/rade_qnet/analysis/visuals/evaluation.py
1fadb124361a60af  src/rade_qnet/analysis/visuals/export.py
56267c058caa227e  src/rade_qnet/analysis/visuals/jobset.py
ee99c29d23ab2900  src/rade_qnet/analysis/visuals/primitives.py
f6b3228dede4416b  src/rade_qnet/analysis/visuals/style.py
6449731d6368dfe8  src/rade_qnet/analysis/visuals/training.py
b0d6b5a02cec0ab7  src/rade_qnet/analysis/visuals/tuning.py
5288c88f526124bc  src/rade_qnet/core/__init__.py
30037d65e0ddf548  src/rade_qnet/core/capability/__init__.py
0cd0a0d1750e8b1c  src/rade_qnet/core/capability/definition.py
438101cea82e2646  src/rade_qnet/core/capability/protocols.py
8e2b5101bac2f3be  src/rade_qnet/core/capability/supervised.py
03e5e8f82e62f796  src/rade_qnet/core/contract/__init__.py
81ce5965d3e537f2  src/rade_qnet/core/contract/base.py
231daf12828c92fa  src/rade_qnet/core/contract/bundle.py
7683689102e6d630  src/rade_qnet/core/contract/data.py
b0fd78b19bc479cd  src/rade_qnet/core/contract/requirement.py
42cbfb1d3debd019  src/rade_qnet/core/contract/result.py
fcb7ccefc6bb9043  src/rade_qnet/core/contract/signature.py
0f58535c7104056f  src/rade_qnet/core/contract/source.py
c9795030254a3196  src/rade_qnet/core/contract/state.py
041af9b78a9927f4  src/rade_qnet/core/runtime/__init__.py
d4892d7b478201e0  src/rade_qnet/core/runtime/components.py
4a20729f709d8771  src/rade_qnet/core/runtime/context.py
f13eb08c3452f71f  src/rade_qnet/core/runtime/errors.py
f6d9f4a484edae26  src/rade_qnet/core/runtime/hashing.py
a12ae9f374abea76  src/rade_qnet/core/runtime/hooks.py
f107778d8a86e24c  src/rade_qnet/core/runtime/logging.py
49e5a0d89f731b58  src/rade_qnet/core/runtime/pipeline.py
bf9dcd660de9c2cb  src/rade_qnet/core/runtime/seeding.py
8d034c0a484bd379  src/rade_qnet/core/spec/__init__.py
0a27b41675af387a  src/rade_qnet/core/spec/base.py
460a756d41e5f3a6  src/rade_qnet/core/spec/data.py
82afd37e1cf52534  src/rade_qnet/core/spec/hardware.py
9c2d0251603666ce  src/rade_qnet/core/spec/jobs.py
89eeb600e0a48ae8  src/rade_qnet/core/spec/merge.py
090156f7b9a0c8a7  src/rade_qnet/core/spec/reports.py
1b82a04a916ca8f0  src/rade_qnet/core/spec/run.py
70e3ec454e04fd70  src/rade_qnet/core/spec/training.py
80cff9f039f91d2f  src/rade_qnet/core/spec/tune.py
ab977f1168bbb0b0  src/rade_qnet/engines/__init__.py
51a0d4d8b27f4e4d  src/rade_qnet/engines/base.py
9687c52982db50ec  src/rade_qnet/engines/sklearn/__init__.py
db160d94f8229598  src/rade_qnet/engines/sklearn/adapters.py
5afad89979d0fb3e  src/rade_qnet/engines/sklearn/engine.py
a386a3c89b5a653f  src/rade_qnet/engines/torch/__init__.py
dd2f519795a353cc  src/rade_qnet/engines/torch/callbacks.py
ede6db2968503f60  src/rade_qnet/engines/torch/checkpoint.py
0c0492e6dc473abb  src/rade_qnet/engines/torch/distributed.py
2d7faa241a18fb2b  src/rade_qnet/engines/torch/engine.py
1e6d10f5dc810e55  src/rade_qnet/engines/torch/hardware.py
856ddcfb1c2ca4e9  src/rade_qnet/engines/torch/loaders.py
f43d783b05f0da87  src/rade_qnet/engines/torch/loops.py
43b81664fc961647  src/rade_qnet/engines/torch/losses.py
b19a9c1b5031cb03  src/rade_qnet/engines/torch/materialise.py
87c7e1bff8d8aba1  src/rade_qnet/engines/torch/predictor.py
34f9f4cfe1207350  src/rade_qnet/engines/torch/seeding.py
dc9200473af4132e  src/rade_qnet/engines/torch/learners/__init__.py
5306ebe1215c4405  src/rade_qnet/engines/torch/learners/supervised.py
d6ff5a7febff9ffa  src/rade_qnet/engines/xgboost/__init__.py
453ebc9e6ebf2cf4  src/rade_qnet/engines/xgboost/engine.py
7632eb119762cbe8  src/rade_qnet/models/__init__.py
d99cf28ec67f4ec0  src/rade_qnet/models/hybrid_gnn_rnn/__init__.py
4a15225894ac6a82  src/rade_qnet/models/hybrid_gnn_rnn/data.py
e16abf150582ac66  src/rade_qnet/models/hybrid_gnn_rnn/model.py
66fbc28dae99d634  src/rade_qnet/models/hybrid_gnn_rnn/register.py
11d6f5c7bbcf3f41  src/rade_qnet/models/hybrid_gnn_rnn/reports.py
a177505b14867755  src/rade_qnet/models/hybrid_gnn_rnn/spec.py
2e5ee264c98828d9  src/rade_qnet/models/hybrid_gnn_rnn/state.py
3ed7119fc00abd58  src/rade_qnet/models/hybrid_gnn_rnn/visuals.py
011ecbecf59e5538  src/rade_qnet/models/hybrid_gnn_rnn/features/__init__.py
dd3b20cf7b1ef685  src/rade_qnet/models/hybrid_gnn_rnn/features/basis.py
02df79a5aad79a4d  src/rade_qnet/models/hybrid_gnn_rnn/features/encoder.py
03c836026a25e1fa  src/rade_qnet/models/hybrid_gnn_rnn/features/graph.py
f2731bbc54fbfd3c  src/rade_qnet/models/hybrid_gnn_rnn/layers/__init__.py
6fb45ef8e7810e21  src/rade_qnet/models/hybrid_gnn_rnn/layers/attention.py
2f03b1c804ddac1c  src/rade_qnet/models/hybrid_gnn_rnn/layers/fusion.py
c7254b8adf42554e  src/rade_qnet/models/hybrid_gnn_rnn/layers/gnn.py
6a55cc47c7c76109  src/rade_qnet/models/hybrid_gnn_rnn/layers/projection.py
858a3e85fbd754ca  src/rade_qnet/models/hybrid_gnn_rnn/layers/rnn.py
2ee06142a0e2a995  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/__init__.py
67a230246fd9e7fd  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/eval.py
5e713339832a8ff7  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/train.py
ed3556248dba4e94  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/tune.py
8ed63db8039871d8  src/rade_qnet/models/lstm_tabular/__init__.py
338b3f7edf0d2b0e  src/rade_qnet/models/lstm_tabular/data.py
cac89769b93bf1b1  src/rade_qnet/models/lstm_tabular/model.py
6c27ef7ca1b2f4c6  src/rade_qnet/models/lstm_tabular/register.py
684004c8b0263ebe  src/rade_qnet/models/lstm_tabular/spec.py
8e878ba14afd847c  src/rade_qnet/models/ridge/__init__.py
45111227031bd139  src/rade_qnet/models/ridge/data.py
86653e260ade081f  src/rade_qnet/models/ridge/model.py
1cacbb9686650b6a  src/rade_qnet/models/ridge/register.py
968365b42f941d08  src/rade_qnet/models/ridge/spec.py
8a5273793dc9fa45  src/rade_qnet/models/xgb_tabular/__init__.py
f7ff36bf6ec13828  src/rade_qnet/models/xgb_tabular/data.py
c7cf7a7b0632cefd  src/rade_qnet/models/xgb_tabular/model.py
902f0cd66d6c1738  src/rade_qnet/models/xgb_tabular/register.py
446633bfa7a1d14b  src/rade_qnet/models/xgb_tabular/spec.py
41743506e557df79  src/rade_qnet/orchestration/__init__.py
4c51e1a4f4256585  src/rade_qnet/orchestration/compute/__init__.py
06675d4d776bb767  src/rade_qnet/orchestration/compute/base.py
355ebcbfadec27d6  src/rade_qnet/orchestration/compute/gpus.py
875720e4a6d53480  src/rade_qnet/orchestration/compute/local.py
144dd00b30eed227  src/rade_qnet/orchestration/compute/policy.py
8ad395c7655c5e83  src/rade_qnet/orchestration/compute/processes.py
38c0c4fe2f3800a8  src/rade_qnet/orchestration/jobs/__init__.py
53090f7ef26e54f6  src/rade_qnet/orchestration/jobs/fanout.py
e51d04f043e63356  src/rade_qnet/orchestration/jobs/groups.py
0a5173658b335f6b  src/rade_qnet/orchestration/jobs/manifest.py
8aa5850984437273  src/rade_qnet/orchestration/jobs/set.py
dbcca9ec2df2b2c8  src/rade_qnet/orchestration/jobs/unit.py
f610fecec664a1cf  src/rade_qnet/orchestration/pipelines/__init__.py
3150d1c68327da62  src/rade_qnet/orchestration/pipelines/evaluate.py
9c1ae0b09dee881d  src/rade_qnet/orchestration/pipelines/infer.py
610dd23976455dee  src/rade_qnet/orchestration/pipelines/reload.py
af2865110166c9da  src/rade_qnet/orchestration/pipelines/resolve.py
3d037aa08bb52fc8  src/rade_qnet/orchestration/pipelines/scoring.py
bc5a82484ab0bcb8  src/rade_qnet/orchestration/pipelines/search.py
c39a4a5b04adcfb7  src/rade_qnet/orchestration/pipelines/train.py
60a311d8aa2dc1ea  src/rade_qnet/orchestration/pipelines/tune.py
0933cff34ac5afc2  src/rade_qnet/sources/__init__.py
8e7381b320b6c155  src/rade_qnet/sources/batching/__init__.py
40393e1e40fc3e40  src/rade_qnet/sources/batching/dataset.py
2c7ff96aeafa2241  src/rade_qnet/sources/dataset/__init__.py
9ae0d7768a5d9898  src/rade_qnet/sources/dataset/io.py
1734abceb05cab41  src/rade_qnet/sources/dataset/module.py
4b2cf2882be64f4a  src/rade_qnet/sources/dataset/rebuild.py
dd04e7a448a725a6  src/rade_qnet/sources/dataset/splits.py
6d3de29fb11b2a0c  src/rade_qnet/sources/dataset/transforms/__init__.py
85ad501d4119cef3  src/rade_qnet/sources/dataset/transforms/composite.py
5eaafa54164192bf  src/rade_qnet/sources/dataset/transforms/encoding.py
baeefd1eca6f0d24  src/rade_qnet/sources/dataset/transforms/reduction.py
8fc3a8a94b8876db  src/rade_qnet/sources/dataset/transforms/scaling.py
2244f719b1b09f14  src/rade_qnet/sources/dataset/transforms/sequence.py
3cf2edd3f52440a4  src/rade_qnet/sources/environment/__init__.py
d4274fd3c2191e72  src/rade_qnet/sources/environment/adapters/__init__.py
39b4f86a55beec16  src/rade_qnet/storage/__init__.py
2bcc86c5342517ed  src/rade_qnet/storage/bundle.py
49cd75d16f6407da  src/rade_qnet/storage/catalog.py
e4de3cbef37a2525  src/rade_qnet/storage/locking.py
aedb0848af6c4401  src/rade_qnet/storage/manifest.py
bd900ec7476ad023  src/rade_qnet/storage/registry.py
49262e26b707ae17  src/rade_qnet/storage/tracker.py
be147ade92895e66  src/rade_qnet/testkit/__init__.py
d4305777a4651f68  src/rade_qnet/testkit/conformance.py
77c26e7f74462931  src/rade_qnet/testkit/fixtures.py
ed3df40d131fa2e1  src/rade_qnet/testkit/parity.py
```

