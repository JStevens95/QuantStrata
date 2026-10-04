# Porting `rade_qnet` through a markdown-only proxy

37 documents, 165 files, 45,367 lines, 1,598,679 bytes.

Each document below covers one directory. Work down the list in order: a parent directory always appears before its children, so the tree is importable at every step.

## What is not here

`src/rade_qnet/docs/` is already markdown, so it crosses the proxy unchanged — fetch those files directly rather than through this set. The test suite, fixtures and examples are excluded by scope; the golden parity fixtures under `tests/fixtures/rade_qnet/` are binary `.npy` files and cannot travel as text at all.

## Directories, in creation order

| # | Document | Directory | Files | Bytes |
| --- | --- | --- | ---: | ---: |
| 1 | [`_root.md`](_root.md) | `src/rade_qnet` | 3 | 31,604 |
| 2 | [`analysis.md`](analysis.md) | `src/rade_qnet/analysis` | 1 | 955 |
| 3 | [`analysis__metrics.md`](analysis__metrics.md) | `src/rade_qnet/analysis/metrics` | 4 | 36,696 |
| 4 | [`analysis__reports.md`](analysis__reports.md) | `src/rade_qnet/analysis/reports` | 6 | 48,444 |
| 5 | [`analysis__visuals.md`](analysis__visuals.md) | `src/rade_qnet/analysis/visuals` | 9 | 87,515 |
| 6 | [`core.md`](core.md) | `src/rade_qnet/core` | 1 | 1,277 |
| 7 | [`core__capability.md`](core__capability.md) | `src/rade_qnet/core/capability` | 4 | 37,838 |
| 8 | [`core__contract.md`](core__contract.md) | `src/rade_qnet/core/contract` | 9 | 91,828 |
| 9 | [`core__runtime.md`](core__runtime.md) | `src/rade_qnet/core/runtime` | 9 | 75,880 |
| 10 | [`core__spec.md`](core__spec.md) | `src/rade_qnet/core/spec` | 10 | 97,248 |
| 11 | [`domains.md`](domains.md) | `src/rade_qnet/domains` | 1 | 1,251 |
| 12 | [`domains__pnl.md`](domains__pnl.md) | `src/rade_qnet/domains/pnl` | 4 | 25,948 |
| 13 | [`engines.md`](engines.md) | `src/rade_qnet/engines` | 2 | 19,516 |
| 14 | [`engines__sklearn.md`](engines__sklearn.md) | `src/rade_qnet/engines/sklearn` | 3 | 33,522 |
| 15 | [`engines__torch.md`](engines__torch.md) | `src/rade_qnet/engines/torch` | 12 | 136,698 |
| 16 | [`engines__torch__learners.md`](engines__torch__learners.md) | `src/rade_qnet/engines/torch/learners` | 2 | 11,016 |
| 17 | [`engines__xgboost.md`](engines__xgboost.md) | `src/rade_qnet/engines/xgboost` | 2 | 29,750 |
| 18 | [`models.md`](models.md) | `src/rade_qnet/models` | 1 | 4,459 |
| 19 | [`models__hybrid_gnn_rnn.md`](models__hybrid_gnn_rnn.md) | `src/rade_qnet/models/hybrid_gnn_rnn` | 8 | 97,597 |
| 20 | [`models__hybrid_gnn_rnn__features.md`](models__hybrid_gnn_rnn__features.md) | `src/rade_qnet/models/hybrid_gnn_rnn/features` | 4 | 55,018 |
| 21 | [`models__hybrid_gnn_rnn__layers.md`](models__hybrid_gnn_rnn__layers.md) | `src/rade_qnet/models/hybrid_gnn_rnn/layers` | 6 | 59,229 |
| 22 | [`models__hybrid_gnn_rnn__pipelines.md`](models__hybrid_gnn_rnn__pipelines.md) | `src/rade_qnet/models/hybrid_gnn_rnn/pipelines` | 4 | 24,964 |
| 23 | [`models__lstm_tabular.md`](models__lstm_tabular.md) | `src/rade_qnet/models/lstm_tabular` | 5 | 14,768 |
| 24 | [`models__ridge.md`](models__ridge.md) | `src/rade_qnet/models/ridge` | 5 | 10,757 |
| 25 | [`models__xgb_tabular.md`](models__xgb_tabular.md) | `src/rade_qnet/models/xgb_tabular` | 5 | 11,131 |
| 26 | [`orchestration.md`](orchestration.md) | `src/rade_qnet/orchestration` | 1 | 1,227 |
| 27 | [`orchestration__compute.md`](orchestration__compute.md) | `src/rade_qnet/orchestration/compute` | 6 | 47,783 |
| 28 | [`orchestration__jobs.md`](orchestration__jobs.md) | `src/rade_qnet/orchestration/jobs` | 4 | 37,853 |
| 29 | [`orchestration__pipelines.md`](orchestration__pipelines.md) | `src/rade_qnet/orchestration/pipelines` | 9 | 116,562 |
| 30 | [`sources.md`](sources.md) | `src/rade_qnet/sources` | 1 | 1,247 |
| 31 | [`sources__batching.md`](sources__batching.md) | `src/rade_qnet/sources/batching` | 2 | 17,008 |
| 32 | [`sources__dataset.md`](sources__dataset.md) | `src/rade_qnet/sources/dataset` | 5 | 92,968 |
| 33 | [`sources__dataset__transforms.md`](sources__dataset__transforms.md) | `src/rade_qnet/sources/dataset/transforms` | 6 | 69,409 |
| 34 | [`sources__environment.md`](sources__environment.md) | `src/rade_qnet/sources/environment` | 1 | 1,563 |
| 35 | [`sources__environment__adapters.md`](sources__environment__adapters.md) | `src/rade_qnet/sources/environment/adapters` | 1 | 580 |
| 36 | [`storage.md`](storage.md) | `src/rade_qnet/storage` | 5 | 44,725 |
| 37 | [`testkit.md`](testkit.md) | `src/rade_qnet/testkit` | 4 | 122,845 |

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
9027c766e9b505ad  src/rade_qnet/__init__.py
e8212d45f53fd701  src/rade_qnet/api.py
f84c1e95253049a7  src/rade_qnet/ruff.toml
c610c7393d3bee5b  src/rade_qnet/analysis/__init__.py
6bc47bdcc1b578ae  src/rade_qnet/analysis/metrics/__init__.py
902254ea33203124  src/rade_qnet/analysis/metrics/drift.py
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
5cc711d855910062  src/rade_qnet/analysis/visuals/tuning.py
5288c88f526124bc  src/rade_qnet/core/__init__.py
091c40db891a0eac  src/rade_qnet/core/capability/__init__.py
0cd0a0d1750e8b1c  src/rade_qnet/core/capability/definition.py
438101cea82e2646  src/rade_qnet/core/capability/protocols.py
5961853f29703f32  src/rade_qnet/core/capability/simple.py
03e5e8f82e62f796  src/rade_qnet/core/contract/__init__.py
81ce5965d3e537f2  src/rade_qnet/core/contract/base.py
231daf12828c92fa  src/rade_qnet/core/contract/bundle.py
7683689102e6d630  src/rade_qnet/core/contract/data.py
bff48fa703f13c1f  src/rade_qnet/core/contract/requirement.py
42cbfb1d3debd019  src/rade_qnet/core/contract/result.py
fcb7ccefc6bb9043  src/rade_qnet/core/contract/signature.py
0f58535c7104056f  src/rade_qnet/core/contract/source.py
c9795030254a3196  src/rade_qnet/core/contract/state.py
041af9b78a9927f4  src/rade_qnet/core/runtime/__init__.py
472ee6303390d874  src/rade_qnet/core/runtime/components.py
8b11496fe3a01eee  src/rade_qnet/core/runtime/context.py
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
52fd20676baea8dc  src/rade_qnet/core/spec/jobs.py
89eeb600e0a48ae8  src/rade_qnet/core/spec/merge.py
090156f7b9a0c8a7  src/rade_qnet/core/spec/reports.py
1b82a04a916ca8f0  src/rade_qnet/core/spec/run.py
70e3ec454e04fd70  src/rade_qnet/core/spec/training.py
04de021fa2194e1d  src/rade_qnet/core/spec/tune.py
ab3c59729154868e  src/rade_qnet/domains/__init__.py
991ce35b86efaba6  src/rade_qnet/domains/pnl/__init__.py
5c72e16b36cc41f9  src/rade_qnet/domains/pnl/clusters.py
a513a7e3381577cf  src/rade_qnet/domains/pnl/portfolio.py
de16151432225b48  src/rade_qnet/domains/pnl/universe.py
ab977f1168bbb0b0  src/rade_qnet/engines/__init__.py
51a0d4d8b27f4e4d  src/rade_qnet/engines/base.py
9687c52982db50ec  src/rade_qnet/engines/sklearn/__init__.py
db160d94f8229598  src/rade_qnet/engines/sklearn/adapters.py
971b2412886894bf  src/rade_qnet/engines/sklearn/engine.py
3634e983af7166b6  src/rade_qnet/engines/torch/__init__.py
dd2f519795a353cc  src/rade_qnet/engines/torch/callbacks.py
ede6db2968503f60  src/rade_qnet/engines/torch/checkpoint.py
0c0492e6dc473abb  src/rade_qnet/engines/torch/distributed.py
2d7faa241a18fb2b  src/rade_qnet/engines/torch/engine.py
1e6d10f5dc810e55  src/rade_qnet/engines/torch/hardware.py
856ddcfb1c2ca4e9  src/rade_qnet/engines/torch/loaders.py
f43d783b05f0da87  src/rade_qnet/engines/torch/loops.py
2992dbf812cdca1e  src/rade_qnet/engines/torch/losses.py
b19a9c1b5031cb03  src/rade_qnet/engines/torch/materialise.py
87c7e1bff8d8aba1  src/rade_qnet/engines/torch/predictor.py
34f9f4cfe1207350  src/rade_qnet/engines/torch/seeding.py
dc9200473af4132e  src/rade_qnet/engines/torch/learners/__init__.py
5306ebe1215c4405  src/rade_qnet/engines/torch/learners/supervised.py
d6ff5a7febff9ffa  src/rade_qnet/engines/xgboost/__init__.py
7c260ae5c39673cf  src/rade_qnet/engines/xgboost/engine.py
7632eb119762cbe8  src/rade_qnet/models/__init__.py
d99cf28ec67f4ec0  src/rade_qnet/models/hybrid_gnn_rnn/__init__.py
4a15225894ac6a82  src/rade_qnet/models/hybrid_gnn_rnn/data.py
e16abf150582ac66  src/rade_qnet/models/hybrid_gnn_rnn/model.py
23f8abc77104eaf8  src/rade_qnet/models/hybrid_gnn_rnn/register.py
11d6f5c7bbcf3f41  src/rade_qnet/models/hybrid_gnn_rnn/reports.py
a177505b14867755  src/rade_qnet/models/hybrid_gnn_rnn/spec.py
d0b0bbe4416507d0  src/rade_qnet/models/hybrid_gnn_rnn/state.py
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
cf2bbeb6ebecc16c  src/rade_qnet/models/hybrid_gnn_rnn/pipelines/tune.py
8ed63db8039871d8  src/rade_qnet/models/lstm_tabular/__init__.py
338b3f7edf0d2b0e  src/rade_qnet/models/lstm_tabular/data.py
0df5a27b646f6ec5  src/rade_qnet/models/lstm_tabular/model.py
d6a19e8200e6cb55  src/rade_qnet/models/lstm_tabular/register.py
684004c8b0263ebe  src/rade_qnet/models/lstm_tabular/spec.py
8e878ba14afd847c  src/rade_qnet/models/ridge/__init__.py
45111227031bd139  src/rade_qnet/models/ridge/data.py
86653e260ade081f  src/rade_qnet/models/ridge/model.py
0eee88e46a642960  src/rade_qnet/models/ridge/register.py
968365b42f941d08  src/rade_qnet/models/ridge/spec.py
8a5273793dc9fa45  src/rade_qnet/models/xgb_tabular/__init__.py
f7ff36bf6ec13828  src/rade_qnet/models/xgb_tabular/data.py
c7cf7a7b0632cefd  src/rade_qnet/models/xgb_tabular/model.py
a16f7ac199cfcc32  src/rade_qnet/models/xgb_tabular/register.py
446633bfa7a1d14b  src/rade_qnet/models/xgb_tabular/spec.py
41743506e557df79  src/rade_qnet/orchestration/__init__.py
4c51e1a4f4256585  src/rade_qnet/orchestration/compute/__init__.py
06675d4d776bb767  src/rade_qnet/orchestration/compute/base.py
355ebcbfadec27d6  src/rade_qnet/orchestration/compute/gpus.py
875720e4a6d53480  src/rade_qnet/orchestration/compute/local.py
144dd00b30eed227  src/rade_qnet/orchestration/compute/policy.py
8ad395c7655c5e83  src/rade_qnet/orchestration/compute/processes.py
6d1b581f5bd21ac3  src/rade_qnet/orchestration/jobs/__init__.py
0a5173658b335f6b  src/rade_qnet/orchestration/jobs/manifest.py
309bbb0a049e589c  src/rade_qnet/orchestration/jobs/set.py
dbcca9ec2df2b2c8  src/rade_qnet/orchestration/jobs/unit.py
f610fecec664a1cf  src/rade_qnet/orchestration/pipelines/__init__.py
509da2af4be7d5ca  src/rade_qnet/orchestration/pipelines/evaluate.py
1b6d06b15a104a75  src/rade_qnet/orchestration/pipelines/infer.py
260a299a85ba559c  src/rade_qnet/orchestration/pipelines/reload.py
af2865110166c9da  src/rade_qnet/orchestration/pipelines/resolve.py
7ebf299601abd91d  src/rade_qnet/orchestration/pipelines/scoring.py
0f1f9353cef00b38  src/rade_qnet/orchestration/pipelines/search.py
7610e61b277c2c10  src/rade_qnet/orchestration/pipelines/train.py
f81deb7b747fcf9c  src/rade_qnet/orchestration/pipelines/tune.py
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
254809fcd93a9f5a  src/rade_qnet/storage/__init__.py
aa4fc03feda237ed  src/rade_qnet/storage/bundle.py
27ca69a90dac9cfb  src/rade_qnet/storage/catalog.py
aedb0848af6c4401  src/rade_qnet/storage/manifest.py
49262e26b707ae17  src/rade_qnet/storage/tracker.py
be147ade92895e66  src/rade_qnet/testkit/__init__.py
d4305777a4651f68  src/rade_qnet/testkit/conformance.py
77c26e7f74462931  src/rade_qnet/testkit/fixtures.py
ed3df40d131fa2e1  src/rade_qnet/testkit/parity.py
```

