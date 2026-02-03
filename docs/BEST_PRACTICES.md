# QuantStrata Best Practices

A short guide for contributors and users: coding standards, testing, performance, and project conventions.

---

## Coding standards

### Style and formatting

- Use **Python 3.12+** (the project uses features such as `dataclass(slots=True)`).
- Prefer **type hints** on public functions and class methods. Use `from __future__ import annotations` at the top of files to allow forward references and keep runtime cost minimal.
- Use **docstrings** (Google or NumPy style) for public modules, classes, and functions. Include a short description, `Parameters`/`Returns`/`Raises` where helpful, and `Attributes` for dataclasses.
- Keep **line length** reasonable (e.g. 100–120 characters); break long lines for readability.

### Data and configuration

- **Config and result types:** Use immutable **dataclasses** with `frozen=True` and `slots=True` for configuration and result objects (e.g. `VarConfig`, `VarResult`, `GreeksSummary`). Use `field(default_factory=list)` for mutable defaults.
- **No pandas in core library code:** The risk reporting layer (e.g. `SensitivitiesReport`, `AttributionReport`, `RiskReport`) and related code use plain Python types (`list`, `dict`, `Mapping`) and `.to_dicts()` / `.to_csv()` for export so that the library does not depend on pandas. Use pandas in notebooks or application code if needed.
- **Interfaces:** Prefer **protocols** or small abstract interfaces for pluggable behaviour (e.g. `MarketDataProvider`, `BrokerageAdapter`, `StreamingMarketDataProtocol`).

---

## Testing

### Running tests

- Create and activate a **virtual environment** (e.g. `python -m venv .venv`, `source .venv/bin/activate`).
- Install dependencies: `pip install -r requirements.txt`.
- Run the full test suite from the **repository root** so that `src` is importable:
  ```bash
  pytest tests/ -v
  ```
- Use **Python 3.12** for unit tests (some tests rely on 3.10+ features).
- Optional/extra dependencies (e.g. matplotlib, JAX): tests that require them use `pytest.importorskip("module")` so they are skipped when the module is not installed.

### Test structure and patterns

- Unit tests live under `tests/unit/`, mirroring `src/` package structure.
- Prefer **small, focused tests** that assert one behaviour. Use fixtures for shared setup.
- For reporting and serialisation, test **round-trips** (e.g. `to_dicts()` / `to_csv()` content) and that console output contains expected fields.

---

## Performance

- **Caching:** Use `CachingMarketDataProvider` and `CachingPortfolioPricer` when the same market or portfolio is priced repeatedly (e.g. in scenarios or risk runs). See [Performance and Scalability](guides/performance/performance_and_scalability.md).
- **Parallel pricing:** For large portfolios, use `ParallelPortfolioPricer` (thread pool) from `src.portfolio.parallel` to price positions in parallel.
- **JAX (optional):** When JAX is installed, the library can use a JAX-based MC pricer for FX vanilla options (`pricer_id="jax_mc"`). Use it for high-throughput pricing where applicable.
- **Profiling:** Use the existing benchmarking utilities in `src/core/performance/` to measure and compare pricer or pipeline performance.

---

## Documentation and examples

- **Guides** (`docs/guides/`) are user-focused; **reference** (`docs/reference/`) covers APIs and mathematical detail. **Tutorials** (`docs/tutorials/`) are Jupyter notebooks.
- When adding a new feature, add or update the relevant guide or reference and, if useful, a short example in `examples/` or a tutorial notebook.
- Link new docs from `docs/guides/README.md`, `docs/tutorials/README.md`, or `docs/reference/README.md` as appropriate.

---

## Conventions summary

| Area        | Convention |
|------------|------------|
| Python     | 3.12+ |
| Config/result types | Immutable dataclasses, `frozen=True`, `slots=True` |
| Core risk/reporting | No pandas; use `.to_dicts()` / `.to_csv()` for export |
| Pluggable behaviour | Protocols / small interfaces |
| Tests      | `tests/unit/`, pytest, run from repo root with venv |
| Performance | Caching and parallel pricers where appropriate; optional JAX |

For more detail, see the [development](development/) docs and the [roadmap](development/roadmap.md).
