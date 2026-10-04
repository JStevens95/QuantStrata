# rade_qnet — Coding Standards

The standard every contribution to `rade_qnet` is held to. It is referenced by
every phase's definition of done in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

Three things make this document worth reading rather than skimming:

- Most of it is **machine-checked**. Where a rule can be enforced by Ruff it
  is, so review time goes on design rather than on formatting.
- Every rule states **why**. A rule whose reason you disagree with is worth
  challenging; a rule you follow without knowing why gets applied wrongly.
- The section on [comments](#3-comments) **departs from a literal reading of
  the brief**, and explains the reasoning. Please read it and push back if you
  disagree.

---

## Contents

1. [Tooling](#1-tooling)
2. [Docstrings](#2-docstrings)
3. [Comments](#3-comments)
4. [Typing](#4-typing)
5. [Naming](#5-naming)
6. [Errors](#6-errors)
7. [Purity and side effects](#7-purity-and-side-effects)
8. [Imports](#8-imports)
9. [Tests](#9-tests)
10. [Review checklist](#10-review-checklist)

---

## 1. Tooling

```bash
# Lint — must report zero findings.
.venv/bin/python -m ruff check src/rade_qnet tests/rade_qnet

# Format — must report zero files needing change.
.venv/bin/python -m ruff format --check src/rade_qnet tests/rade_qnet

# Test — must pass.
.venv/bin/python -m pytest tests/rade_qnet -q
```

Configuration lives in [`../ruff.toml`](../ruff.toml), with the test tree's
[`tests/rade_qnet/ruff.toml`](../../../tests/rade_qnet/ruff.toml) extending it so
both trees are held to one standard stated in one place.

Two deliberate choices about that configuration:

**It is scoped to `rade_qnet`.** Ruff resolves configuration by walking up from
each linted file, so a config inside the package applies to the package and
nothing else. The repository's older packages predate this standard and would
bury a clean run in pre-existing findings, which would make the signal
worthless on day one.

**Rules are selected explicitly, not by exclusion.** The `select` list is long
and enumerated. That means upgrading Ruff cannot silently introduce new
failures, and the list itself documents the standard rather than deferring to a
default that may change under us. `select` covers the flake8 rule families
(`F`, `E`, `W`), import order (`I`), naming (`N`), docstrings (`D`), typing
(`ANN`), likely bugs (`B`), and a dozen more — see the file for the annotated
list.

> **Note.** Ruff is not currently in `requirements.txt`, and neither is
> `pydantic`, though `pydantic` 2.12 is installed in the virtual environment.
> Both need adding; see the open items in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

---

## 2. Docstrings

NumPy convention, enforced by `D` with `convention = "numpy"`. Chosen to match
the scientific Python stack this framework sits on, so a `rade_qnet` docstring
has the same shape as a NumPy or SciPy one.

**Every module, class, function and method needs one.** No exceptions for
private helpers — a private helper is exactly the thing a reader encounters
with no context.

A docstring states the **contract**: what goes in, what comes out, what can go
wrong, and any invariant a caller must respect. It is written for someone who
will call the function without reading it.

```python
def split_chronologically(
    n_scenarios: int,
    validation_fraction: float,
    test_fraction: float,
    sequence_length: int,
) -> SplitIndices:
    """
    Split a scenario axis into train, validation and test by time order.

    Boundaries are placed so that no sequence window straddles a split: the
    last ``sequence_length - 1`` scenarios before each boundary are dropped.
    Without that gap a window beginning in the training set would extend into
    validation, and the validation score would be partly a memory of training.

    Parameters
    ----------
    n_scenarios
        Length of the scenario axis.
    validation_fraction, test_fraction
        Fractions of the axis assigned to validation and test. Must sum to
        less than one.
    sequence_length
        Window length the model consumes. Pass ``1`` for a non-sequential
        model.

    Returns
    -------
    SplitIndices
        Disjoint index arrays covering the axis apart from the dropped gaps.

    Raises
    ------
    SpecError
        If the fractions are not in ``(0, 1)``, if they sum to one or more, or
        if the requested splits leave no training scenarios.
    """
```

### Package charters

A package's `__init__.py` docstring is its **charter** and carries more weight
than a normal module docstring. It states what belongs in the package, what
does not, which modules it will hold and in which phase, and — for a top-level
package — its dependency rule.

The charter is the first thing a contributor reads, and it is the only thing
that stops a sub-package quietly becoming a dumping ground. Every `__init__.py`
in the tree has one already; keep them current as modules land.

---

## 3. Comments

**This section departs from a literal reading of the original requirement
("plenty of code comments to allow a user to understand what each line is
doing"), and the reason is given below.**

### Why not comment every line

Taken literally, per-line commenting is actively harmful, for three reasons.

*It drifts.* A comment is not executed, so nothing detects when it stops being
true. Per-line comments produce the maximum possible surface area for drift,
and a comment that contradicts its code is worse than no comment — a reader who
trusts it is misled, and a reader who does not has learned to ignore all of
them.

*It buries the signal.* When every line is commented, the comment marking the
genuinely subtle line looks exactly like the comment on `total = a + b`. The
important remark becomes invisible by being surrounded.

*It substitutes for naming.* `x = d * 86400  # convert days to seconds` is a
comment compensating for a bad name. `seconds = days * SECONDS_PER_DAY` needs
no comment and cannot drift. Reaching for a comment is often a signal that the
code should be clearer instead.

### What this framework does instead

The underlying goal — *a reader can follow what is happening and why* — is
taken seriously and met by four mechanisms, applied in this order:

1. **Names that make the line self-explanatory.** The first tool, always.
2. **Docstrings that carry the contract.** Required everywhere, as above.
3. **Comments on intent, constraint and non-obvious mechanics.** Generous
   where they earn their place.
4. **Tests that demonstrate behaviour.** An executable explanation, and the
   only kind that cannot drift.

### When to write a comment

Write one when the code cannot say it itself:

| Write a comment for | Example |
| --- | --- |
| **Why**, when the reason is not local | `# Wrapping before materialise gives DDP an empty parameter group.` |
| **An ordering constraint** | `# Must precede any import of torch; CUDA caches visibility at import.` |
| **A non-obvious invariant** | `# train_indices is sorted; the gap logic below relies on it.` |
| **A deliberate deviation** | `# Fitting on all rows reproduces the legacy path for parity. Not the default.` |
| **A reference** | `# Clipping follows Schulman et al. (2017), eq. 7.` |
| **A rejected alternative** | `# A set would be faster here but the order is part of the saved state.` |

Do not write one to restate the code, to mark a section that should be a
function, to record who changed what (that is `git`), or to leave code
commented out (`ERA` rejects this).

### Density in practice

The commented lines in this framework cluster, as they should, around the parts
that are genuinely hard: the ordering of lazy-parameter materialisation,
split-boundary arithmetic, the device-visibility dance before import, the merge
rules for job overrides. Straightforward construction and delegation code
carries few comments and needs none.

As a calibration: `tests/rade_qnet/test_scaffold.py` is commented at roughly the
intended density. Every constant that encodes a decision explains that
decision; the loops that walk directories do not.

**If you want literal per-line commenting instead, say so and the standard will
be changed — but it should be a deliberate decision, not an accident of
phrasing.**

---

## 4. Typing

Full annotations on every signature, enforced by `ANN`. Return types included,
including `-> None`.

```python
from __future__ import annotations
```

At the top of every module. It makes annotations lazy, which means a forward
reference needs no quotes and a type-only import costs nothing at run time.

**`Protocol` over inheritance for extension points.** A `BatchSource`, an
`Engine`, a `Report` and every capability are protocols. A user's class
satisfies one by having the right methods, with no import of ours and no base
class — which is what lets a model live outside this repository.

**No bare `Any`.** If a type is genuinely open, say so precisely: a `TypeVar`
with a bound, a narrow union, or `object` with a documented narrowing. `Any`
disables checking exactly where a reader most needs help.

**`TYPE_CHECKING` for heavy imports.** A type-only import of `torch` in a
`core` module would violate the layering rule and slow every worker's start-up:

```python
if TYPE_CHECKING:
    from torch import Tensor
```

---

## 5. Naming

Names are the primary documentation mechanism, so they get more attention than
style guides usually give them.

| Rule | Instead of | Write |
| --- | --- | --- |
| No abbreviations a newcomer must decode | `cfg`, `ds`, `idx`, `tgt`, `hist` | `spec`, `source`, `indices`, `target`, `history` |
| Booleans read as assertions | `flag`, `check` | `is_fitted`, `has_static_inputs`, `should_stop` |
| Collections are plural; elements singular | `job` holding many | `jobs`, iterated as `job` |
| Functions are verb phrases | `data_build` | `build_data` |
| Predicates start `is_` / `has_` / `can_` | `valid` | `is_valid` |
| Dimensions named, not numbered | `n`, `m`, `d` | `n_scenarios`, `n_entities`, `n_features` |
| Units in the name when ambiguous | `timeout` | `timeout_seconds` |
| No stuttering | `spec.spec_version` | `spec.version` |

**Domain vocabulary is used consistently and exactly.** `scenario` is a point
on the time axis. `entity` is an instrument. `elementary` and `target` are the
two instrument roles. `job` is one run in a set, never "member", "cluster" or
"model". Where a word has an established meaning on the desk, that meaning
wins; where it does not, pick one and use it everywhere.

**Single-letter names are permitted in exactly one place:** a short
mathematical expression where the symbols match a cited formula, with the
citation in a comment. Not in a loop, not for a DataFrame, not for a path.

---

## 6. Errors

**Never swallow an exception.** `except Exception: pass` is how
`use_deterministic_algorithms(True)` came to be set globally in the previous
implementation without anybody knowing whether it had worked. If a failure is
tolerable, log it at warning with the exception attached and say in the message
what the consequence is.

```python
try:
    tracker.log_metrics(metrics)
except TrackerError:
    # Tracking is optional infrastructure: an unreachable server must not
    # discard a completed training run. The metrics are already in the bundle.
    logger.warning("metric tracking failed; metrics remain in the bundle", exc_info=True)
```

**Raise the framework's own error types,** from `core.runtime.errors`. A
`SpecError` is actionable by the user; a `ContractError` is a framework bug.
Collapsing both into `ValueError` loses that distinction at exactly the moment
it matters.

**Fail at the earliest possible point.** A spec is validated before a source is
built; a source is validated before a model is built. The cost of a late
failure is not the exception — it is the four hours of training that preceded
it.

**Error messages state three things:** what was expected, what was received,
and what to do.

```python
raise SpecError(
    f"split fractions must sum to less than 1.0, received "
    f"validation={validation_fraction} + test={test_fraction} "
    f"= {validation_fraction + test_fraction}; reduce one of them"
)
```

---

## 7. Purity and side effects

The layering is enforced by test. These rules are the finer-grained version,
enforced by review.

| Rule | Reason |
| --- | --- |
| `core` imports only stdlib, `pydantic`, `numpy` | A worker that only parses a spec must not import PyTorch. Enforced by test. |
| A spec holds no state, handles or tensors | So it can be hashed, logged and pickled to a worker. |
| A metric is a pure function of arrays | So it is testable against a hand-computed value. |
| A visual returns a figure and writes nothing | So the same function serves a report, a notebook, a dashboard and a test. |
| Only `reports` and `storage` write to a run directory | So "what did this run produce?" has one answer. |
| No module-level mutable state | Two jobs in one process must not interfere. |
| No global library configuration at import | Importing `rade_qnet` must not change NumPy's print options or Matplotlib's backend. |

---

## 8. Imports

**Relative inside the package, absolute outside.**

```python
from ..core.contract import DataBundle    # inside rade_qnet
from ...core.runtime.errors import StageError
```

This is what keeps the package **relocatable**: it behaves identically imported
as `src.rade_qnet` from this repository or as `rade_qnet` from an installed
distribution. Absolute self-imports would pin it to one of those and break the
other. (`TID252`, which bans parent-relative imports, is therefore switched off
with that reasoning recorded in the config.)

Order is handled by Ruff's `I` rules: future, standard library, third party,
first party, local — each group separated by a blank line.

**No star imports, and no imports for side effects** other than registry
population, which happens in exactly one place per model package and is
commented as such.

---

## 9. Tests

Tests are specifications, and are held to the same standard as the code. The
only rules relaxed in the test tree are argument annotations, literal
comparisons and unused fixtures — see the test `ruff.toml` for the reasoning.
**Docstrings stay mandatory**, because a test name plus its docstring is how a
failure gets diagnosed from a CI log by someone who did not write it.

| Rule | Reason |
| --- | --- |
| Assert against hand-computed values, not a second implementation | Two implementations of a wrong formula agree. |
| One behaviour per test | A test asserting six things reports one failure and hides five. |
| Deterministic: fixed seeds, no clock, no network, no shared directory | A flaky test gets re-run until it passes, then ignored. |
| Test the failure path, not just the happy path | Most of these components exist *because* of a failure mode. |
| Protocol implementations go through one shared contract suite | So "add a new source" stays a safe operation. |
| Mark and skip hardware-dependent tests, never silently omit them | An untested GPU path should be visible, not absent. |
| Name files `test_<package>_<module>.py` | Matches this repository's existing suites. |

Every test package's `__init__.py` lists its planned modules and the phase that
delivers them, so the suite doubles as a build checklist.

---

## 10. Review checklist

Mechanical:

- [ ] `ruff check` reports zero findings.
- [ ] `ruff format --check` reports zero files needing change.
- [ ] `pytest tests/rade_qnet` passes.
- [ ] New packages have a charter; new modules have a docstring.
- [ ] `test_scaffold.py` still passes — layering and mirroring intact.

Judgement:

- [ ] Every public signature is fully annotated, with no bare `Any`.
- [ ] Docstrings state the contract, not the implementation.
- [ ] Comments explain *why*; none restates its line.
- [ ] Names need no decoding; domain vocabulary is used exactly.
- [ ] No exception is swallowed; framework error types are raised.
- [ ] Failures happen as early as they can be detected.
- [ ] Pure things stayed pure; nothing new writes to disk outside `reports`
      and `storage`.
- [ ] Tests assert hand-computed values and cover the failure path.
- [ ] The phase's definition of done is met in full.
