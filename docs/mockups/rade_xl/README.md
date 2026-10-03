# rade_xl interface mockups

Three static HTML pages that show what a `rade_xl` front end could look like.
They are **mockups, not working software**: every number is fabricated, nothing
calls into `src/rade_xl`, and no control does anything when you click it. Their
job is to settle what the real thing should show before anyone writes the
backend for it.

Open any of them directly in a browser — they have no dependencies, no CDN
links and no JavaScript.

| File | What it proposes | Build cost |
| --- | --- | --- |
| `model_lab.html` | A teaching surface. One page per model family explaining what that family of models actually does, with a live playground on a toy dataset. | Medium — needs a small server to refit on parameter change. |
| `run_report.html` | The artifact a training run writes to its own run directory. Self-contained, archivable, diffable. | Low — a Jinja template plus the metrics the pipeline already computes. |
| `console.html` | A UI for composing a job set and launching train/evaluate, with live progress. | High — needs a job runner, a progress channel and auth. |

## Regenerating the screenshots

```bash
.venv/bin/pip install playwright && .venv/bin/playwright install chromium
.venv/bin/python docs/mockups/rade_xl/shoot.py
```

Output lands in `shots/` at 1512 CSS pixels wide and 2x device scale, which is
large enough to drop into a slide without resampling artefacts.

Playwright is deliberately **not** in any `requirements*.txt`. It exists only to
photograph these mockups and is not a dependency of the framework.

## Where the chart data came from

The SVG paths in these pages are not hand-drawn. They were generated from
seeded random data so the shapes are statistically honest — the tree fit really
is the piecewise-constant least-squares fit to those sixty points, and the
residual histogram really is drawn from the distribution its summary statistics
claim. Fake numbers that are internally inconsistent make a mockup harder to
reason about, not easier.
