"""
Understanding a run.

``analysis`` is split three ways, and the split between the last two is the one
that usually gets lost in research code:

``metrics``
    Numbers computed from predictions and targets.  Pure functions.
``visuals``
    Figure factories.  A visual takes data and returns a figure object.  It
    never writes a file, never reads configuration and never touches the run
    context.
``reports``
    Writers.  A report decides *what* to produce for a run, calls visuals and
    metrics, and persists the output.

Because visuals are pure, the same function serves a saved report, a notebook,
a dashboard and a test.  Plotting code that writes files as a side effect
cannot be reused anywhere, which is why the two are kept apart.

Dependency rule
---------------
May import: ``core``.
May not import: ``engines``, ``orchestration``, ``models``.  Analysis receives
plain arrays, never a live model.
"""

__all__: tuple[str, ...] = ()
