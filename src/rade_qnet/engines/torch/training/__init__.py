"""
How a fit is executed: the drivers, and everything that watches them.

The split from ``learners``
---------------------------
A learner decides *what one update means*.  This package decides *when* an
update happens, when to validate, when to write a checkpoint and when to
stop.  That is the whole of the Phase 2 loop/learner split, and the reason it
is worth a package boundary is that the two change for entirely different
reasons: a new algorithm adds a learner and touches nothing here, while a new
stopping rule changes ``callbacks.py`` and touches no algorithm.

Why this is not a framework-level package
------------------------------------------
Every module here is irreducibly PyTorch.  The loop calls ``optimiser.step``,
the callbacks save ``state_dict`` objects and drive
``torch.optim.lr_scheduler``, the losses are ``torch.nn`` modules.  The part
that genuinely *is* library-agnostic was hoisted long ago and lives in
:mod:`rade_qnet.core.contract.result` as ``FitOutcome`` and ``EpochRecord`` --
which is why the xgboost engine, whose fit is a single call with no loop at
all, still produces the same history type as a hundred-epoch gradient run.

Modules
-------
``loops.py``
    The two drivers.  ``fit_epochs`` makes passes over a finite dataset;
    ``fit_steps`` spends a budget of interaction against an unbounded one.
    Also the two learner protocols, ``Learner`` and ``PolicyLearner``.
``callbacks.py``
    Observation and intervention between epochs: ``EarlyStopping``,
    ``BestCheckpoint``, ``LearningRateSchedule``, ``GradientNorms``, and the
    ``EpochContext`` they all read.
``losses.py``
    Objective functions, resolved by name from a training specification.
``checkpoint.py``
    Reading and writing weights, so a run can be resumed or a bundle loaded.
"""

__all__: tuple[str, ...] = ()
