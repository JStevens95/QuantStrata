"""
How a run is assembled, and how somebody changes it without forking it.

Five modules, and they are the whole extension model.  A specification names
a model as a string; something has to turn that string into a class.  A
pipeline is a sequence of stages; something has to define what a stage is and
what it may reach.  A user wants to watch a run without subclassing it;
something has to give them a place to stand.

This package is what a contributor reads when the question is *how do I plug
something in*.  Its sibling :mod:`rade_qnet.core.provenance` is what somebody
reads when the question is *can you prove this number*.  Those are different
people on different days, which is why the eight modules that used to share a
``runtime`` package are now two packages of five and three.

The four customisation tiers, and where each one lives
-------------------------------------------------------
A model should use the lowest tier that works, and the tiers only stay
distinct because the mechanisms are distinct:

1. **Spec only.**  Write no code.  ``components.py`` resolves the names.
2. **Observe.**  Attach a hook or enable a report.  ``hooks.py``.
3. **Override one stage.**  Subclass a pipeline, replace one method.
   ``pipeline.py`` is what makes a stage small enough to be worth replacing.
4. **Override ``run()``.**  Reserved for genuinely different sequences.

Conflating 2 and 3 is the failure this split prevents.  Without hooks, a user
who wants to log something subclasses a pipeline and overrides a stage to add
a print statement -- an override whose only purpose is observation, which then
silently stops matching the base implementation it copied.

Modules
-------
``registry.py``
    ``Registry[T]`` and ``RegistryEntry`` -- the generic container, which
    knows how to hold named classes and refuse a duplicate and nothing about
    what a model is.
``components.py``
    The four concrete registries (models, engines, learners, reports), the
    decorators that populate them, and the getters that read them.  Also
    ``registration_modules`` and ``import_registrations``, which are how a
    worker process replays the imports that made a name resolvable -- the
    defect that made a job set work from one entry point and fail from
    another.
``pipeline.py``
    The template-method base every pipeline derives from.
``context.py``
    ``RunContext``: the ambient state every stage can reach -- where to
    write, which run, which seed, who is observing.  Threading those through
    every signature would make each stage's parameters mostly plumbing;
    module globals would make two concurrent runs in one process impossible.
``errors.py``
    The error hierarchy.  Every framework error carries the one thing a bare
    ``ValueError`` cannot: whose fault it is.  A ``SpecError`` means the user
    can fix it by editing their configuration; a ``ContractError`` means the
    framework or a model broke an internal promise.  It sits here rather than
    in ``provenance`` because an error is raised *by a stage*, and because
    every module in the framework imports it.
"""

__all__: tuple[str, ...] = ()
