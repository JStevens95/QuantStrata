"""
Where the computation runs, and whether it runs the same way twice.

Nothing in this package knows what is being trained.  It answers three
questions a model author should never have to ask: which device is available,
how work is split across more than one of them, and what has to be pinned so
that the same specification produces the same numbers tomorrow.

Why determinism belongs with hardware
--------------------------------------
``determinism.py`` looks like it should sit beside the framework's own
seeding, and it does not, because what it seeds is the hardware: CUDA
generators, cuDNN algorithm selection, the choice between a fast
non-deterministic kernel and a slower reproducible one.  Those are device
facts, not framework facts.  :mod:`rade_qnet.core.provenance.seeding` holds
the library-agnostic half and calls into this one through a registered
callback, which is how ``core`` manages to seed PyTorch without importing it.

Modules
-------
``devices.py``
    Resolving a hardware specification against what is actually present:
    CUDA, MPS or CPU, with mixed precision and memory settings.
``distributed.py``
    Multi-process training, and putting the model back into a shape that can
    be saved once the processes are done with it.
``determinism.py``
    Seeding PyTorch, and the registration that lets ``core`` ask for it.
"""

__all__: tuple[str, ...] = ()
