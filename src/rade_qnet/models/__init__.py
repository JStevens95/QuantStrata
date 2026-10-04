"""
The model library, and the file convention every model in it obeys.

One shape, four tiers
---------------------
Every model is a **package** directly under this one. There are no loose
modules here and no category sub-folders: ``rade_qnet.models.<name>`` is a
model, always, and that is the whole addressing rule.

Every model package contains the same five files, whatever its size.  Three
are the same whatever it learns; the other two name the paradigm::

    models/<name>/
        __init__.py     the charter: what this model is, and its tier
        spec.py         what can be configured
        register.py     how it plugs in

        model.py        what is computed            } a supervised model
        data.py         where the data comes from   }

        policy.py       what is computed            } an agent
        environment.py  what it acts in             }

Two shapes rather than one is a deliberate cost.  Identical filenames across
every package would be tidier, and would mean a file called ``data.py``
holding an environment -- a name that lies, in the one directory a new joiner
is told to copy.  A reader can tell what a package learns from ``ls``, and
that is worth more than the symmetry.  ``spec.py`` and ``register.py`` stay
constant because "what can be configured" and "how does it plug in" have the
same answer for a ridge regression and for a hedging agent.

Files are then *added* as the model needs them, never renamed and never
rearranged::

    tier 2    + state.py        a fitted artefact beyond the engine's own
    tier 3    + layers/         when model.py outgrows one file
              + features/       model-specific feature construction
    tier 4    + pipelines/      train.py, eval.py, tune.py stage overrides

and two optional files at any tier: ``reports.py`` and ``visuals.py``, for
artefacts only this model can produce.

The layout is enforced by ``tests/rade_qnet/models/test_model_layout.py``, so
a file named anything else is a failing test rather than a convention
somebody did not know about.

Why the smallest model pays the ceremony too
---------------------------------------------
Ridge is about thirty statements and could be one file. Making it three
costs a reader perhaps a minute and buys four things:

*One procedure.* There is no "is this simple enough to be one file?"
judgement call, so there is no boundary to argue about and no review where
the answer differs from last time.

*A data contract that is declared rather than discovered.* ``data.py``
exports a ``REQUIRES`` naming the inputs the model consumes and their
ranks, and the pipeline checks it against the data build before the model
is constructed. That is the one point in a run where information flows
model to data; everywhere else the build declares what it made and the
model copes, and coping is what turns a wrong data build into a plausible
loss curve rather than an error. See
:mod:`rade_qnet.core.contract.requirement` for the two defects that
motivated it.

*No migration.* A model that grows a fitted state adds ``state.py``. It
does not get taken apart first. The commit that adds capability contains
only the capability, which is the difference between a reviewable diff and
an unreviewable one.

*Mathematics you can read without the framework.* ``model.py`` imports no
registry and no run specification -- a rule the layout test checks. A quant
reviewing the model opens one file and finds only the model; a platform
engineer opens ``register.py`` and finds only wiring. Neither skims the
other's half.

*Unit tests that are actually unit tests.* ``build(settings)`` can be
called directly with no run spec, no registry and no engine, because
nothing in ``model.py`` needs them.

The cost is three short files instead of one. The thing it buys is that the
answer to "where do I put this?" is never a matter of taste.

Where to start
--------------
Copy :mod:`rade_qnet.models.ridge`. It is the tier 1 template with nothing
added. Its ``data.py`` returns the framework's own table module, which is
what most models start with and almost none finish with -- the moment the
data lives in a store rather than a CSV, that function returns your
``DataModule`` subclass instead and nothing else in the package moves. Then follow ``docs/MODEL_IMPLEMENTATION.md``, which is the procedure
this convention exists to make possible.

Sub-packages
------------
``ridge``
    Tier 1. A library estimator; the template, and the measure of whether
    the framework stays cheap for small models.
``xgb_tabular``
    Tier 1. Boosted trees -- the model most likely to beat the flagship.
``lstm_tabular``
    Tier 1. The flagship's temporal stream alone, which is what makes the
    graph's contribution measurable rather than assumed.
``hybrid_gnn_rnn``
    Tier 4. The flagship: a graph-temporal network for P&L replication,
    run as a single member or fanned out across a job set.

Dependency rule
---------------
May import: any framework package. Nothing in the framework may import this
package -- models are discovered through the registry, so the framework
never depends on the library it serves.
"""

__all__: tuple[str, ...] = ()
