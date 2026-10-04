"""
The base class every specification derives from.

Three settings are applied to every spec in the framework, and each one exists
because of a specific way configuration goes wrong.

``extra="forbid"``
    A misspelled key is a load-time error. The alternative -- silently
    ignoring it -- means a run that was configured with ``seq_len: 20``
    trains with the default of ``1``, reports plausible numbers, and nobody
    finds out.

``frozen=True``
    A spec cannot be mutated after validation. A pipeline that received a spec
    and a pipeline that recorded it in a bundle must have received the same
    object, or the bundle is a record of something that did not happen.

``str_strip_whitespace=True``
    A trailing space in a YAML string is invisible in a text editor and fatal
    to a registry lookup.

Specs describe *intent only*. They hold no fitted state, no file handles, no
tensors and no live objects, which is what keeps them cheap to hash, log and
send across a process boundary.

How validation failures surface
-------------------------------
Validators in this package raise :class:`~rade_qnet.core.lifecycle.errors.SpecError`.
Pydantic *collects* that rather than letting it propagate, because
``SpecError`` subclasses ``ValueError`` -- which is the entire reason for that
dual inheritance. The practical consequence is a deliberate two-level policy:

- **Direct construction** of a spec in Python raises ``ValidationError``,
  carrying the ``SpecError`` message and the dotted path of the offending
  field. This is the right outcome: constructing an invalid spec in code is a
  programming error, and pydantic's report locates it precisely.
- **Parsing** untrusted input goes through
  :func:`~rade_qnet.core.spec.run.parse_run_spec`, which reformats the
  ``ValidationError`` into a single ``SpecError``. A user who mistyped a YAML
  key sees one framework error, not a pydantic traceback.

So the ``Raises`` sections below name ``ValidationError``, and the ``SpecError``
mentioned alongside it is the message the user ends up reading.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["Spec"]


class Spec(BaseModel):
    """
    Base class for every rade_qnet specification.

    Subclasses add fields and validation; they should not override
    :attr:`model_config`. If a spec appears to need ``extra="allow"``, the
    open-ended part of it belongs in an explicit ``params`` mapping whose
    contents are validated by whoever consumes them -- usually a model's own
    spec class.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        # Validate on construction as well as on parse, so a spec built in
        # Python is held to the same rules as one loaded from YAML. Without
        # this, a test that constructs a spec directly would not exercise the
        # validation that production configuration relies on.
        validate_default=True,
        # Nothing in a spec may be an arbitrary live object; see the module
        # docstring. Leaving this false makes that structural.
        arbitrary_types_allowed=False,
    )
