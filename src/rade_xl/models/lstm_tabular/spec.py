"""What the recurrence needs to know about itself."""

from __future__ import annotations

from pydantic import Field

from ...core.spec.base import Spec

__all__ = ["LstmTabularSpec"]


class LstmTabularSpec(Spec):
    """
    Architecture settings for the recurrent baseline.

    Parameters
    ----------
    units
        Hidden width of the recurrence.
    layers
        How many stacked recurrent layers.
    dropout
        Dropout between layers. Ignored by PyTorch when ``layers`` is one,
        which is its own behaviour rather than something to work around.
    """

    units: int = Field(default=32, ge=1)
    layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
