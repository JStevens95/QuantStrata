"""
Elementary instrument definition.

An :class:`ElementaryTrade` is one member of the replicating basis. It is
asset-class-agnostic on purpose: the asset plugin's generator decides *which*
trades to emit and its pricer knows how to read ``parameters``. The ``trade_id`` is
``|``-delimited so the downstream consumer can recover the underlying via
``id.split("|")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ElementaryTrade:
    """One elementary (replicating-basis) instrument.

    Parameters
    ----------
    trade_id : str
        Unique id, e.g. ``"FX.SPOT.USD.GBP|CALL|K=1.25|T=0.50"``.
    factor_id : str
        Risk factor this instrument is priced against.
    asset_class : str
        Pricer routing key.
    payoff_type : str
        Instrument type (see :class:`~...domain.enums.PayoffType`).
    parameters : dict
        Pricer inputs (must include ``expiry`` in years).
    notional : float
        Unit notional by default — the model learns replicating weights.
    """
    trade_id: str
    factor_id: str
    asset_class: str
    payoff_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    notional: float = 1.0
