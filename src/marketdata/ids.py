from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

Qualifier = Tuple[str, str]


@dataclass(frozen=True, slots=True)
class MarketId:
    """
    Canonical identifier for any market object or quote used across QuantStrata.

    Design goals
    ------------
    1) One consistent key across asset classes and data sources.
    2) Stable string representation suitable for caching, logging, and routing.
    3) Extensible via qualifiers without breaking existing IDs.

    Format
    ------
    The canonical string key is:

        ASSET_CLASS.TYPE.NAME|k1=v1|k2=v2|...

    Examples
    --------
    - FX.SPOT.EURUSD
    - EQ.SPOT.AAPL
    - IR.CURVE.USD.OIS
    - FX.VOL.EURUSD
    - IR.FIXING.USD.SOFR|tenor=3M
    - EQ.DIV.AAPL|type=continuous

    Notes
    -----
    - `asset_class` and `type` are normalized to uppercase.
    - `name` is kept as provided (some names are case-sensitive or dotted).
    - `qualifiers` are stored as an ordered tuple for immutability and stable keys.
    """
    # initiate required variables.
    asset_class: str                                # "FX". "EQ". "IR", "CR"
    mkt_type: str                                   # "SPOT", "CURVE", "VOL", "FIXING"
    name: str                                       # "EURUSD", "AAPL", "USD.OIS", "SPX"
    qualifiers: Tuple[Tuple[str, str], ...] = ()    # optional (key, value)

    def __post_init__(self) -> None:
        # normalise only the fields that should be canonical across the library.
        object.__setattr__(self, "asset_class", self.asset_class.strip().upper())
        object.__setattr__(self, "mkt_type", self.mkt_type.strip().upper())
        object.__setattr__(self, "name", self.name.strip())

        # ensure qualifiers are stable and well-formed.
        normalised: List[Qualifier] = []
        for k, v, in self.qualifiers:
            k_k = str(k).strip()
            v_v = str(v).strip()
            if not k_k:
                raise ValueError("MarketId qualifier key must be non-empty.")
            normalised.append((k_k, v_v))
        object.__setattr__(self, "qualifiers", tuple(normalised))

        if not self.asset_class:
            raise ValueError("MarketId.asset_class must not be empty.")
        if not self.mkt_type:
            raise ValueError("MarketId.type must not be empty.")
        if not self.name:
            raise ValueError("MarketId.name must not be empty.")

    def key(self) -> str:
        """
        Return the canonical key for this market object.

        This is the preferred value to use for:
        - cache keys
        - logging
        - serialization
        """
        base = f"{self.asset_class}.{self.mkt_type}.{self.name}"
        if not self.qualifiers:
            return base
        qual = "|".join(f"{k}={v}" for k, v in self.qualifiers)
        return f"{base}|{qual}"

    @staticmethod
    def parse(text: str) -> "MarketId":
        """
        Parse a canonical MarketId string.

        :param text: string in the format 'ASSET.KIND.NAME|k=v|...'
        :return:
        """
        raw = text.strip()
        if not raw:
            raise ValueError("Cannot parse empty MarketId string.")

        base, *qual_parts = raw.split("|")
        parts = base.split(".", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid MarketId '{text}'. Expected 'ASSET.TYPE.NAME|k=v|...' but got '{base}'.")

        asset_class, mkt_type, name = parts
        qualifiers: List[Qualifier] = []
        for qp in qual_parts:
            if not qp:
                continue
            if "=" not in qp:
                raise ValueError(f"Invalid qualifier '{qp}' in MarketId '{text}'. Expected k=v.")
            k, v = qp.split("=", 1)
            qualifiers.append((k, v))
        return MarketId(asset_class=asset_class, mkt_type=mkt_type, name=name, qualifiers=tuple(qualifiers))

    def with_qualifiers(self, qualifier: Iterable[Qualifier]) -> "MarketId":
        """Return a new MarketId with the provided qualifiers appended."""
        return MarketId(
            asset_class=self.asset_class, mkt_type=self.mkt_type, name=self.name,
            qualifiers=self.qualifiers + tuple(qualifier)
        )

    def with_qualifier(self, key: str, value: str) -> "MarketId":
        """Convenience method for adding a single qualifier."""
        return self.with_qualifiers([(key, value)])
