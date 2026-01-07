from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

Qualifier = Tuple[str, str]


@dataclass(frozen=True, slots=True)
class MarketId:
    """
    Canonical identifier for any market object used across QuantStrata.

    Canonical string format
    -----------------------
        ASSET_CLASS.TYPE.NAME|k1=v1|k2=v2|...

    Examples
    --------
    - FX.SPOT.EURUSD
    - IR.CURVE.USD.OIS
    - FX.VOL.EURUSD|cut=NY|convention=delta25
    """
    asset_class: str
    mkt_type: str
    name: str
    qualifiers: Tuple[Qualifier, ...] = ()

    def __post_init__(self) -> None:
        # Normalize canonical fields (this keeps keys stable across the library).
        object.__setattr__(self, "asset_class", str(self.asset_class).strip().upper())
        object.__setattr__(self, "mkt_type", str(self.mkt_type).strip().upper())
        object.__setattr__(self, "name", str(self.name).strip())

        # Validate required parts are non-empty.
        if not self.asset_class:
            raise ValueError("MarketId.asset_class must not be empty.")
        if not self.mkt_type:
            raise ValueError("MarketId.mkt_type must not be empty.")
        if not self.name:
            raise ValueError("MarketId.name must not be empty.")

        # Normalize and validate qualifiers (stable ordering is preserved as provided).
        normalized: List[Qualifier] = []
        for k, v in self.qualifiers:  # <-- FIXED BUG (removed stray comma)
            kk = str(k).strip()
            vv = str(v).strip()
            if not kk:
                raise ValueError("MarketId qualifier key must be non-empty.")
            normalized.append((kk, vv))

        object.__setattr__(self, "qualifiers", tuple(normalized))

    def key(self) -> str:
        """Return canonical string key suitable for caching/logging/serialization."""
        base = f"{self.asset_class}.{self.mkt_type}.{self.name}"
        if not self.qualifiers:
            return base
        # Join qualifiers in the stored order for stable reproducibility.
        qual = "|".join(f"{k}={v}" for k, v in self.qualifiers)
        return f"{base}|{qual}"

    @staticmethod
    def parse(text: str) -> "MarketId":
        """Parse a canonical MarketId string back into a MarketId."""
        raw = str(text).strip()
        if not raw:
            raise ValueError("Cannot parse empty MarketId string.")

        # Split base from qualifiers.
        base, *qual_parts = raw.split("|")

        # Base must be exactly ASSET.TYPE.NAME (NAME may contain dots, so split max=2).
        parts = base.split(".", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid MarketId {text!r}. Expected 'ASSET.TYPE.NAME|k=v|...' but got base={base!r}."
            )

        asset_class, mkt_type, name = parts

        qualifiers: List[Qualifier] = []
        for qp in qual_parts:
            if not qp:
                continue
            if "=" not in qp:
                raise ValueError(f"Invalid qualifier {qp!r} in MarketId {text!r}. Expected 'k=v'.")
            k, v = qp.split("=", 1)
            qualifiers.append((k, v))

        return MarketId(asset_class=asset_class, mkt_type=mkt_type, name=name, qualifiers=tuple(qualifiers))

    def with_qualifiers(self, qualifiers: Iterable[Qualifier]) -> "MarketId":
        """Return a new MarketId with additional qualifiers appended (order preserved)."""
        return MarketId(
            asset_class=self.asset_class,
            mkt_type=self.mkt_type,
            name=self.name,
            qualifiers=self.qualifiers + tuple(qualifiers),
        )

    def with_qualifier(self, key: str, value: str) -> "MarketId":
        """Convenience helper for adding a single qualifier."""
        return self.with_qualifiers([(key, value)])