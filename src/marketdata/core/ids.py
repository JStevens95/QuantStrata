from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple, Union

Qualifier = Tuple[str, str]
QualifiersInput = Union[None, Mapping[str, str], Iterable[Qualifier]]


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
    qualifiers: QualifiersInput = None  # Option A: allow None / dict / iterable of pairs

    def __post_init__(self) -> None:
        # Normalize canonical fields
        object.__setattr__(self, "asset_class", str(self.asset_class).strip().upper())
        object.__setattr__(self, "mkt_type", str(self.mkt_type).strip().upper())
        object.__setattr__(self, "name", str(self.name).strip())

        if not self.asset_class:
            raise ValueError("MarketId.asset_class must not be empty.")
        if not self.mkt_type:
            raise ValueError("MarketId.mkt_type must not be empty.")
        if not self.name:
            raise ValueError("MarketId.name must not be empty.")

        raw = self.qualifiers

        # Guard against a common bug: passing a string (iterable of chars)
        if isinstance(raw, str):
            raise ValueError("MarketId.qualifiers must be None, a mapping, or an iterable of (k,v) pairs (not str).")

        # Normalize qualifiers into an iterable of (k,v)
        if raw is None:
            items: Iterable[Qualifier] = ()
        elif isinstance(raw, Mapping):
            items = raw.items()
        else:
            items = raw

        normalized: List[Qualifier] = []
        for item in items:
            try:
                k, v = item
            except Exception as exc:
                raise ValueError(
                    "MarketId.qualifiers must be None, a mapping, or an iterable of (key,value) pairs."
                ) from exc

            kk = str(k).strip()
            vv = str(v).strip()
            if not kk:
                raise ValueError("MarketId qualifier key must be non-empty.")
            normalized.append((kk, vv))

        # Internally store as a tuple for immutability + stable key() output
        object.__setattr__(self, "qualifiers", tuple(normalized))

    def key(self) -> str:
        base = f"{self.asset_class}.{self.mkt_type}.{self.name}"
        quals = self.qualifiers  # now guaranteed to be tuple[(k,v),...]
        if not quals:
            return base
        qual = "|".join(f"{k}={v}" for k, v in quals)
        return f"{base}|{qual}"

    @staticmethod
    def parse(text: str) -> "MarketId":
        raw = str(text).strip()
        if not raw:
            raise ValueError("Cannot parse empty MarketId string.")

        base, *qual_parts = raw.split("|")
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

    def with_qualifiers(self, qualifiers: QualifiersInput) -> "MarketId":
        current = self.qualifiers  # tuple of pairs
        if qualifiers is None:
            extra: Tuple[Qualifier, ...] = ()
        elif isinstance(qualifiers, Mapping):
            extra = tuple((str(k), str(v)) for k, v in qualifiers.items())
        else:
            extra = tuple((str(k), str(v)) for k, v in qualifiers)
        return MarketId(
            asset_class=self.asset_class,
            mkt_type=self.mkt_type,
            name=self.name,
            qualifiers=current + extra,
        )

    def with_qualifier(self, key: str, value: str) -> "MarketId":
        return self.with_qualifiers([(key, value)])