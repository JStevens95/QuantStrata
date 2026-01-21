from __future__ import annotations

import json
import os

import numpy as np
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel

from src.marketdata.curves.factory import FlatRateCurveFactory, ZeroRateCurveFactory
from src.marketdata.surfaces.factory import FlatVolFactory, GridVolFactory


# =============================================================================
# Public API
# =============================================================================

def save_market_dataset(
    dataset: MarketDataset,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save a MarketDataset to disk as a directory artifact.

    Artifact format (V1)
    --------------------
    <path>/
      manifest.json
      panels/<market_id_key>.npy
      curve_params/<market_id_key>.npy
      vol_params/<market_id_key>.npy

    Design goals
    ------------
    - Fast & deterministic.
    - Explicit, human-readable manifest.
    - Robust round-trip including curve/vol factories.
    - No pickle (avoid arbitrary code execution risks).

    Parameters
    ----------
    dataset:
        Dataset to persist.
    path:
        Directory path for the artifact.
    overwrite:
        If False, refuse to overwrite an existing directory.

    Returns
    -------
    Path
        The directory path that was written.
    """
    root = Path(path).expanduser().resolve()

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Artifact path already exists: {root}")
        if root.is_file():
            raise FileExistsError(f"Artifact path is a file, expected directory: {root}")
        # Clear directory contents deterministically.
        _remove_tree(root)

    root.mkdir(parents=True, exist_ok=True)

    # Create subfolders
    (root / "panels").mkdir(parents=True, exist_ok=True)
    (root / "curve_params").mkdir(parents=True, exist_ok=True)
    (root / "vol_params").mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "format": "QuantStrata.MarketDatasetArtifact",
        "version": "v1",
        "dates": list(dataset.dates),
        "n_scenarios": int(dataset.n_scenarios),
        "meta": dict(dataset.meta or {}),
        "panels": {},
        "curve_params": {},
        "vol_params": {},
        "curve_factories": {},
        "vol_factories": {},
    }

    # ---- Save quote panels ----
    for mid, panel in dataset.panels.items():
        key = mid.key()
        filename = _safe_filename_from_key(key) + ".npy"
        rel = str(Path("panels") / filename)
        abs_path = root / rel
        np.save(abs_path, np.asarray(panel.data), allow_pickle=False)

        manifest["panels"][key] = {
            "axis_names": list(panel.axis_names),
            "file": rel,
        }

    # ---- Save curve params + factories ----
    for mid, panel in dataset.curve_params.items():
        key = mid.key()
        filename = _safe_filename_from_key(key) + ".npy"
        rel = str(Path("curve_params") / filename)
        abs_path = root / rel
        np.save(abs_path, np.asarray(panel.data), allow_pickle=False)

        manifest["curve_params"][key] = {
            "axis_names": list(panel.axis_names),
            "file": rel,
        }

        factory = dataset.curve_factories.get(mid)
        if factory is None:
            raise ValueError(f"Missing curve factory for MarketId {key}.")
        manifest["curve_factories"][key] = _serialize_factory(factory)

    # ---- Save vol params + factories ----
    for mid, panel in dataset.vol_params.items():
        key = mid.key()
        filename = _safe_filename_from_key(key) + ".npy"
        rel = str(Path("vol_params") / filename)
        abs_path = root / rel
        np.save(abs_path, np.asarray(panel.data), allow_pickle=False)

        manifest["vol_params"][key] = {
            "axis_names": list(panel.axis_names),
            "file": rel,
        }

        factory = dataset.vol_factories.get(mid)
        if factory is None:
            raise ValueError(f"Missing vol factory for MarketId {key}.")
        manifest["vol_factories"][key] = _serialize_factory(factory)

    # Write manifest last (atomic-ish): write temp then replace.
    manifest_path = root / "manifest.json"
    tmp_path = root / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(manifest_path)

    return root


def load_market_dataset(path: str | Path) -> MarketDataset:
    """
    Load a MarketDataset artifact written by save_market_dataset(...).

    Parameters
    ----------
    path:
        Directory containing manifest.json and array files.

    Returns
    -------
    MarketDataset
        Reconstructed dataset including factories.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in artifact directory: {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validate_manifest_header(manifest)

    dates = list(manifest["dates"])
    n_scenarios = int(manifest["n_scenarios"])
    meta = dict(manifest.get("meta") or {})

    panels: Dict[MarketId, Panel] = {}
    curve_params: Dict[MarketId, Panel] = {}
    vol_params: Dict[MarketId, Panel] = {}

    curve_factories: Dict[MarketId, object] = {}
    vol_factories: Dict[MarketId, object] = {}

    # ---- Load quote panels ----
    for key, spec in (manifest.get("panels") or {}).items():
        mid = MarketId.parse(key)
        file_path = _resolve_artifact_file(root=root, rel=str(spec["file"]), expected_dir="panels")
        arr = np.load(file_path, allow_pickle=False)
        panels[mid] = Panel(data=np.asarray(arr), axis_names=tuple(spec["axis_names"]))

    # ---- Load curve params ----
    for key, spec in (manifest.get("curve_params") or {}).items():
        mid = MarketId.parse(key)
        file_path = _resolve_artifact_file(root=root, rel=str(spec["file"]), expected_dir="curve_params")
        arr = np.load(file_path, allow_pickle=False)
        curve_params[mid] = Panel(data=np.asarray(arr), axis_names=tuple(spec["axis_names"]))

    # ---- Load vol params ----
    for key, spec in (manifest.get("vol_params") or {}).items():
        mid = MarketId.parse(key)
        file_path = _resolve_artifact_file(root=root, rel=str(spec["file"]), expected_dir="vol_params")
        arr = np.load(file_path, allow_pickle=False)
        vol_params[mid] = Panel(data=np.asarray(arr), axis_names=tuple(spec["axis_names"]))

    # ---- Load factories ----
    for key, fser in (manifest.get("curve_factories") or {}).items():
        mid = MarketId.parse(key)
        curve_factories[mid] = _deserialize_factory(fser)

    for key, fser in (manifest.get("vol_factories") or {}).items():
        mid = MarketId.parse(key)
        vol_factories[mid] = _deserialize_factory(fser)

    # MarketDataset will validate "every param panel has a factory"
    return MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels=panels,
        curve_params=curve_params,
        curve_factories=curve_factories,  # type: ignore[arg-type]
        vol_params=vol_params,
        vol_factories=vol_factories,      # type: ignore[arg-type]
        meta=meta,
    )


# =============================================================================
# Factory serialization (explicit allow-list)
# =============================================================================

def _serialize_factory(factory: object) -> Dict[str, Any]:
    """
    Serialize a known factory into a JSON-friendly dict.

    IMPORTANT
    ---------
    We intentionally restrict to a known allow-list to keep artifacts safe and stable.
    """
    if isinstance(factory, ZeroRateCurveFactory):
        return {
            "type": "ZeroRateCurveFactory",
            "params": {
                "extrapolation": str(factory.extrapolation),
            },
        }

    if isinstance(factory, FlatRateCurveFactory):
        return {
            "type": "FlatRateCurveFactory",
            "params": {},
        }

    if isinstance(factory, GridVolFactory):
        return {
            "type": "GridVolFactory",
            "params": {
                "expiries": np.asarray(factory.expiries, dtype=float).reshape(-1).tolist(),
                "strikes": np.asarray(factory.strikes, dtype=float).reshape(-1).tolist(),
                "extrapolation": str(factory.extrapolation),
            },
        }

    if isinstance(factory, FlatVolFactory):
        return {
            "type": "FlatVolFactory",
            "params": {},
        }

    # Dataclass fallback (only if you *want* it later). Keep disabled by default.
    if is_dataclass(factory):
        raise TypeError(
            "Unsupported factory dataclass type for artifact serialization.\n"
            f"  type={type(factory).__name__}\n"
            "Add it explicitly to _serialize_factory/_deserialize_factory allow-list."
        )

    raise TypeError(
        "Unsupported factory type for artifact serialization.\n"
        f"  type={type(factory).__name__}\n"
        "Add it explicitly to _serialize_factory/_deserialize_factory allow-list."
    )


def _deserialize_factory(payload: Mapping[str, Any]) -> object:
    """
    Reconstruct a factory from the serialized representation produced by _serialize_factory.
    """
    ftype = str(payload.get("type", "")).strip()
    params = dict(payload.get("params") or {})

    if ftype == "ZeroRateCurveFactory":
        return ZeroRateCurveFactory(extrapolation=str(params.get("extrapolation", "flat")))

    if ftype == "FlatRateCurveFactory":
        return FlatRateCurveFactory()

    if ftype == "GridVolFactory":
        expiries = np.asarray(params.get("expiries", []), dtype=float)
        strikes = np.asarray(params.get("strikes", []), dtype=float)
        extrapolation = str(params.get("extrapolation", "flat"))
        return GridVolFactory(expiries=expiries, strikes=strikes, extrapolation=extrapolation)

    if ftype == "FlatVolFactory":
        return FlatVolFactory()

    raise TypeError(
        "Unsupported factory type in artifact.\n"
        f"  type={ftype!r}\n"
        "Add it explicitly to _deserialize_factory allow-list."
    )


# =============================================================================
# Manifest validation + small utilities
# =============================================================================
def _resolve_artifact_file(*, root: Path, rel: str, expected_dir: str) -> Path:
    """
    Resolve a manifest-relative file path safely.

    Security:
    - Reject absolute paths.
    - Reject '..' traversal.
    - Enforce file is under <root>/<expected_dir>/...
    """
    p = Path(str(rel))

    if p.is_absolute():
        raise ValueError(f"Invalid artifact manifest: absolute path not allowed: {rel!r}")

    # Normalize and ensure it's inside expected_dir
    full = (root / p).resolve()
    expected_root = (root / expected_dir).resolve()

    # Must be under expected directory
    if expected_root not in full.parents:
        raise ValueError(
            "Invalid artifact manifest: file path escapes expected directory.\n"
            f"  rel={rel!r}\n"
            f"  expected_dir={expected_dir!r}"
        )

    if not full.exists() or not full.is_file():
        raise FileNotFoundError(f"Artifact file not found: {full}")

    return full

def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    fmt = str(manifest.get("format", "")).strip()
    ver = str(manifest.get("version", "")).strip()

    if fmt != "QuantStrata.MarketDatasetArtifact":
        raise ValueError(f"Unsupported artifact format: {fmt!r}")
    if ver != "v1":
        raise ValueError(f"Unsupported artifact version: {ver!r}")

    if "dates" not in manifest or "n_scenarios" not in manifest:
        raise ValueError("Invalid manifest: missing required fields 'dates'/'n_scenarios'.")


def _safe_filename_from_key(key: str) -> str:
    """
    Convert a MarketId.key() into a filesystem-safe name.

    Notes
    -----
    - This is not a reversible encoding (manifest stores the true MarketId key anyway).
    - Keeps artifacts portable across OS/filesystems.
    """
    s = str(key)
    # Replace separators commonly problematic on filesystems
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
        s = s.replace(ch, "_")
    return s


def _remove_tree(path: Path) -> None:
    """
    Remove all files/directories under path (but not path itself).
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            (Path(root) / f).unlink(missing_ok=True)
        for d in dirs:
            (Path(root) / d).rmdir()