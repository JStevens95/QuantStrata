"""
Singleton wrapper around ``EnsembleSession``.

All dashboard modules call ``get_session()`` to access the shared
session.  The session is initialised once at app startup with
``initialise()`` and can be reloaded for a different version with
``reload()``.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.rade_ml_pt.ensemble.session import EnsembleSession

logger = logging.getLogger(__name__)

_session: Optional[EnsembleSession] = None
_registry_dir: Optional[str] = None
_artifacts_dir: Optional[str] = None


def initialise(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
) -> None:
    """
    Create the singleton ``EnsembleSession`` and run Phase 1 + 2.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load.
    """
    global _session, _registry_dir, _artifacts_dir

    _registry_dir = registry_dir
    _artifacts_dir = artifacts_dir

    _session = EnsembleSession(
        registry_dir=registry_dir,
        artifacts_dir=artifacts_dir,
    )
    _session.load_metadata(version)
    _session.load_display_artifacts()

    logger.info(
        "Session initialised: version=%s, clusters=%d",
        _session.ensemble_version,
        _session.config.n_members,
    )


def get_session() -> EnsembleSession:
    """
    Return the singleton ``EnsembleSession``.

    Raises
    ------
    RuntimeError
        If ``initialise()`` has not been called.
    """
    if _session is None:
        raise RuntimeError(
            "Session not initialised. Call initialise() at app startup."
        )
    return _session


def reload(version: str = "latest") -> None:
    """
    Reload the session for a different ensemble version.

    Tears down the current session and re-runs Phase 1 + 2.
    Inference state (Phase 3) is not carried over.
    """
    if _registry_dir is None or _artifacts_dir is None:
        raise RuntimeError("Cannot reload — session was never initialised.")
    initialise(_registry_dir, _artifacts_dir, version)
