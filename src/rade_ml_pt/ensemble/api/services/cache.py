"""Mtime-keyed LRU cache for parquet / JSON reads.

The cache key is ``(str(path), mtime_ns, loader)``.  When a file is
rewritten on disk its mtime changes, the cache key changes, and the
loader runs again automatically — no manual invalidation required.

Important
---------
``functools.lru_cache`` keys on **all** positional arguments, including
the ``loader`` callable itself.  Only pass **module-level** functions as
loaders — inline lambdas or closures create fresh function objects each
call and defeat the cache.  See :mod:`rade_ml_pt.ensemble.api.services.reader`
for the canonical ``_load_parquet`` / ``_load_json`` helpers.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def read_with_mtime_cache(
    path: Path,
    loader: Callable[[Path], T],
) -> T:
    """Read ``path`` via ``loader``, caching the result until mtime changes."""
    mtime_ns = path.stat().st_mtime_ns
    return _cached(str(path), mtime_ns, loader)


@lru_cache(maxsize=128)
def _cached(path_str: str, mtime_ns: int, loader: Callable[[Path], T]) -> T:
    path = Path(path_str)
    logger.debug("Reading %s (mtime_ns=%d)", path, mtime_ns)
    return loader(path)


def clear_cache() -> None:
    """Drop all cached entries — useful in tests."""
    _cached.cache_clear()
