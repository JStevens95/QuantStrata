"""
Execution backends - SerialBackend, ThreadingBackend, MultiprocessingBackend

All backends implement a single .map(fn, items) interface so pipeline stages can switch execution strategy without
changing their logic.

SerialBackend - debug and unit tests. Single thread.
ThreadingBackend - IO-bound stages (opt-in). ThreadPoolExecutor.
MultiprocessingBackend - CPU-bound PnL computation. ProcessPoolExecutor numba kernels are warmed up in each worker..
"""
from __future__ import annotations

import os
import logging

from typing import Callable, Iterable, List, TypeVar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# define module level logging.
logger = logging.getLogger(__name__)

# define type variables.
T = TypeVar("T")
R = TypeVar("R")


class SerialBackend:
    """
    Runs all tasks sequentially in the calling thread.

    Use for: debugging, unit tests, single-item batches.
    """

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Apply fn to each item sequentially, return list of results."""
        return [fn(item) for item in items]

    def __repr__(self) -> str:
        """String representation of backend."""
        return "SerialBackend()"


class ThreadingBackend:
    """
    Runs tasks in a thread pool.

    Use for: IO-bound stages e.g. reading multiple CSV files, API calls where the caller explicitly opts in.

    Note: not used for SAGE/STAR API calls - those are sequential only.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initiate ThreadingBackend instance."""

        # initiate required variables.
        self._max_workers = max_workers

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Apply fn to each item in a thread pool, preserving order."""
        item_list = list(items)
        if not item_list:
            return []
        results: List[R] = [None] * len(item_list)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(fn, item): idx for idx, item in enumerate(item_list)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def __repr__(self) -> str:
        """String representation of backend."""
        return f"ThreadingBackend(max_workers={self._max_workers})"


def _worker_init() -> None:
    """Initiate each process worker in MultiprocessingBackend."""
    try:
        from src.static_replication.execution.warmup import warmup_kernels
        warmup_kernels()
    except Exception as exc:
        logger.warning("Worker warmup failed (non-fatal): %s", exc)


class MultiprocessingBackend:
    """Runs tasks in a multi-processing pool."""

    def __init__(self, max_workers: int = None) -> None:
        """Initiate MultiprocessingBackend instance."""

        # initiate required variables.
        self._max_workers = max_workers or os.cpu_count() or 4

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
        """Apply fn to each item sequentially, return list of results."""
        item_list = list(items)
        if not item_list:
            return []
        results: List[R] = [None] * len(item_list)  # type: ignore[list-item]
        with ProcessPoolExecutor(max_workers=self._max_workers, initializer=_worker_init) as executor:
            futures = {executor.submit(fn, item): idx for idx, item in enumerate(item_list)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def __repr__(self) -> str:
        """String representation of backend."""
        return f"MultiprocessingBackend(max_workers={self._max_workers})"
