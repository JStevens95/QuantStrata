"""
Optional-Numba shim.

Where Numba is installed this re-exports the real symbols; where it is not, it
provides no-op fallbacks so kernels still run as plain Python/NumPy. Always import
``njit``/type symbols from here, never from ``numba`` directly.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on environment
    from numba import boolean, float64, int64, njit, prange, vectorize  # type: ignore

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def vectorize(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore

    class _NumbaType:
        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, item):
            return self

    float64 = _NumbaType()  # type: ignore
    int64 = _NumbaType()  # type: ignore
    boolean = _NumbaType()  # type: ignore
