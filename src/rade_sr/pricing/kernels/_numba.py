"""
Optional-Numba shim.

The pricing kernels are written to be ``@njit`` compiled for throughput. In
environments where Numba is installed this re-exports the real symbols. Where it
is not (CI, lightweight images, quick local runs), it provides no-op fallbacks so
the kernels still run as plain Python/NumPy — correct, just slower.

Import ``njit``/``prange``/type symbols from here, never from ``numba`` directly,
so the whole package degrades gracefully.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on environment
    from numba import boolean, float64, int64, njit, prange, vectorize  # type: ignore

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        """No-op stand-in for ``numba.njit`` (bare or parametrised use)."""
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
        """Callable/subscriptable placeholder so type signatures are inert.

        Supports ``float64``, ``float64(float64)``, ``float64[:]`` forms used in
        kernel signatures, all collapsing to a harmless sentinel.
        """

        def __call__(self, *args, **kwargs):
            return self

        def __getitem__(self, item):
            return self

    float64 = _NumbaType()  # type: ignore
    int64 = _NumbaType()  # type: ignore
    boolean = _NumbaType()  # type: ignore
