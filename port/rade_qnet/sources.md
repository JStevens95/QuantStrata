# `src/rade_qnet/sources`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 34 | 1247 | `0933cff34ac5afc2` |

---

## 1. `src/rade_qnet/sources/__init__.py`

1247 bytes · SHA-256 `0933cff34ac5afc2`

```python
"""
Where a training signal comes from.

Supervised learning and reinforcement learning differ far less than their
tooling suggests.  Both consume a stream of batches; they disagree only on
where those batches originate.  ``sources`` makes that the *only* difference:
a fixed dataset, a live environment and a differentiable simulator are each
wrapped into the same ``BatchSource`` protocol, after which a single training
loop can drive any of them.

This is the design decision that lets the framework host both a supervised
graph-temporal network and a policy-gradient agent without two parallel
code paths.

Sub-packages
------------
``dataset``
    Fixed, finite data: loading, leakage-aware splitting, fitted transforms and
    cached artifacts.
``environment``
    Interactive data: the environment protocol, spaces, vectorisation,
    wrappers, episode recording, and an adapter for Gymnasium environments.
``batching``
    The adapters that present either of the above as a ``BatchSource``.

Dependency rule
---------------
May import: ``core``.
May not import: ``engines``, ``orchestration``, ``models``.
Engine-specific concerns (device placement, tensor construction) belong in
``engines``, not here.
"""

__all__: tuple[str, ...] = ()
```

