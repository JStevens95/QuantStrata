"""PRISM / Rade API — FastAPI service exposing ensemble evaluation artifacts.

Public surface:

* :class:`~src.rade_ml_pt.ensemble.api.client.RadeApiClient` — typed
  synchronous HTTP client for every ``/prism/v1/*`` endpoint.
* :class:`~src.rade_ml_pt.ensemble.api.client.RadeApiError` — single
  exception class carrying status code + detail + url.
"""
from src.rade_ml_pt.ensemble.api.client import RadeApiClient, RadeApiError

__all__ = ["RadeApiClient", "RadeApiError"]
