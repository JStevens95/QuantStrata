"""
Internal market-data API client — real-environment wiring stub.

This class is the production counterpart to
``rade_sr.sources.market_data.MockMarketDataClient``. It implements the same
:class:`~rade_sr.sources.market_data.MarketDataClient` protocol surface, so once
the methods below are wired to your firm's APIs it can be injected into the
orchestrator in place of the mock with **no other code changes**.

Each method must return the plain dict shape documented in its docstring (the
same shapes ``MockMarketDataClient`` returns and the asset ``_build_*`` methods
consume). Business logic does NOT belong here — this is a transport adapter.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class InternalAPIClient:
    """Market-data + scenario client implementing the MarketDataClient protocol.

    Parameters
    ----------
    base_url : str
        API base URL.
    auth_token : str or None
        Authentication token (or handled by environment/middleware).
    timeout : int
        Request timeout in seconds.

    TODO(wire): Adapt the constructor to your auth mechanism (OAuth, API key,
    Kerberos, etc.) and implement each method below.
    """

    def __init__(
        self,
        base_url: str = "",
        auth_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout

    # ── FX market data ────────────────────────────────────────────────

    def get_fx_spot(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"spot": float, "dates": [str], "values": [float]}``."""
        raise NotImplementedError("Wire get_fx_spot() to your FX spot/history API")

    def get_fx_forward_points(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"tenors": [str], "points": [float]}``."""
        raise NotImplementedError("Wire get_fx_forward_points() to your FX forwards API")

    def get_fx_atm_vol(self, pair: str, cob: Any = None) -> Dict[str, float]:
        """Return ``{tenor_label: atm_vol}`` (e.g. ``{"1M": 0.091, ...}``)."""
        raise NotImplementedError("Wire get_fx_atm_vol() to your FX vol API")

    def get_fx_smile_vol(self, pair: str, cob: Any = None) -> Dict[str, Dict[str, float]]:
        """Return ``{tenor_label: {delta_label: vol}}``."""
        raise NotImplementedError("Wire get_fx_smile_vol() to your FX vol API")

    def get_fx_shocks(self, pair: str, cob: Any = None) -> Dict[str, Any]:
        """Return the FX shock dict (see MockMarketDataClient.get_fx_shocks)."""
        raise NotImplementedError("Wire get_fx_shocks() to your scenario API")

    # ── IR market data ────────────────────────────────────────────────

    def get_ir_curve(self, name: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"tenors": [str], "rates": [float]}``."""
        raise NotImplementedError("Wire get_ir_curve() to your IR curve API")

    def get_ir_history(self, name: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"dates": [str], "tenors": [str], "values": [[float]]}``."""
        raise NotImplementedError("Wire get_ir_history() to your IR history API")

    def get_ir_atm_vol(self, name: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"expiries": [str], "swap_tenors": [str], "values": [[float]]}``."""
        raise NotImplementedError("Wire get_ir_atm_vol() to your swaption vol API")

    def get_ir_smile_vol(self, name: str, cob: Any = None) -> Dict[str, Any]:
        """Return ``{"expiries", "swap_tenors", "strikes", "values"}`` (cube)."""
        raise NotImplementedError("Wire get_ir_smile_vol() to your swaption vol API")

    def get_ir_shocks(self, name: str, cob: Any = None) -> Dict[str, Any]:
        """Return the IR shock dict (see MockMarketDataClient.get_ir_shocks)."""
        raise NotImplementedError("Wire get_ir_shocks() to your scenario API")

    # ── Helpers ───────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        """Build authentication headers. TODO(wire): adapt to your auth."""
        headers: Dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers
