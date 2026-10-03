"""Rade Analytics — the primary Dash UI for the ensemble platform.

Follows ``docs/platform_designs/RADE_UI_DESIGN.md`` as the visual and
functional contract, and consumes :class:`RadeApiClient` for all data.

Do not import from this package at module level in ``app.py`` of other
Dash apps — this package owns its own Dash instance.
"""
