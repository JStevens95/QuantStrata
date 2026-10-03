"""Pydantic schemas for the API's discovery / liveness endpoints.

Not tied to any evaluation artifact — these describe the running server
and the versions it can see.  Lifted into a module so the typed client
can consume them with the same models the server emits.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """``GET /health`` payload — used for readiness + probes."""

    status: str = Field(..., description="Always ``\"ok\"`` while the server is live.")
    version: str = Field(..., description="Resolved concrete ensemble version.")
    artifacts_dir: str = Field(..., description="Root of the artifact store in use.")


class VersionsResponse(BaseModel):
    """``GET /versions`` payload — active + all available versions."""

    active: str = Field(..., description="Version currently served.")
    available: List[str] = Field(
        ..., description="Versions visible in the registry (any lifecycle state)."
    )
