"""
Pre-computed Ensemble Analytics dashboard.

Uses pre-computed artifacts from ``evaluation/`` instead of building
a ``GlobalPredictionStore``, resulting in sub-second load times
even at 75K+ targets.
"""
from src.ui.apps.ensemble_analytics_db.app import create_app

__all__ = ["create_app"]
