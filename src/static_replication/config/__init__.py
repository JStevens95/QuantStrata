"""Config package - YAML loader, query templates, static mappings."""
from src.static_replication.config.loader import load_api_config, load_yaml, save_yaml

__all__ = [
    "load_api_config", "load_yaml", "save_yaml"
]
