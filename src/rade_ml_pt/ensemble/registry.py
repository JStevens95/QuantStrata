"""
Ensemble-level registry.

Stores ensemble metadata (member versions, cluster mapping, aggregation
config, trade-to-cluster map) as a versioned bundle.  Uses the underlying
``ModelRegistry`` for member model storage and adds ensemble-specific
index / metadata files.

Storage layout::

    root_dir/
      ensemble/
        {ensemble_version}/
          ensemble_config.json       # Full EnsembleConfig
          member_versions.json       # {cluster_id: member_version}
          trade_cluster_map.json     # {trade_id: cluster_id}
          member_summary.json        # {cluster_id: {n_trades, metrics...}}
        index.json                   # tag -> ensemble_version
"""
from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.model import EnsembleModel

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "index.json"


class EnsembleRegistry:
    """
    Local filesystem registry for ensemble model bundles.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for ensemble registrations.  The ``ensemble/``
        subdirectory is created automatically.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir) / "ensemble"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root_dir / _INDEX_FILENAME
        self._index: Dict[str, str] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        config: EnsembleConfig,
        member_versions: Dict[str, str],
        member_summary: Optional[Dict[str, Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register an ensemble version.

        Parameters
        ----------
        config : EnsembleConfig
        member_versions : dict
            ``{cluster_id: registry_version_string}``
        member_summary : dict or None
            ``{cluster_id: {n_trades, mae, mse, ...}}`` for the UI.
        tags : list of str or None
            Labels for retrieval (e.g. ``["production"]``).

        Returns
        -------
        str
            The generated ensemble version identifier.
        """
        tags = tags or []
        version = self._generate_version()
        version_dir = self.root_dir / version
        version_dir.mkdir(parents=True, exist_ok=False)

        config.to_json(version_dir / "ensemble_config.json")

        with open(version_dir / "member_versions.json", "w") as f:
            json.dump(member_versions, f, indent=2)

        trade_cluster_map = {}
        for cid, tids in config.cluster_mapping.items():
            for tid in tids:
                trade_cluster_map[tid] = cid
        with open(version_dir / "trade_cluster_map.json", "w") as f:
            json.dump(trade_cluster_map, f, indent=2)

        if member_summary is not None:
            with open(version_dir / "member_summary.json", "w") as f:
                json.dump(member_summary, f, indent=2)

        for t in tags:
            self._index[t] = version
        self._index["latest"] = version
        self._save_index()

        logger.info("Registered ensemble version '%s' with tags %s", version, tags)
        return version

    def load(
        self,
        version_or_tag: str = "latest",
    ) -> tuple:
        """
        Load ensemble config and member versions.

        Returns
        -------
        tuple of (EnsembleConfig, Dict[str, str], str)
            (config, member_versions, resolved_version)
        """
        version = self._resolve_version(version_or_tag)
        version_dir = self.root_dir / version

        config = EnsembleConfig.from_json(version_dir / "ensemble_config.json")

        with open(version_dir / "member_versions.json", "r") as f:
            member_versions = json.load(f)

        logger.info("Loaded ensemble version '%s' (from '%s')", version, version_or_tag)
        return config, member_versions, version

    def get_metadata(self, version_or_tag: str = "latest") -> Dict[str, Any]:
        """Load the member summary and trade-cluster map for a version."""
        version = self._resolve_version(version_or_tag)
        version_dir = self.root_dir / version

        meta: Dict[str, Any] = {"version": version}

        summary_path = version_dir / "member_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                meta["member_summary"] = json.load(f)

        tcm_path = version_dir / "trade_cluster_map.json"
        if tcm_path.exists():
            with open(tcm_path, "r") as f:
                meta["trade_cluster_map"] = json.load(f)

        return meta

    def tag(self, version: str, tag: str) -> None:
        """Add a tag to an existing ensemble version."""
        version_dir = self.root_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Ensemble version '{version}' not found.")
        self._index[tag] = version
        self._save_index()
        logger.info("Tagged ensemble version '%s' with '%s'", version, tag)

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered ensemble versions with basic metadata."""
        versions = []
        for d in sorted(self.root_dir.iterdir()):
            config_path = d / "ensemble_config.json"
            if not config_path.exists():
                continue
            config = EnsembleConfig.from_json(config_path)
            versions.append({
                "version": d.name,
                "n_members": config.n_members,
                "n_trades": len(config.all_trade_ids),
                "aggregation": config.aggregation,
            })
        return versions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_version(self, version_or_tag: str) -> str:
        if version_or_tag in self._index:
            return self._index[version_or_tag]
        version_dir = self.root_dir / version_or_tag
        if version_dir.exists():
            return version_or_tag
        raise KeyError(
            f"'{version_or_tag}' is not a known tag or ensemble version. "
            f"Available tags: {list(self._index.keys())}"
        )

    def _load_index(self) -> Dict[str, str]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def _generate_version() -> str:
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(now.isoformat().encode()).hexdigest()[:6]
        return f"ens_{ts}_{short_hash}"
