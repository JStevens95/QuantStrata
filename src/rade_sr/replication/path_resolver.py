"""
Path resolver stage — derives artifact paths per cluster.

Translates a root directory + naming conventions into the four canonical
file paths (target PnL, target attributes, elementary PnL, elementary
attributes) for each cluster.

Subclass or replace to support different path layouts (e.g. database-backed,
cloud storage, date-partitioned directories).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.rade_sr.core.types import ClusterPaths

logger = logging.getLogger(__name__)


class ConventionPathResolver:
    """Derives artifact paths from a root directory + naming convention.

    The convention is: ``{root}/{cluster_id}/{filename}`` for each of the
    four artifact types. Filenames come from ``path_config``.

    This resolver is stateless and can be shared across clusters.
    """

    def resolve(self, cluster_id: str, path_config: Dict[str, Any]) -> ClusterPaths:
        """Resolve the four artifact paths for one cluster.

        Parameters
        ----------
        cluster_id : str
            Cluster identifier (used as subdirectory name).
        path_config : dict
            Path configuration. Required keys:

            - ``root`` (str): Root directory for all cluster data.
            - ``target_pnl_file`` (str): Filename for target PnL
              (e.g. ``"target_pnl.parquet"``).
            - ``target_attr_file`` (str): Filename for target attributes.
            - ``elem_pnl_file`` (str): Filename for elementary PnL.
            - ``elem_attr_file`` (str): Filename for elementary attributes.

        Returns
        -------
        ClusterPaths
            Frozen dataclass with the four resolved paths.

        TODO(wire): If your path convention is different (e.g. paths include
        dates, versions, or are not cluster-subdirectories), override this
        method or write a new PathResolver implementation.
        """
        root = Path(path_config["root"])
        cluster_dir = root / cluster_id

        return ClusterPaths(
            target_pnl=cluster_dir / path_config["target_pnl_file"],
            target_attributes=cluster_dir / path_config["target_attr_file"],
            elem_pnl=cluster_dir / path_config["elem_pnl_file"],
            elem_attributes=cluster_dir / path_config["elem_attr_file"],
        )
