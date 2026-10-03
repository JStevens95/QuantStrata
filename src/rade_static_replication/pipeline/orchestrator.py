"""
Orchestrator — sequences the pipeline and owns the audit trail.

Each stage method reads contracts off the :class:`RunContext`, runs one stage, records
its output + timing, and returns it. :meth:`run` executes the full sequence and writes
the auditable artifact store; individual stages can also be driven by hand for
debugging or partial runs.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

from src.rade_static_replication import __version__
from src.rade_static_replication.artifacts.store import RunArtifactStore
from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.assets.registry import default_registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import ArtifactManifest
from src.rade_static_replication.pipeline import stages
from src.rade_static_replication.pipeline.context import RunContext

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the static-replication preprocessing pipeline end to end."""

    def __init__(
        self,
        config: OrchestratorConfig,
        portfolio_client: PortfolioClient,
        market_data_client: MarketDataClient,
        registry: Optional[Registry] = None,
    ) -> None:
        self.ctx = RunContext(
            config=config,
            portfolio_client=portfolio_client,
            market_data_client=market_data_client,
            registry=registry or default_registry(),
        )
        self.store = RunArtifactStore(config, library_version=__version__)

    @contextmanager
    def _timed(self, name: str):
        start = time.perf_counter()
        logger.info("stage start: %s", name)
        yield
        self.ctx.timings_ms[name] = (time.perf_counter() - start) * 1000.0

    # ---- individual stages ----

    def load(self) -> None:
        with self._timed("load"):
            self.ctx.raw_portfolio = stages.load_portfolio(self.ctx.portfolio_client, self.ctx.cob_date)

    def normalise(self) -> None:
        with self._timed("normalise"):
            self.ctx.portfolio, self.ctx.warnings = stages.normalise_portfolio(self.ctx.raw_portfolio)

    def resolve(self) -> None:
        with self._timed("resolve"):
            self.ctx.universe = stages.resolve_universe(self.ctx.portfolio, self.ctx.config)

    def build(self) -> None:
        with self._timed("build_market"):
            self.ctx.factor_data = stages.build_factor_data(
                self.ctx.universe, self.ctx.registry, self.ctx.market_data_client, self.ctx.cob_date,
            )

    def generate(self) -> None:
        with self._timed("generate"):
            self.ctx.elementary = stages.generate_elementary(
                self.ctx.universe, self.ctx.factor_data, self.ctx.registry, self.ctx.config,
            )

    def price(self) -> None:
        with self._timed("price"):
            self.ctx.base_prices = stages.price_base(
                self.ctx.elementary, self.ctx.factor_data, self.ctx.registry,
            )

    def pnl(self) -> None:
        with self._timed("pnl"):
            self.ctx.pnl = stages.compute_pnl(
                self.ctx.elementary, self.ctx.factor_data, self.ctx.base_prices,
                self.ctx.registry, self.ctx.config.engine,
            )

    def cluster(self) -> None:
        with self._timed("cluster"):
            self.ctx.clusters = stages.resolve_clusters(
                self.ctx.portfolio, self.ctx.universe, self.ctx.config,
            )

    def write(self) -> ArtifactManifest:
        with self._timed("write"):
            entries = self.store.write_clusters(
                self.ctx.clusters, self.ctx.portfolio, self.ctx.pnl, self.ctx.elementary,
            )
        return ArtifactManifest(entries=entries, root=self.store.layout.run_dir)

    # ---- full run ----

    def run(self) -> RunContext:
        """Execute the full pipeline and write the artifact store."""
        self.store.begin()
        try:
            self.load()
            self.normalise()
            self.resolve()
            self.store.record_portfolio(self.ctx.portfolio, self.ctx.universe)
            self.build()
            self.generate()
            self.price()
            self.pnl()
            self.cluster()
            self.write()
            self.store.finalise(counts=self._counts(), timings_ms=self.ctx.timings_ms)
        except Exception as exc:
            self.store.fail(str(exc))
            raise
        return self.ctx

    def _counts(self) -> dict:
        return {
            "trades": len(self.ctx.portfolio.trade_ids) if self.ctx.portfolio else 0,
            "risk_factors": len(self.ctx.universe.specs) if self.ctx.universe else 0,
            "elementary_trades": self.ctx.elementary.n_trades if self.ctx.elementary else 0,
            "clusters": len(self.ctx.clusters.clusters) if self.ctx.clusters else 0,
        }
