"""data - external data acquisition: SAGe/STAR clients, risk factor builders, cluster resolvers."""
from src.static_replication.data.client import SageClient, StarClient, CsvSageClient, CsvStarClient
from src.static_replication.data.builders import FxRiskFactorBuilder, IrRiskFactorBuilder
from src.static_replication.data.cluster_resolver import AttributeClusterResolver
from src.static_replication.core.trade_schema import LINEAR_DEFAULTS, OPTIONS_REQUIRED_COLUMNS, is_option_product

__all__ = [
    # Clients
    "SageClient", "StarClient", "CsvSageClient", "CsvStarClient"
    # Builders / cluster resolvers.
    "FxRiskFactorBuilder", "IrRiskFactorBuilder", "AttributeClusterResolver"
    # Product-family helpers
    "LINEAR_DEFAULTS", "OPTIONS_REQUIRED_COLUMNS", "is_option_product"
]
