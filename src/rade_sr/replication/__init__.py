"""
Replication pipeline — preprocessing engine, stages, and orchestration.

Flat package containing the portfolio-first preprocessing pipeline and all
concrete stage implementations:

    pipeline.py          PipelineConfig, PreprocessingPipeline
    orchestrator.py      build_pipeline(), run_preprocessing()
    registry.py          RiskFactorRegistry (decorator-based dispatch)
    builders.py          FXRiskFactorBuilder, RatesRiskFactorBuilder
    trade_generator.py   RiskFactorAwareTradeGenerator
    pnl_engine.py        VectorisedPnLEngine
    cluster_resolver.py  ClusterResolver wrapper
    path_resolver.py     ConventionPathResolver
    portfolio.py         PortfolioManager (post-pipeline aggregation)
"""
