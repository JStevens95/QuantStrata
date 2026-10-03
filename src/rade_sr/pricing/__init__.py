"""
Pricing layer — PnL computation engines.

Provides batch pricing for elementary trades across shock scenarios.
The pricer is agnostic to the pipeline; it takes instrument parameters
and market data, returns PnL arrays.
"""
