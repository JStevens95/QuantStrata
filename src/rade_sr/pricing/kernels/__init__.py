"""
JIT-compiled pricing kernels — pure numeric functions via Numba @njit.

These are the computational hot path. Each kernel takes only primitive
numeric inputs (float64, float64[:], float64[:,:]) and returns numeric
outputs. No Python objects, no dicts, no classes.

Architecture:
  - _math.py     : shared numeric utilities (normal CDF/PDF, discount factors)
  - fx.py        : FX forward, vanilla, digital kernels (GK convention)
  - rates.py     : IR swap, swaption, cap/floor kernels

The instrument classes in instruments/ delegate to these kernels:
  instrument.price(asset) → extracts floats → calls kernel → returns result

Scenario PnL is computed by the OptionPricer (pricing/option_pricer.py)
which loops over scenarios, calling scalar kernels with fully-shocked
market data for each scenario.
"""
