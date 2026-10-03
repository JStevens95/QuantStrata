"""
Asset market data objects - self-contained packages per risk factor.

Each asset class (FX< IR, EQ< CR) has a concrete subclass that loads holds and explores market data
(spot, curves, surfaces), dependent assets and shock scenarios.

Separation from instruments/
- instruments/ defines what elementary trades are used (product specs, pricing, greeks).
- assets/ defines what market looks like (data, curves, shocks)
"""