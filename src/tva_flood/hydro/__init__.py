"""Distributed forecast model: neural runoff generation, mass-conserving routing, learned assimilation.

One model serves every evaluation: gauged forecasts (flow history at gauges), ungauged
forecasts (no history at the scored gauge) and forecasts at every one of the 58,834 reaches.

- ``paths``: data locations and the fixed time windows.
- ``prepare``: antecedent-storage features, per-reach NWM statistics, routing groups.
- ``data``: array access on one six-hour timeline (training and test caches), samples.
- ``model``: runoff network, routing, bias and assimilation heads.
- ``train``: stage A (simulation on NWM at every reach) and stage B (observed flow + NWM prior).
- ``evaluate``: the three protocols (gauged, ungauged, every reach) in one pass.
"""
