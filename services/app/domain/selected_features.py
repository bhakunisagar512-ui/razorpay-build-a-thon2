"""Active diagnosis inputs; rejected Phase 2 candidates are opt-in experiments.

Feature selection must establish both predictive benefit and no measured safety
regression before extending this tuple. The serving baseline is not changed here.
All 21 additions failed the per-partition safety constraint in the committed
docs/phase2-feature-selection.json development experiment.
The subsequent corrected-data all-26 validation choice also fails held-out
safety; see docs/phase2-safety.md. Keep these inputs until a candidate is promoted.
The binary-veto follow-up also fails fresh confirmation (docs/binary-safety.md).
"""
MODEL_FEATURES = (
    'gw_error_code', 'gw_error_source', 'gw_error_step', 'method', 'instrument_hint',
)
