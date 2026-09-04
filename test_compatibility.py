"""
Test compatibility patches across gradient boosted tree and XAI versions.
"""
import pytest
import numpy as np
import shap
import shap.explainers._tree as tree_mod
from explainer import DualParadigmExplainer


def test_shap_xgboost_bracketed_base_score_compatibility():
    """
    Verify that TreeExplainer handles XGBoost bracketed base_score (e.g. '[4E-1]')
    without raising 'ValueError: could not convert string to float'.
    """
    import xgboost as xgb

    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'logloss'}, dtrain, num_boost_round=3)

    # Simulate older SHAP / newer XGBoost bracketed base_score
    orig_decode = tree_mod.decode_ubjson_buffer

    def bracket_injecting_decode(fd):
        j = orig_decode(fd)
        if isinstance(j, dict) and "learner" in j:
            lmp = j["learner"].get("learner_model_param", {})
            lmp["base_score"] = "[4E-1]"
        return j

    tree_mod.decode_ubjson_buffer = bracket_injecting_decode
    try:
        from explainer import _apply_shap_xgb_compatibility_patch
        _apply_shap_xgb_compatibility_patch()
        explainer = shap.TreeExplainer(bst)
        assert explainer is not None
    finally:
        # Re-apply our standard patch
        _apply_shap_xgb_compatibility_patch()
