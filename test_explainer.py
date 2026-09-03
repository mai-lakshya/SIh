import json
import numpy as np
import pandas as pd
import pytest
from hybrid_model import HybridRiskPredictor
from explainer import DualParadigmExplainer

@pytest.fixture(scope="module")
def sample_explainer_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y_cls = np.random.randint(0, 2, 100)
    y_crs = np.random.rand(100) * 100
    y_days = np.random.rand(100) * 1000
    return X, y_cls, y_crs, y_days

@pytest.fixture(scope="module")
def fitted_explainer(sample_explainer_data):
    X, y_cls, y_crs, y_days = sample_explainer_data
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    feature_names = list(X.columns)
    return DualParadigmExplainer(predictor, feature_names, background_data=X.head(50))

def test_explainer_schema_and_finiteness(fitted_explainer, sample_explainer_data):
    X, _, _, _ = sample_explainer_data
    explainer = fitted_explainer
    
    # Test on a single instance
    X_local = X.iloc[0:1]
    
    payload_str = explainer.generate_json_payload(X_local)
    payload = json.loads(payload_str)
    
    assert "global_importance_approx" in payload
    assert "local_explanation_full" in payload
    assert "risk_drivers" in payload
    assert "category_breakdown" in payload
    
    # Check risk drivers length (up to 5)
    assert len(payload["risk_drivers"]) <= 5
    
    # Check finite values in local explanation
    for item in payload["local_explanation_full"]:
        assert "feature" in item
        assert np.isfinite(item["shap_impact"])
        assert np.isfinite(item["attention"])
        assert np.isfinite(item["unified_score"])
        assert 0.0 <= item["unified_score"] <= 1.0

    # Check category breakdown sums to 1.0
    bd = payload["category_breakdown"]
    assert pytest.approx(sum(bd.values()), abs=1e-3) == 1.0

def test_explainer_batch_processing(fitted_explainer, sample_explainer_data):
    X, _, _, _ = sample_explainer_data
    explainer = fitted_explainer

    # Batch input (5 rows)
    X_batch = X.iloc[0:5]
    payloads = explainer.explain(X_batch)

    assert isinstance(payloads, list)
    assert len(payloads) == 5

    for payload in payloads:
        assert "global_importance_approx" in payload
        assert "local_explanation_full" in payload
        assert "risk_drivers" in payload
        assert "category_breakdown" in payload
        assert len(payload["local_explanation_full"]) == len(explainer.feature_names)

def test_explainer_robust_input_coercion(fitted_explainer):
    explainer = fitted_explainer

    # Dictionary input with stringified numbers and missing features
    raw_dict = {
        "f0": "0.85",
        "f1": 1.2,
        "f2": "true",
        # f3 and f4 are omitted
    }

    payload = explainer.explain(raw_dict)
    assert isinstance(payload, dict)
    assert "risk_drivers" in payload
    assert len(payload["local_explanation_full"]) == len(explainer.feature_names)
    for item in payload["local_explanation_full"]:
        assert np.isfinite(item["value"])
        assert np.isfinite(item["unified_score"])

def test_background_dataset_global_importance(fitted_explainer, sample_explainer_data):
    X, _, _, _ = sample_explainer_data
    explainer = fitted_explainer

    # Global importance should reflect feature variance across background data
    global_imp, norm_tree, _ = explainer.get_global_importance(X.head(40))
    assert len(global_imp) == len(explainer.feature_names)
    assert np.all(global_imp >= 0.0)
    assert np.all(global_imp <= 1.0)
    assert np.any(global_imp > 0.0)
