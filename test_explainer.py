import numpy as np
import pandas as pd
import pytest
import json
from hybrid_model import HybridRiskPredictor
from explainer import DualParadigmExplainer

@pytest.fixture
def sample_explainer_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(25, 5), columns=[f'f{i}' for i in range(5)])
    y_cls = np.random.randint(0, 2, 25)
    y_crs = np.random.rand(25) * 100
    y_days = np.random.rand(25) * 1000
    return X, y_cls, y_crs, y_days

def test_explainer_schema_and_finiteness(sample_explainer_data):
    X, y_cls, y_crs, y_days = sample_explainer_data
    
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    
    feature_names = list(X.columns)
    explainer = DualParadigmExplainer(predictor, feature_names)
    
    # Test on a single instance
    X_local = X.iloc[0:1]
    
    payload_str = explainer.generate_json_payload(X_local)
    payload = json.loads(payload_str)
    
    assert "global_importance" in payload
    assert "local_explanation" in payload
    assert "risk_drivers" in payload
    
    # Check risk drivers length (up to 5)
    assert len(payload["risk_drivers"]) <= 5
    
    # Check finite values in local explanation
    for item in payload["local_explanation"]:
        assert "feature" in item
        assert np.isfinite(item["shap_impact"])
        assert np.isfinite(item["attention"])
        assert np.isfinite(item["unified_score"])
        
    # Check bounds of unified score
    for item in payload["local_explanation"]:
        assert 0.0 <= item["unified_score"] <= 1.0
