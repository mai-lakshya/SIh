import numpy as np
import pandas as pd
import pytest
from hybrid_model import HybridRiskPredictor, safe_logit

def test_safe_logit():
    # Should not be inf or -inf
    assert np.isfinite(safe_logit(0.0))
    assert np.isfinite(safe_logit(1.0))
    assert np.isfinite(safe_logit(0.5))

@pytest.fixture
def sample_data():
    np.random.seed(42)
    # 50 samples, 5 features
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f'f{i}' for i in range(5)])
    y_cls = np.random.randint(0, 2, 50)
    y_crs = np.random.rand(50) * 100
    y_days = np.random.rand(50) * 1000
    return X, y_cls, y_crs, y_days

def test_hybrid_risk_predictor_fit_predict(sample_data):
    X, y_cls, y_crs, y_days = sample_data
    
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    
    preds = predictor.predict(X, blend_monotonicity=False)
    
    assert 'delay_probability' in preds
    assert 'crs' in preds
    assert 'delay_days' in preds
    
    # Check bounds
    assert np.all((preds['delay_probability'] >= 0.0) & (preds['delay_probability'] <= 1.0))
    
def test_hybrid_risk_predictor_monotonicity_blend(sample_data):
    X, y_cls, y_crs, y_days = sample_data
    
    predictor = HybridRiskPredictor(random_state=42)
    predictor.fit(X, y_cls, y_crs, y_days)
    
    preds_unblended = predictor.predict(X, blend_monotonicity=False)
    preds_blended = predictor.predict(X, blend_monotonicity=True)
    
    # Simple check to ensure blending mathematically applies
    # pred_days = pred_days * (0.5 + delay_prob)
    expected_blended_days = preds_unblended['delay_days'] * (0.5 + preds_unblended['delay_probability'])
    
    # There's a maximum(0, ...) clip in the blending
    expected_blended_days = np.maximum(0, expected_blended_days)
    
    np.testing.assert_allclose(preds_blended['delay_days'], expected_blended_days, rtol=1e-5)
