import numpy as np
import pandas as pd
import pytest
from timeline_predictor import create_structured_survival_array, NonLinearTimelinePredictor

@pytest.fixture
def sample_survival_data():
    np.random.seed(42)
    # 100 samples, 3 features
    X = pd.DataFrame(np.random.rand(100, 3), columns=['f1', 'f2', 'f3'])
    # Status: 1 (event occurred), 0 (censored)
    status = np.random.choice([0, 1], size=100)
    # Durations between 10 and 400
    durations = np.random.randint(10, 400, size=100)
    
    # Ensure there's a range of statuses and durations for valid tests
    status[0] = 1 
    durations[0] = 300
    
    return X, status, durations

def test_structured_survival_array():
    status = np.array([1, 0, 1])
    durations = np.array([100, 200, 300])
    y_surv = create_structured_survival_array(status, durations)
    
    assert y_surv.dtype.names == ('event', 'time')
    assert np.all(y_surv['event'] == [True, False, True])
    assert np.all(y_surv['time'] == [100.0, 200.0, 300.0])

def test_timeline_predictor_fit_evaluate(sample_survival_data):
    X, status, durations = sample_survival_data
    
    # Split into train/test
    X_tr, X_te = X.iloc[:80], X.iloc[80:]
    st_tr, st_te = status[:80], status[80:]
    dur_tr, dur_te = durations[:80], durations[80:]
    
    predictor = NonLinearTimelinePredictor(random_state=42)
    predictor.fit(X_tr, st_tr, dur_tr)
    
    # Check thresholding
    median_times = predictor.get_dynamic_risk_threshold(X_te)
    assert len(median_times) == len(X_te)
    assert np.all(median_times > 0)
    
    # Check evaluation
    eval_metrics = predictor.evaluate(X_tr, st_tr, dur_tr, X_te, st_te, dur_te)
    assert 'c_index_uno' in eval_metrics
    assert 'integrated_brier_score' in eval_metrics

def test_timeline_predictor_edge_cases():
    X = pd.DataFrame(np.zeros((50, 3)))
    status = np.ones(50)
    durations = np.linspace(10, 100, 50)
    
    predictor = NonLinearTimelinePredictor(random_state=42)
    predictor.fit(X, status, durations)
    
    median_times = predictor.get_dynamic_risk_threshold(X)
    assert len(median_times) == 50
    # Should fallback to something reasonable, not crash
    assert np.all(np.isfinite(median_times))
