import numpy as np
import pandas as pd
import pytest
from pipeline import get_preprocessing_pipeline, LogTransformer, OOFTargetEncoder, SMOTENCDynamicWrapper

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'project_cost_cr': [100.0, 1000.0, 50.0, 50000.0, 10.0, 200.0],
        'land_area_hectares': [10, 50, 5, 1000, 2, 20],
        'affected_families_count': [0, 100, 0, 5000, 0, 10],
        'state': ['A', 'A', 'B', 'B', 'C', 'C'],
        'district': ['X', 'Y', 'X', 'Y', 'X', 'Z']
    })
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y

def test_log_transformer(sample_data):
    X, y = sample_data
    transformer = LogTransformer(cols=['project_cost_cr', 'land_area_hectares', 'affected_families_count'])
    X_out = transformer.fit_transform(X)
    
    assert X_out.shape == X.shape
    assert np.isclose(X_out['project_cost_cr'].iloc[0], np.log1p(100.0))
    # Check shape remains same
    assert list(X_out.columns) == list(X.columns)

def test_oof_target_encoder_finite(sample_data):
    X, y = sample_data
    te = OOFTargetEncoder(cols=['state', 'district'], cv=2)
    X_encoded = te.fit_transform(X, y)
    
    assert X_encoded.shape == X.shape
    # Ensure no NaN or infinite values
    assert np.all(np.isfinite(X_encoded['state']))
    assert np.all(np.isfinite(X_encoded['district']))

def test_smotenc_wrapper(sample_data):
    X, y = sample_data
    # Imbalance the dataset
    X = pd.concat([X, X.iloc[0:1], X.iloc[0:1]])
    y = np.concatenate([y, [0, 0]])
    
    wrapper = SMOTENCDynamicWrapper(random_state=42)
    X_res, y_res = wrapper.fit_resample(X, y)
    
    # Check that minority class has been oversampled
    assert sum(y_res == 1) == sum(y_res == 0)
    
    # Test that transform does nothing
    X_trans = wrapper.transform(X)
    assert len(X_trans) == len(X)
    
def test_pipeline_integration(sample_data):
    X, y = sample_data
    pipeline = get_preprocessing_pipeline(
        cat_cols=['state', 'district'],
        log_cols=['project_cost_cr', 'land_area_hectares'],
        te_cols=['state', 'district'],
        use_smote=False
    )
    
    X_out = pipeline.fit_transform(X, y)
    assert X_out.shape == X.shape
    assert 'state' in X_out.columns
    assert 'district' in X_out.columns
    # Ensure they are numeric after encoding
    assert pd.api.types.is_numeric_dtype(X_out['state'])
