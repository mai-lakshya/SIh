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

def test_explainer_schema_and_finiteness(fitted_explainer, sample_explainer_data, caplog):
    import logging
    X, _, _, _ = sample_explainer_data
    explainer = fitted_explainer

    # Verify TabNet absence emits an INFO level log rather than a UserWarning
    with caplog.at_level(logging.INFO):
        test_exp = DualParadigmExplainer(explainer.hybrid_predictor, explainer.feature_names)
        assert test_exp.tabnet_model is None
        assert any("No TabNet neural attention estimator detected" in r.message for r in caplog.records)
        assert not any(r.levelno >= logging.WARNING and "TabNet" in r.message for r in caplog.records)
    
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


def test_validate_additivity_with_tabnet():
    """Regression test: fits an ensemble WITH a tabnet-like estimator and asserts is_exact stays True."""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    class MockTabNet(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            N = len(X)
            p = np.full((N, 1), 0.65)
            return np.hstack([1 - p, p])

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def explain(self, X):
            N, D = X.shape
            return np.ones((N, D), dtype=np.float32) / D, None

    class DummyPredictor:
        def __init__(self, stacker):
            self.classifier = stacker

        def predict(self, X):
            return {'delay_probability': self.classifier.predict_proba(X)[:, 1]}

    X = pd.DataFrame(np.random.rand(30, 5), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, 30)

    tab = MockTabNet().fit(X, y)
    et = ExtraTreesClassifier(n_estimators=10, random_state=42).fit(X, y)

    stacker = StackingClassifier(
        estimators=[('et', et), ('tab', tab)],
        final_estimator=LogisticRegression(),
        cv=2
    )
    stacker.fit(X, y)

    pred = DummyPredictor(stacker)
    explainer = DualParadigmExplainer(pred, list(X.columns), background_data=X.head(15))

    assert explainer.tabnet_model is not None, "TabNet model was not detected/extracted"
    assert 'tab' in explainer.meta_coefficients, "TabNet meta coefficient was not extracted"

    res = explainer.validate_additivity(X.head(10))
    assert res['is_exact'] is True, f"Additivity check failed with TabNet: {res}"
    assert res['max_absolute_error'] < 1e-4


def test_model_failure_surfacing_and_fallback():
    """Tests that base model exceptions are surfaced, warned, tracked in models_failed, and raise if all fail."""
    import warnings

    class FaultyModel:
        n_features_in_ = 5

        def predict_proba(self, X):
            raise ValueError("Forced CatBoost tree dump failure for testing")

    class WorkingModel:
        n_features_in_ = 5

        def predict_proba(self, X):
            return np.ones((len(X), 2)) * 0.5

    class DummyPredictor:
        def predict(self, X):
            return {'delay_probability': np.full(len(X), 0.5)}

    X = pd.DataFrame(np.random.rand(5, 5), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, 5)

    # 1. When all models fail and fallback is NOT allowed: must raise RuntimeError
    explainer = DualParadigmExplainer(DummyPredictor(), list(X.columns), allow_fallback=False)
    explainer.tree_models = {'cat': FaultyModel()}
    explainer.meta_coefficients = {'cat': 1.0}

    with pytest.raises(RuntimeError) as excinfo:
        with warnings.catch_warnings(record=True):
            explainer.explain(X.iloc[0:1])
    assert "cat" in str(excinfo.value)

    # 2. When all models fail but fallback is allowed: warns, tracks models_failed, returns fallback payload
    with warnings.catch_warnings(record=True) as recorded_warns:
        warnings.simplefilter("always")
        payload = explainer.explain(X.iloc[0:1], allow_fallback=True)
        assert any("cat" in str(w.message) for w in recorded_warns)
        assert "models_failed" in payload
        assert "cat" in payload["models_failed"]
        assert payload["risk_drivers"][0]["source"] == "Fallback_Heuristic"

    # 3. When one model fails and one succeeds: warns, tracks in models_failed, but succeeds without fallback
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=5, random_state=42).fit(X, y)
    explainer.tree_models = {'cat': FaultyModel(), 'et': et}
    explainer.meta_coefficients = {'cat': 1.0, 'et': 1.0}

    with warnings.catch_warnings(record=True) as recorded_warns:
        warnings.simplefilter("always")
        payload = explainer.explain(X.iloc[0:1])
        assert any("cat" in str(w.message) for w in recorded_warns)
        assert "models_failed" in payload
        assert "cat" in payload["models_failed"]
        assert "et" not in payload["models_failed"]
        assert payload["risk_drivers"][0]["source"] == "TreeSHAP"


def test_ci_faithfulness_gates():
    """
    CI-enforced faithfulness gates:
    - ExtraTrees ('et') present in tree_models
    - validate_additivity is_exact True / max_error < 1e-4
    - Deletion/insertion directional fidelity >= 0.70
    """
    import os
    import joblib

    required_artifacts = ['ensemble.joblib', 'pipeline.joblib', 'indian_infrastructure_projects_dataset.csv']
    missing = [f for f in required_artifacts if not os.path.exists(f)]
    if missing:
        pytest.skip(f"Required artifact(s) missing for CI faithfulness gates: {', '.join(missing)}")

    predictor = joblib.load('ensemble.joblib')
    pipeline = joblib.load('pipeline.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    X_raw = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X_raw.head(30))
    bg = X_tf.iloc[:20].copy()
    neutral_medians = bg.median(numeric_only=True).to_dict()

    explainer = DualParadigmExplainer(predictor, list(X_tf.columns), background_data=bg)

    # Gate 1: 'et' must be present whenever it exists in stacker
    stacker = predictor.calibrated_classifier.calibrated_classifiers_[0].estimator if hasattr(predictor, 'calibrated_classifier') else predictor.classifier
    stacker_estimator_names = [name for name, _ in stacker.estimators] if hasattr(stacker, 'estimators') else []
    if 'et' in stacker_estimator_names:
        assert 'et' in explainer.tree_models, "ExtraTrees ('et') missing from explainer.tree_models"

    # Gate 2: Exact logit additivity
    add_res = explainer.validate_additivity(X_tf.head(10))
    assert add_res['is_exact'] is True, f"Additivity check failed: {add_res}"
    assert add_res['max_absolute_error'] < 1e-4

    # Gate 3: Directional fidelity on non-zero pre-calibration deltas >= 0.70
    matches = []
    for i in range(10):
        row = X_tf.iloc[[i]]
        p_raw = float(stacker.predict_proba(row)[0, 1])
        exp = explainer.explain(row)
        for driver in exp['risk_drivers'][:3]:
            feat = driver['feature']
            dir_claimed = driver['direction']
            ref = neutral_medians.get(feat, 0.0)
            row_del = row.copy()
            row_del[feat] = ref
            p_del = float(stacker.predict_proba(row_del)[0, 1])
            delta = p_raw - p_del
            if abs(delta) > 1e-5:
                m = (delta > 0) if dir_claimed == 'increases_delay' else (delta < 0)
                matches.append(m)

    assert len(matches) > 0, "No non-zero pre-calibration deltas observed"
    directional_fidelity = np.mean(matches)
    assert directional_fidelity >= 0.70, f"Directional fidelity {directional_fidelity:.3f} below 0.70 threshold"


def test_category_breakdown_real_feature_isolation():
    """Tests that COLUMN_CATEGORY_MAPPING keys strictly isolate geography from environmental_clearance."""
    mapping = DualParadigmExplainer.COLUMN_CATEGORY_MAPPING
    feature_names = list(mapping.keys())

    class DummyPredictor:
        pass

    explainer = DualParadigmExplainer(DummyPredictor(), feature_names)

    # 1. Geographic/administrative features must NOT leak into environmental clearance
    geo_scores = np.zeros(len(feature_names))
    for geo_col in ['land_area_hectares', 'land_area_log', 'state', 'district', 'project_start_year']:
        idx = feature_names.index(geo_col)
        geo_scores[idx] = 10.0

    bd_geo = explainer._compute_category_breakdown(geo_scores)
    assert bd_geo['environmental_clearance'] == 0.0, "Geography/administrative features leaked into environmental_clearance!"
    assert bd_geo['administrative_workflow'] == 1.0
    assert pytest.approx(sum(bd_geo.values()), abs=1e-4) == 1.0

    # 2. Environmental features allocate to environmental_clearance
    env_scores = np.zeros(len(feature_names))
    for env_col in ['forest_clearance_status', 'terrain_type', 'eco_sensitive']:
        idx = feature_names.index(env_col)
        env_scores[idx] = 5.0

    bd_env = explainer._compute_category_breakdown(env_scores)
    assert bd_env['environmental_clearance'] == 1.0
    assert bd_env['administrative_workflow'] == 0.0
    assert pytest.approx(sum(bd_env.values()), abs=1e-4) == 1.0


@pytest.fixture(scope="module")
def rsf_survival_model():
    """
    Provides the RandomSurvivalForest component decoupled from PyTorch/DeepSurv.
    Loads rsf_only.joblib if available; otherwise unpickles timeline.joblib and caches rsf_only.joblib.
    """
    import os
    import joblib

    if os.path.exists('rsf_only.joblib'):
        return joblib.load('rsf_only.joblib')

    if os.path.exists('timeline.joblib'):
        tl = joblib.load('timeline.joblib')
        rsf = getattr(tl, 'rsf', None)
        if rsf is not None:
            try:
                joblib.dump(rsf, 'rsf_only.joblib')
            except Exception:
                pass
            return rsf
        raise ValueError("timeline.joblib does not contain an 'rsf' attribute")

    return None


def test_timeline_permutation_explainer(rsf_survival_model):
    """Tests TimelinePermutationExplainer produces finite Uno's C-index permutation importances summing to 1.0."""
    from timeline_explainer import TimelinePermutationExplainer
    import os
    import joblib

    missing = []
    if rsf_survival_model is None and not (os.path.exists('rsf_only.joblib') or os.path.exists('timeline.joblib')):
        missing.append("rsf_only.joblib (or timeline.joblib)")
    if not os.path.exists('pipeline.joblib'):
        missing.append('pipeline.joblib')
    if not os.path.exists('indian_infrastructure_projects_dataset.csv'):
        missing.append('indian_infrastructure_projects_dataset.csv')

    if missing:
        pytest.skip(f"Required artifact(s) missing for timeline permutation explainer: {', '.join(missing)}")

    pipeline = joblib.load('pipeline.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv', nrows=30)
    X_raw = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X_raw)
    events = df['delay_binary_label'].values.astype(bool)
    times = df.get('Actual_Delay_Days', df['delay_binary_label'] * 90).replace(0, 365).values.astype(float)

    # Test initialization via decoupled classmethod from_rsf (avoids PyTorch/DeepSurv dependencies)
    explainer = TimelinePermutationExplainer.from_rsf(rsf_survival_model, list(X_tf.columns), n_repeats=1)
    explainer.fit(X_tf, events, times)

    res = explainer.explain(X_tf.iloc[0:1])
    assert "top_drivers" in res
    assert "feature_importance" in res
    assert "rationale" in res

    importances = [item["importance"] for item in res["feature_importance"]]
    assert len(importances) == len(X_tf.columns)
    assert all(np.isfinite(imp) for imp in importances)
    assert all(imp >= 0.0 for imp in importances)
    assert pytest.approx(sum(importances), abs=1e-4) == 1.0
    assert len(res["rationale"]) > 0
    assert len(res["top_drivers"]) > 0

    # Also verify direct instantiation with bare RSF model works identically
    explainer_direct = TimelinePermutationExplainer(rsf_survival_model, list(X_tf.columns), n_repeats=1)
    assert explainer_direct.rsf_model is not None


def test_timeline_permutation_explainer_torch_free(rsf_survival_model):
    """
    Regression test verifying TimelinePermutationExplainer runs successfully without
    PyTorch imported or available, confirming zero coupling to DeepSurv/torch.
    """
    from timeline_explainer import TimelinePermutationExplainer
    import sys

    class BlockTorch:
        def find_spec(self, fullname, path, target=None):
            if fullname == 'torch' or fullname.startswith('torch.'):
                raise ModuleNotFoundError('No module named ' + fullname)
            return None

    blocker = BlockTorch()
    sys.meta_path.insert(0, blocker)
    try:
        explainer = TimelinePermutationExplainer.from_rsf(rsf_survival_model, ["f0", "f1"], n_repeats=1)
        assert explainer.rsf_model is not None
        assert explainer.timeline_predictor is not None
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)


