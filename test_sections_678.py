import os
import time
import json
import joblib
import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from hybrid_model import HybridRiskPredictor
from explainer import DualParadigmExplainer
from extract_shap import extract_xgb_model

def resolve_model_path(filename):
    """Checks for model weights in models/ and root directory."""
    for candidate in [os.path.join("models", filename), filename]:
        if os.path.exists(candidate):
            return candidate
    return None

# Check artifacts existence for skip marks
PIPELINE_EXISTS = resolve_model_path("final_pipeline_cpu.joblib") is not None
PREDICTOR_EXISTS = resolve_model_path("sih_risk_engine_final.joblib") is not None
DATASET_EXISTS = os.path.exists("indian_infrastructure_projects_dataset.csv")

pytestmark = pytest.mark.skipif(
    not (PIPELINE_EXISTS and PREDICTOR_EXISTS and DATASET_EXISTS),
    reason="Model artifacts (final_pipeline_cpu.joblib, sih_risk_engine_final.joblib) or dataset not found on disk"
)

@pytest.fixture(scope="module")
def models_and_explainer():
    """Initializes pipeline, predictor, background data, and DualParadigmExplainer."""
    pipe_path = resolve_model_path("final_pipeline_cpu.joblib")
    pred_path = resolve_model_path("sih_risk_engine_final.joblib")
    
    pipeline = joblib.load(pipe_path)
    predictor = HybridRiskPredictor.load(pred_path)

    df = pd.read_csv("indian_infrastructure_projects_dataset.csv")
    X_raw = df.drop(columns=[
        'delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index',
        'delay_risk_tier', 'CRS_tier', 'section_11_notification_days'
    ], errors='ignore').head(100)
    
    X_bg = pipeline.transform(X_raw)
    feature_names = list(X_bg.columns)
    
    explainer = DualParadigmExplainer(predictor, feature_names, background_data=X_bg)
    return pipeline, predictor, explainer, feature_names, X_bg

@pytest.fixture
def sample_payloads():
    """Provides validated test payloads for low-risk and high-risk scenarios."""
    payload_low = {
        'project_id': 'TEST-LOW',
        'state': 'Gujarat',
        'district': 'Unknown',
        'land_area_hectares': 10.0,
        'land_area_log': 1.0,
        'project_type': 'Highway',
        'terrain_type': 'Plain',
        'estimated_cost_inr_crore': 50.0,
        'affected_families_count': 10,
        'title_dispute_rate_percent': 1.0,
        'local_protest_flag': False,
        'compensation_multiplier_demand': 1.0,
        'sia_approval_status': 'Approved',
        'forest_clearance_status': 'Approved',
        'fund_disbursement_percent': 90.0,
        'project_start_year': 2022,
        'project_age_years': 1,
        'sia_approval_status_risk_score': 0.1,
        'forest_clearance_status_risk_score': 0.1,
        'C_r': 0.1, 'F_r': 0.1, 'H_r': 0.1, 'W_r': 0.1, 'P_r': 0.1
    }

    payload_high = {
        'project_id': 'TEST-HIGH',
        'state': 'Gujarat',
        'district': 'Unknown',
        'land_area_hectares': 500.0,
        'land_area_log': 6.0,
        'project_type': 'Highway',
        'terrain_type': 'Hilly',
        'estimated_cost_inr_crore': 5000.0,
        'affected_families_count': 2000,
        'title_dispute_rate_percent': 25.0,
        'local_protest_flag': True,
        'compensation_multiplier_demand': 3.5,
        'sia_approval_status': 'Pending',
        'forest_clearance_status': 'Pending',
        'fund_disbursement_percent': 10.0,
        'project_start_year': 2018,
        'project_age_years': 5,
        'sia_approval_status_risk_score': 0.9,
        'forest_clearance_status_risk_score': 0.9,
        'C_r': 0.9, 'F_r': 0.9, 'H_r': 0.9, 'W_r': 0.9, 'P_r': 0.9
    }
    return payload_low, payload_high


# ==============================================================================
# SECTION 6: Validation of Top Drivers
# ==============================================================================
def test_section_6_high_risk_top_drivers(models_and_explainer, sample_payloads):
    """
    Validates that high-risk inputs attribute risk positively ('increases_delay')
    to critical risk factors such as disputes, pending clearances, or protests.
    """
    pipeline, _, explainer, _, _ = models_and_explainer
    _, payload_high = sample_payloads

    df_high = pd.DataFrame([payload_high])
    X_high = pipeline.transform(df_high)

    out_high = explainer.explain(X_high)
    assert "risk_drivers" in out_high
    assert len(out_high["risk_drivers"]) > 0

    # Verify at least one primary driver has positive risk attribution
    directions = [rd["direction"] for rd in out_high["risk_drivers"]]
    assert "increases_delay" in directions, "High-risk payload must contain drivers that increase delay"

    # Verify impact scores are finite and bounded in [0, 1]
    for rd in out_high["risk_drivers"]:
        assert 0.0 <= rd["impact_score"] <= 1.0
        assert rd["source"] in ["TreeSHAP", "TabNet_Attention", "Fallback_Heuristic"]


def test_section_6_low_risk_top_drivers(models_and_explainer, sample_payloads):
    """
    Validates that low-risk inputs correctly show mitigating attributions
    ('decreases_delay') for primary factors.
    """
    pipeline, _, explainer, _, _ = models_and_explainer
    payload_low, _ = sample_payloads

    df_low = pd.DataFrame([payload_low])
    X_low = pipeline.transform(df_low)

    out_low = explainer.explain(X_low)
    assert "risk_drivers" in out_low

    # Verify low-risk factors indicate delay mitigation
    directions = [rd["direction"] for rd in out_low["risk_drivers"]]
    assert "decreases_delay" in directions, "Low-risk payload should contain drivers that decrease delay"


# ==============================================================================
# SECTION 7: Dual-Path SHAP Alignment
# ==============================================================================
def test_section_7_dual_path_shap_alignment(models_and_explainer):
    """
    Compares DualParadigmExplainer ensembled attribution against standalone XGBoost
    SHAP on baseline samples. Verifies Spearman rank correlation and top-5 Jaccard overlap.
    """
    _, predictor, explainer, feature_names, X_bg = models_and_explainer
    baseline_subset = X_bg.head(50)

    # 1. DualParadigmExplainer global importance
    dual_global, _, _ = explainer.get_global_importance(baseline_subset)

    # 2. Standalone XGBoost TreeSHAP
    xgb_model = extract_xgb_model(predictor)
    X_clean = baseline_subset.map(lambda x: float(x) if not pd.isna(x) else 0.0).fillna(0.0)
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_vals = xgb_explainer.shap_values(X_clean)
    if isinstance(xgb_shap_vals, list):
        xgb_shap_vals = xgb_shap_vals[1]
    xgb_global = np.mean(np.abs(xgb_shap_vals), axis=0)

    # Spearman rank correlation
    corr, pval = spearmanr(dual_global, xgb_global)
    assert not np.isnan(corr), "Spearman correlation between dual and XGBoost SHAP must be non-NaN"
    assert corr >= 0.40, f"Dual-path SHAP correlation ({corr:.3f}) below alignment threshold 0.40"

    # Top-5 feature overlap (Jaccard Index)
    top5_dual = set(np.argsort(dual_global)[-5:])
    top5_xgb = set(np.argsort(xgb_global)[-5:])
    jaccard = len(top5_dual.intersection(top5_xgb)) / len(top5_dual.union(top5_xgb))
    assert jaccard >= 0.40, f"Top-5 Jaccard overlap ({jaccard:.3f}) below threshold 0.40"


# ==============================================================================
# SECTION 8: Latency Benchmarks & Regressions
# ==============================================================================
def test_section_8_latency_benchmark_and_sla(models_and_explainer, sample_payloads):
    """
    Benchmarks single-row inference and explanation latency over multiple iterations.
    Asserts compliance with the < 500ms SLA ceiling.
    """
    pipeline, predictor, explainer, _, _ = models_and_explainer
    _, payload_high = sample_payloads
    df_high = pd.DataFrame([payload_high])
    X_high = pipeline.transform(df_high)

    # Warmup
    _ = predictor.predict(X_high)
    _ = explainer.explain(X_high)

    # Benchmark bare prediction latency
    pred_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = predictor.predict(X_high)
        pred_times.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark explanation latency
    expl_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = explainer.explain(X_high)
        expl_times.append((time.perf_counter() - t0) * 1000.0)

    p50_expl = float(np.percentile(expl_times, 50))
    p95_expl = float(np.percentile(expl_times, 95))
    p50_pred = float(np.percentile(pred_times, 50))

    # Assert SLA thresholds (< 500ms for explanation, < 200ms for prediction)
    assert p50_pred < 200.0, f"Prediction P50 latency ({p50_pred:.1f}ms) exceeded 200ms SLA"
    assert p50_expl < 500.0, f"Explanation P50 latency ({p50_expl:.1f}ms) exceeded 500ms SLA"
    assert p95_expl < 500.0, f"Explanation P95 latency ({p95_expl:.1f}ms) exceeded 500ms SLA"
