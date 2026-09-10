import time
import os
import sys
import platform
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib
from hybrid_model import HybridRiskPredictor
from explainer import DualParadigmExplainer

def run_benchmark():
    print("=" * 70)
    print("XAI ENGINE PROGRAMMATIC BENCHMARK & SYSTEM TELEMETRY")
    print("=" * 70)
    
    # 1. Capture exact hardware & environment programmatically
    env_info = {
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    
    print("\n--- ENVIRONMENT & PLATFORM TELEMETRY ---")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
        
    # 2. Load artifacts
    print("\n--- LOADING PRODUCTION ARTIFACTS ---")
    pipeline = joblib.load('pipeline.joblib')
    predictor = HybridRiskPredictor.load('ensemble.joblib')
    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    X = df.drop(columns=['delay_binary_label', 'Actual_Delay_Days', 'CRS', 'project_index'], errors='ignore')
    X_tf = pipeline.transform(X.head(200))
    feature_names = X_tf.columns.tolist()
    
    print(f"Loaded pipeline and ensemble. Baseline dataset rows: {len(X_tf)}, features: {len(feature_names)}")
    
    explainer = DualParadigmExplainer(predictor, feature_names, X_tf)
    test_instance = X_tf.iloc[[0]]
    
    # 3. Warmup
    print("\nWarming up predictor and explainer...")
    for _ in range(3):
        _ = predictor.predict(test_instance)
        _ = explainer.explain(test_instance)
    print("Warmup complete.")
    
    # 4. Bare Prediction Latency Benchmark (25 samples)
    n_iterations = 25
    pred_samples_ms = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        _ = predictor.predict(test_instance)
        t1 = time.perf_counter()
        pred_samples_ms.append(round((t1 - t0) * 1000, 3))
        
    # 5. Explanation Latency Benchmark (25 samples)
    explain_samples_ms = []
    for i in range(n_iterations):
        t0 = time.perf_counter()
        _ = explainer.explain(test_instance)
        t1 = time.perf_counter()
        explain_samples_ms.append(round((t1 - t0) * 1000, 3))
        
    def get_stats(samples):
        arr = np.array(samples)
        return {
            "min_ms": round(float(np.min(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2)
        }
        
    pred_stats = get_stats(pred_samples_ms)
    explain_stats = get_stats(explain_samples_ms)
    
    # 6. Alignment with Single-Path XGBoost (Section 7 check)
    print("\n--- ATTRIBUTION ALIGNMENT BENCHMARK (DUAL-PATH vs XGBOOST SHAP) ---")
    import shap
    from extract_shap import extract_xgb_model
    xgb_model = extract_xgb_model(predictor)
    clean_row = test_instance.map(lambda x: float(x) if not pd.isna(x) else 0.0).fillna(0.0)
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(clean_row)
    if isinstance(xgb_shap_values, list):
        xgb_shap = np.array(xgb_shap_values[1] if len(xgb_shap_values) > 1 else xgb_shap_values[0])[0]
    elif len(xgb_shap_values.shape) == 3:
        xgb_shap = xgb_shap_values[0, :, 1]
    else:
        xgb_shap = xgb_shap_values[0]
        
    exp_payload = explainer.explain(test_instance)
    full_local = exp_payload["local_explanation_full"]
    ensemble_shap = np.array([item["shap_impact"] for item in full_local])
    
    # Rank correlation
    rho, pval = spearmanr(np.abs(ensemble_shap), np.abs(xgb_shap))
    
    # Top-5 Jaccard
    k = 5
    top_ensemble_idx = set(np.argsort(np.abs(ensemble_shap))[-k:])
    top_xgb_idx = set(np.argsort(np.abs(xgb_shap))[-k:])
    top_ensemble_feats = [feature_names[i] for i in np.argsort(np.abs(ensemble_shap))[-k:][::-1]]
    top_xgb_feats = [feature_names[i] for i in np.argsort(np.abs(xgb_shap))[-k:][::-1]]
    jaccard = len(top_ensemble_idx.intersection(top_xgb_idx)) / len(top_ensemble_idx.union(top_xgb_idx))
    
    alignment_results = {
        "spearman_rho": round(float(rho), 4),
        "spearman_pvalue": float(pval),
        "top5_jaccard": round(float(jaccard), 4),
        "shared_features_count": len(top_ensemble_idx.intersection(top_xgb_idx)),
        "top5_dual_paradigm": top_ensemble_feats,
        "top5_standalone_xgb": top_xgb_feats
    }
    
    results = {
        "telemetry": env_info,
        "raw_prediction_samples_ms": pred_samples_ms,
        "prediction_statistics": pred_stats,
        "raw_explanation_samples_ms": explain_samples_ms,
        "explanation_statistics": explain_stats,
        "sla_target_ms": 500.0,
        "sla_met": explain_stats["p95_ms"] < 500.0,
        "alignment": alignment_results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- BARE PREDICTION BENCHMARK ---")
    print(f"Samples ({len(pred_samples_ms)}): {pred_samples_ms}")
    print(f"Stats: {pred_stats}")
    
    print("\n--- EXPLANATION LATENCY BENCHMARK ---")
    print(f"Samples ({len(explain_samples_ms)}): {explain_samples_ms}")
    print(f"Stats: {explain_stats}")
    
    print("\n--- ATTRIBUTION ALIGNMENT ---")
    print(f"Spearman rho: {rho:.4f} (p = {pval:.4e})")
    print(f"Top-5 Jaccard: {jaccard:.4f} ({alignment_results['shared_features_count']} of 5 shared features)")
    print(f"Top 5 Ensemble: {top_ensemble_feats}")
    print(f"Top 5 XGBoost:  {top_xgb_feats}")
    print("\nBenchmark saved to benchmark_results.json.")
    print("=" * 70)
    
if __name__ == "__main__":
    run_benchmark()
