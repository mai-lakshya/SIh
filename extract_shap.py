import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_model_path(filename):
    """Resolve file in models/ or workspace root."""
    candidate = os.path.join("models", filename)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(filename):
        return filename
    return candidate

def extract_xgb_model(hybrid_predictor):
    """Robustly extracts the underlying XGBoost model from HybridRiskPredictor."""
    stacker = None
    if hasattr(hybrid_predictor, 'calibrated_classifier') and hasattr(
        hybrid_predictor.calibrated_classifier, 'calibrated_classifiers_'
    ) and len(hybrid_predictor.calibrated_classifier.calibrated_classifiers_) > 0:
        stacker = hybrid_predictor.calibrated_classifier.calibrated_classifiers_[0].estimator
    elif hasattr(hybrid_predictor, 'classifier'):
        stacker = hybrid_predictor.classifier

    if stacker is None:
        raise ValueError("No valid classifier found in hybrid predictor")

    if hasattr(stacker, 'named_estimators_') and 'xgb' in stacker.named_estimators_:
        estimator = stacker.named_estimators_['xgb']
        return estimator.model if hasattr(estimator, 'model') else estimator

    if hasattr(stacker, 'estimators_') and hasattr(stacker, 'estimators'):
        for (name, _), estimator in zip(stacker.estimators, stacker.estimators_):
            if name == 'xgb':
                return estimator.model if hasattr(estimator, 'model') else estimator

    raise ValueError("XGBoost model ('xgb') not found in ensemble estimators")

def run_shap_extraction(final_hybrid=None, pipeline=None, df=None, sample_size=200):
    """
    Runs standalone XGBoost TreeSHAP extraction on preprocessed dataset features.
    Returns (shap_values, feature_names, mean_abs_shap_dict).
    """
    if final_hybrid is None:
        model_p = get_model_path("sih_risk_engine_final.joblib")
        final_hybrid = joblib.load(model_p)
    if pipeline is None:
        pipe_p = get_model_path("final_pipeline_cpu.joblib")
        pipeline = joblib.load(pipe_p)
    if df is None:
        data_p = "indian_infrastructure_projects_dataset.csv"
        df = pd.read_csv(data_p)

    X = df.drop(columns=[
        'delay_binary_label', 'section_11_notification_days', 'CRS',
        'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier'
    ], errors='ignore')

    if sample_size is not None and len(X) > sample_size:
        X = X.head(sample_size)

    X_proc = pipeline.transform(X)
    feature_names = list(X_proc.columns)
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

    def clean_val(x):
        if isinstance(x, str):
            x = x.replace('[', '').replace(']', '')
            try: return float(x)
            except: return 0.0
        elif isinstance(x, (list, np.ndarray)):
            return float(x[0]) if len(x) > 0 else 0.0
        try: return float(x)
        except: return 0.0

    X_proc_df = X_proc_df.map(clean_val).fillna(0.0)
    xgb_model = extract_xgb_model(final_hybrid)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_proc_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    vals = np.abs(shap_values).mean(0)
    shap_dict = {feat: float(v) for feat, v in zip(feature_names, vals)}
    return shap_values, feature_names, shap_dict

if __name__ == "__main__":
    try:
        print("Loading model and data for SHAP extraction...")
        final_hybrid = joblib.load(get_model_path("sih_risk_engine_final.joblib"))
        pipeline = joblib.load(get_model_path("final_pipeline_cpu.joblib"))
        df = pd.read_csv("indian_infrastructure_projects_dataset.csv")

        shap_values, feature_names, shap_dict = run_shap_extraction(final_hybrid, pipeline, df, sample_size=200)

        os.makedirs("models", exist_ok=True)
        shap_df = pd.DataFrame([
            {"Feature": f, "SHAP_Importance": imp} for f, imp in shap_dict.items()
        ]).sort_values(by="SHAP_Importance", ascending=False)
        shap_df.to_csv("models/shap_feature_importance.csv", index=False)
        print("SHAP feature importance saved to models/shap_feature_importance.csv!")

        # Create plot
        plt.figure(figsize=(10, 8))
        X = df.drop(columns=[
            'delay_binary_label', 'section_11_notification_days', 'CRS',
            'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier'
        ], errors='ignore').head(200)
        X_proc_df = pipeline.transform(X).map(lambda x: float(x) if not pd.isna(x) else 0.0).fillna(0.0)
        shap.summary_plot(shap_values, X_proc_df, show=False)
        plt.savefig("models/shap_summary_plot.png", bbox_inches="tight")
        plt.close()
        print("SHAP summary plot saved to models/shap_summary_plot.png!")
    except Exception as e:
        print(f"SHAP Error: {e}")
