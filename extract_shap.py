import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    print("Loading model and data for SHAP extraction...")
    final_hybrid = joblib.load("models/sih_risk_engine_final.joblib")
    pipeline = joblib.load("models/final_pipeline_cpu.joblib")

    df = pd.read_csv('indian_infrastructure_projects_dataset.csv')
    X = df.drop(columns=['delay_binary_label', 'section_11_notification_days', 'CRS', 'project_index', 'Actual_Delay_Days', 'delay_risk_tier', 'CRS_tier'], errors='ignore')
    
    X_proc = pipeline.transform(X)
    cols = [f"Feature_{i}" for i in range(X_proc.shape[1])]
    X_proc_df = pd.DataFrame(X_proc, columns=cols)

    print("Extracting XGBoost model from stack...")
    # Get the named estimator correctly and extract the underlying model from the wrapper
    xgb_model = final_hybrid.classifier.named_estimators_['xgb'].model

    print("Calculating SHAP values...")
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
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_proc_df)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_proc_df, show=False)
    plt.savefig('models/shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    vals = np.abs(shap_values).mean(0)
    shap_df = pd.DataFrame(list(zip(X_proc_df.columns, vals)), columns=['Feature', 'SHAP_Importance'])
    shap_df.sort_values(by=['SHAP_Importance'], ascending=False, inplace=True)
    shap_df.to_csv('models/shap_feature_importance.csv', index=False)
    print("SHAP feature importance saved to models/shap_feature_importance.csv!")
except Exception as e:
    print(f"SHAP Error: {e}")
