import os
import datetime
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_model_card(version_tag, output_dir="."):
    """
    Phase 10: Governance & Audit
    Generates a Markdown-based Model Card summarizing the model's capabilities, 
    training logic, and compliance details.
    """
    try:
        from monitor import ModelMonitor
        monitor = ModelMonitor()
        perf = monitor.get_latest_performance()
    except Exception:
        perf = {"roc_auc": "N/A", "ece": "N/A", "rmse": "N/A", "timestamp": "N/A"}

    content = f"""# Model Card: Land Acquisition Risk Engine
**Version:** {version_tag}
**Generated On:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Model Details
- **Architecture:** Hybrid Ensemble (XGBoost, LightGBM, CatBoost, PyTorch TabNet, FT-Transformer) stacked via Meta-Learner.
- **Survival Analysis:** Random Survival Forest & DeepSurv neural network.
- **Task:** Predict land acquisition delays (binary classification) and estimate delay duration/cost impacts.

## 2. Intended Use
- **Primary Use Case:** Proactive risk identification for Land Acquisition for Indian projects (Highways, Railways, Airports, Renewable Energy).
- **Out of Scope:** Small-scale municipal projects or projects outside of the defined taxonomy.

## 3. Performance Metrics (Latest Hold-out)
- **ROC-AUC (Classification):** {perf.get('roc_auc', 'N/A')}
- **Expected Calibration Error (ECE):** {perf.get('ece', 'N/A')}
- **Composite Risk Score RMSE:** {perf.get('rmse', 'N/A')}
- **Evaluated At:** {perf.get('timestamp', 'N/A')}

## 4. Explainability & Fairness
- **Methodology:** Dual-Paradigm explanation utilizing TreeSHAP (for gradient boosting models) and internal sequential attention masks (for PyTorch TabNet).
- **Compliance:** Oversampling was exclusively handled via `SMOTENC` during the `.fit()` phase strictly to avoid target leakage and categorical corruption.

## 5. Limitations & Risk Factors
- **Data Drift:** The model's validity degrades if macroeconomic factors shift. Continuous monitoring via Population Stability Index (PSI) is active.
- **Recommendations:** Auto-generated recommendations should act as decision-support tools and not replace human legal/engineering judgment.

---
*Generated automatically by Phase 10 Continuous Learning Pipeline.*
"""
    
    file_path = os.path.join(output_dir, f"model_card_{version_tag}.md")
    with open(file_path, "w") as f:
        f.write(content)
        
    logging.info(f"Model Card generated successfully at: {file_path}")

if __name__ == "__main__":
    generate_model_card("v2026.08.29")
