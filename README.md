# Advanced Multi-Modal Predictive Risk & Survival Engine (V2)

An enterprise-scale, production-ready machine learning system tailored for predicting delays and estimating risk for Indian infrastructure projects. This V2 refactor leverages **state-of-the-art tabular deep learning**, **non-linear survival analysis**, and **multi-objective hyperparameter optimization**.

## System Architecture

```mermaid
graph TD
    A[Raw Data] --> B[Leak-Free Preprocessing Pipeline]
    B -->|Log Transforms| B1[LogTransformer]
    B1 -->|SMOTE-NC (Train Only)| B2[DynamicSMOTENC]
    B2 -->|OOF Target Encoding| B3[OOFTargetEncoder]
    
    B3 --> C[Tabular Ensemble & Survival Engine]
    
    C --> D[Hybrid Classifier & Regressor]
    D --> |LightGBM, XGBoost, CatBoost, TabNet| D1[Stacking Meta-Learner]
    D1 --> |Hold-out Calibration| D2[Probability Calibration]
    
    C --> E[Non-Linear Timeline Predictor]
    E --> |Random Survival Forest| E1[RSF]
    E --> |DeepSurv (Neural Cox)| E2[DeepSurvMLP]
    E1 --> E3[Dynamic Risk-Phase Thresholding]
    
    D2 --> F[Unified Explainer]
    E3 --> F
    F --> |TreeSHAP + TabNet Attention| G[JSON Payload Output]
```

## Features

1. **Leakage-Free Preprocessing (`pipeline.py`)**: Strict out-of-fold target encoding and `fit()`-only `SMOTE-NC` using `imblearn` pipelines.
2. **Hybrid Tabular Ensemble (`hybrid_model.py`)**: A Stacking Ensemble integrating XGBoost, LightGBM, CatBoost, and PyTorch TabNet. Employs `CalibratedClassifierCV` for well-calibrated risk probabilities.
3. **Timeline Survival Engine (`timeline_predictor.py`)**: Random Survival Forest + DeepSurv for exact duration prediction and dynamic phase risk thresholds.
4. **Pareto-Optimal Hyperopt (`evaluate_model.py`)**: Multi-Objective Optuna (NSGA-II) combined with 5x5 Nested Cross-Validation, tracked completely in MLflow.
5. **Dual-Paradigm Interpretability (`explainer.py`)**: Aggregated TreeSHAP + TabNet sequence attention for localized JSON explanation payloads.

## Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/mai-lakshya/SIh.git
cd SIh

# Step 1: Install CPU-only PyTorch (CPU Wheels)
# Prevents downloading >1.5GB of unnecessary CUDA/cuDNN/Triton binaries on machines without a dedicated GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining pinned dependencies
pip install -r requirements.txt

# Alternatively, install all dependencies in a single step using the CPU profile:
# pip install -r requirements-cpu.txt
```

## Running the Pipeline

### 1. Execute Unit & Invariant Stress Tests
Ensure all system invariants (like monotonicity and zero-variance noise checks) are satisfied before training:

```bash
pytest -v
```

### 2. Multi-Objective Nested Cross Validation
This command will execute the 5x5 CV, use Optuna to find the Pareto front (Recall, ECE, MAE), and serialize the best performing model via MLflow.

```bash
# Run the evaluation and hyperparameter optimization protocol
python evaluate_model.py
```
*(Note: For standalone execution, `evaluate_nested_cv` can be imported from `evaluate_model` and run with your Pandas DataFrames).*

## API Inference Example

Once the models are saved (e.g. `hybrid_model.joblib`), they can be loaded for extremely fast inference (validated to run ≤100ms per batch).

```python
import pandas as pd
from hybrid_model import HybridRiskPredictor
from explainer import DualParadigmExplainer

# Load saved model
model = HybridRiskPredictor.load('models/run_XXXX/hybrid_model.joblib')

# Dummy data payload
X_new = pd.DataFrame([{
    'forest_clearance_status': 'Pending',
    'project_cost_cr': 5000.0,
    'land_area_hectares': 200.0,
    'affected_families_count': 1500,
    # ... include other features
}])

# Get Predictions
preds = model.predict(X_new, blend_monotonicity=True)
print(f"Delay Probability: {preds['delay_probability'][0]:.4f}")
print(f"Risk Score (CRS): {preds['crs'][0]:.2f}")
print(f"Delay Days: {preds['delay_days'][0]:.1f}")

# Get Explanations
explainer = DualParadigmExplainer(model, list(X_new.columns))
json_payload = explainer.generate_json_payload(X_new)
print("Explanation Payload:", json_payload)
```
