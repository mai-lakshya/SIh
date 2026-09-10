# Technical Specification (TechSpec)
## Project: NHAI Infrastructure Risk Engine

### 1. Runtime & Stack
- **Language:** Python 3.11+ / Python 3.14 compatible
- **API Framework:** FastAPI with CORS middleware and rate-limiting
- **Machine Learning Core:** Scikit-Learn, CatBoost, XGBoost, LightGBM, Scikit-Survival (`sksurv`)
- **Explainability:** SHAP (TreeExplainer with Meta-Learner logit weighting)
- **Frontend:** Vanilla HTML5 / Modern CSS (OpenDesign tokens) / ES6 JavaScript / Chart.js 4.4.1

### 2. API Endpoints
- `POST /predict`: Comprehensive inference returning calibrated probability, risk tier, predicted delay days, median survival days, TreeSHAP waterfall, and prescriptive actions.
- `POST /simulate`: Real-time scenario sandbox comparing baseline vs. intervention parameters.
- `GET /health`: Healthcheck endpoint with model artifact and monitor status.
- `GET /metrics`: Drift detection and model observability metrics.
- `GET /`: Serves the built-in SYZYGY OpenDesign frontend dashboard.

### 3. Model Artifacts
- `pipeline.joblib`: Preprocessing, imputation, and feature engineering transformations.
- `ensemble.joblib`: Stacking ensemble of CatBoost, XGBoost, LightGBM, and ExtraTrees.
- `timeline.joblib`: Random Survival Forest estimator for time-to-delay modeling.
