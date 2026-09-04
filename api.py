from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import os
import logging
import json
from typing import Optional, List, Dict, Any

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from risk_analysis_system import RiskAnalysisSystem
from monitor import ModelMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Phase 9: Security & Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Land Acquisition Risk API", version="2.0")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = "super-secret-token" # In production, read from env
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(request: Request, api_key_header: Optional[str] = Security(api_key_header)):
    # Accept if matches or if omitted for frontend usage
    if api_key_header and api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header or API_KEY

class ProjectPayload(BaseModel):
    project_id: Optional[str] = 'NHAI-UNKNOWN'
    state: str
    district: Optional[str] = 'Unknown'
    land_area_hectares: float
    land_area_log: Optional[float] = 5.0
    project_type: str
    terrain_type: str
    estimated_cost_inr_crore: float
    affected_families_count: Optional[int] = 500
    title_dispute_rate_percent: Optional[float] = 5.0
    local_protest_flag: Optional[bool] = False
    compensation_multiplier_demand: Optional[float] = 1.5
    sia_approval_status: Optional[str] = 'Pending'
    sia_approval_status_risk_score: Optional[float] = 0.5
    section_11_notification_days: Optional[int] = 30
    forest_clearance_status: Optional[str] = 'Not_Required'
    forest_clearance_status_risk_score: Optional[float] = 0.5
    fund_disbursement_percent: Optional[float] = 10.0
    project_start_year: Optional[int] = 2022
    project_age_years: Optional[int] = 1

class SimulationPayload(BaseModel):
    baseline: ProjectPayload
    interventions: Dict[str, Any]

# Global variables
system: RiskAnalysisSystem = None
monitor: ModelMonitor = None

@app.on_event("startup")
def load_artifacts():
    global system, monitor
    try:
        pipeline_path = 'pipeline.joblib'
        ensemble_path = 'ensemble.joblib'
        timeline_path = 'timeline.joblib'
        
        system = RiskAnalysisSystem(
            pipeline_path=pipeline_path,
            ensemble_path=ensemble_path,
            timeline_path=timeline_path
        )
        monitor = ModelMonitor()
        logging.info("RiskAnalysisSystem and Monitor successfully loaded.")
    except Exception as e:
        logging.error(f"Failed to load artifacts: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "system_ready": system is not None,
        "monitor_ready": monitor is not None,
        "meta_coefficients": getattr(system.explainer, 'meta_coefficients', {}) if (system and system.explainer) else {}
    }

@app.get("/")
def serve_home():
    from fastapi.responses import FileResponse, RedirectResponse
    if os.path.exists("dashboard/index.html"):
        return FileResponse("dashboard/index.html")
    return RedirectResponse(url="/docs")

def _prepare_df(payload_dict: dict) -> pd.DataFrame:
    raw_payload = pd.DataFrame([payload_dict])
    for col in ['C_r', 'F_r', 'H_r', 'W_r', 'P_r']:
        if col not in raw_payload:
            raw_payload[col] = 0.5
    if 'section_11_notification_days' in raw_payload:
        raw_payload = raw_payload.drop(columns=['section_11_notification_days'])
    return raw_payload

def _extract_survival_curve(raw_payload: pd.DataFrame) -> List[Dict[str, Any]]:
    survival_curve = []
    try:
        if system.timeline_predictor and hasattr(system.timeline_predictor, 'rsf') and system.timeline_predictor.rsf is not None:
            X_proc = system.pipeline.transform(raw_payload)
            surv_funcs = system.timeline_predictor.rsf.predict_survival_function(X_proc)
            if len(surv_funcs) > 0:
                fn = surv_funcs[0]
                sample_times = [0, 15, 30, 60, 90, 120, 150, 180, 240, 300, 365, 450, 500, 600, 730]
                max_t = float(fn.x[-1]) if len(fn.x) > 0 else 730.0
                for t in sample_times:
                    if t <= max_t:
                        prob = float(fn(t))
                        survival_curve.append({"day": int(t), "survival_probability": round(prob, 4)})
    except Exception as e:
        logging.warning(f"Could not compute survival curve: {e}")
    return survival_curve

@app.post("/predict")
@limiter.limit("60/minute")
async def predict_risk(request: Request, payload: ProjectPayload, api_key: str = Security(get_api_key)):
    if not system:
        raise HTTPException(status_code=500, detail="Models not loaded")

    raw_payload = _prepare_df(payload.dict(exclude_unset=True))
            
    try:
        result = system.predict(raw_payload)
        survival_curve = _extract_survival_curve(raw_payload)

        # Meta coefficients from StackingClassifier
        meta_coefs = {}
        if system.explainer and hasattr(system.explainer, 'meta_coefficients'):
            meta_coefs = {k: round(float(v), 3) for k, v in system.explainer.meta_coefficients.items()}

        # Top full features with signed TreeSHAP impacts
        full_feats = result['explanation'].get('local_explanation_full', [])
        # Sort full features by absolute impact
        full_feats_sorted = sorted(full_feats, key=lambda x: abs(x.get('shap_impact', 0)), reverse=True)[:10]

        frontend_response = {
            "project_id": payload.project_id,
            "predictions": {
                "delay_probability": round(result['predictions']['delay_probability'] * 100, 1),
                "calibrated_risk_tier": result['predictions']['calibrated_risk_tier'],
                "predicted_delay_days": int(result['predictions']['predicted_delay_days']),
                "median_survival_days": int(result['timeline']['median_survival_days']),
                "crs": round(float(result['predictions'].get('crs', 0.0)), 1),
                "risk_phase": result['timeline'].get('risk_phase', 'Short-term'),
                "predicted_delay_rationale": result['predictions'].get('predicted_delay_rationale', '')
            },
            "explainability": {
                "top_risk_drivers": result['explanation']['risk_drivers'],
                "category_breakdown": result['explanation']['category_breakdown'],
                "local_explanation_full": full_feats_sorted,
                "meta_coefficients": meta_coefs,
                "global_importance": result['explanation'].get('global_importance_approx', [])[:8]
            },
            "survival_curve": survival_curve,
            "recommendations": result.get('recommendations', [])
        }
        return frontend_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/simulate")
@limiter.limit("60/minute")
async def simulate_intervention(request: Request, payload: SimulationPayload, api_key: str = Security(get_api_key)):
    if not system:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # 1. Baseline
        base_dict = payload.baseline.dict(exclude_unset=True)
        base_df = _prepare_df(base_dict)
        base_res = system.predict(base_df)

        # 2. Modified with interventions
        mod_dict = dict(base_dict)
        mod_dict.update(payload.interventions)
        mod_df = _prepare_df(mod_dict)
        mod_res = system.predict(mod_df)

        base_prob = round(base_res['predictions']['delay_probability'] * 100, 1)
        mod_prob = round(mod_res['predictions']['delay_probability'] * 100, 1)
        base_days = int(base_res['predictions']['predicted_delay_days'])
        mod_days = int(mod_res['predictions']['predicted_delay_days'])

        delta_days = base_days - mod_days # positive = saved
        delta_prob = round(base_prob - mod_prob, 1) # positive = reduced risk

        return {
            "baseline": {
                "delay_probability": base_prob,
                "predicted_delay_days": base_days,
                "risk_tier": base_res['predictions']['calibrated_risk_tier']
            },
            "simulated": {
                "delay_probability": mod_prob,
                "predicted_delay_days": mod_days,
                "risk_tier": mod_res['predictions']['calibrated_risk_tier']
            },
            "impact": {
                "days_saved": max(0, delta_days),
                "prob_reduction_percent": max(0.0, delta_prob),
                "status": "Improved" if (delta_days > 0 or delta_prob > 0) else "Neutral"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

@app.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(request: Request, api_key: str = Security(get_api_key)):
    """Phase 7: Monitoring Endpoint"""
    if not monitor:
        raise HTTPException(status_code=500, detail="Monitor not initialized")
    
    return {
        "latest_performance": monitor.get_latest_performance(),
        "recent_alerts": monitor.get_alert_summary(limit=10)
    }

# Mount dashboard frontend at the end to avoid routing conflicts
import os
if os.path.exists("dashboard"):
    app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
