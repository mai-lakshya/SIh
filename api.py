from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import logging
import json
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from risk_analysis_system import RiskAnalysisSystem
from monitor import ModelMonitor
from recommendation_engine import calculate_roi_for_recommendation
from ai_advisor import AIAdvisor, PromptSecurityValidator, DomainGroundingValidator, IndianContextNormalizer

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
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header

class ProjectPayload(BaseModel):
    project_id: Optional[str] = 'NHAI-UNKNOWN'
    state: str
    district: Optional[str] = 'Unknown'
    land_area_hectares: float = Field(..., gt=0.0, description="Land area in hectares, must be positive")
    land_area_log: Optional[float] = 5.0
    project_type: str
    terrain_type: str
    estimated_cost_inr_crore: float = Field(..., gt=0.0, description="Estimated project cost in INR Crores, must be positive")
    affected_families_count: Optional[int] = Field(default=500, ge=0)
    title_dispute_rate_percent: Optional[float] = Field(default=5.0, ge=0.0, le=100.0)
    local_protest_flag: Optional[bool] = False
    compensation_multiplier_demand: Optional[float] = Field(default=1.5, ge=0.0)
    sia_approval_status: Optional[str] = 'Pending'
    sia_approval_status_risk_score: Optional[float] = 0.5
    section_11_notification_days: Optional[int] = 30
    forest_clearance_status: Optional[str] = 'Not_Required'
    forest_clearance_status_risk_score: Optional[float] = 0.5
    fund_disbursement_percent: Optional[float] = Field(default=10.0, ge=0.0, le=100.0)
    project_start_year: Optional[int] = 2022
    project_age_years: Optional[int] = 1
    schedule_tasks: Optional[List[Dict[str, Any]]] = None
    target_completion_days: Optional[float] = None

class AIAdvisoryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    project_metadata: Optional[Dict[str, Any]] = None

# Global variables
system: RiskAnalysisSystem = None
monitor: ModelMonitor = None

@app.on_event("startup")
def load_artifacts():
    global system, monitor
    try:
        import os
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
        "monitor_ready": monitor is not None
    }

@app.get("/")
def serve_home():
    from fastapi.responses import FileResponse, RedirectResponse
    if os.path.exists("dashboard/index.html"):
        return FileResponse("dashboard/index.html")
    return RedirectResponse(url="/docs")

@app.post("/ai/advisory")
@limiter.limit("20/minute")
async def get_ai_advisory(request: Request, req: AIAdvisoryRequest, api_key: str = Security(get_api_key)):
    advisor = AIAdvisor()
    res = await run_in_threadpool(advisor.generate_advisory, req.query, req.context, req.project_metadata)
    return res

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_risk(request: Request, payload: ProjectPayload, api_key: str = Security(get_api_key)):
    if not system:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Security validation on free text / identifiers
    security_validator = PromptSecurityValidator()
    for field_val in [payload.project_id, payload.district]:
        if field_val:
            is_inj, reason = security_validator.detect_injection(str(field_val))
            if is_inj:
                raise HTTPException(status_code=400, detail=f"Security rejection: {reason}")

    payload_dict = payload.model_dump(exclude_unset=True) if hasattr(payload, 'model_dump') else payload.dict(exclude_unset=True)
    # Remove schedule-specific metadata fields so they don't pollute the ML feature dataframe
    sched_tasks = payload_dict.pop('schedule_tasks', None)
    target_comp = payload_dict.pop('target_completion_days', None)
    
    raw_payload = pd.DataFrame([payload_dict])
    
    # Fill defaults for Phase 6 Pipeline
    for col in ['C_r', 'F_r', 'H_r', 'W_r', 'P_r']:
        if col not in raw_payload:
            raw_payload[col] = 0.5
            
    # Drop section_11_notification_days as it was dropped in training
    if 'section_11_notification_days' in raw_payload:
        raw_payload = raw_payload.drop(columns=['section_11_notification_days'])
            
    metadata = {
        'project_id': payload.project_id,
        'estimated_cost_inr_crore': payload.estimated_cost_inr_crore,
        'terrain_type': payload.terrain_type,
        'sia_approval_status': payload.sia_approval_status,
        'forest_clearance_status': payload.forest_clearance_status,
        'title_dispute_rate_percent': payload.title_dispute_rate_percent,
        'local_protest_flag': payload.local_protest_flag,
        'fund_disbursement_percent': payload.fund_disbursement_percent,
        'section_11_notification_days': payload.section_11_notification_days,
        'schedule_tasks': payload.schedule_tasks,
        'target_completion_days': payload.target_completion_days
    }

    try:
        # Non-blocking threadpool offloading to preserve event loop concurrency
        result = await run_in_threadpool(system.predict, raw_payload, metadata=metadata)
        
        # Prescriptive Actions & Dynamic ROI Calculations
        raw_recs = result.get('recommendations', [])
        project_cost_inr = payload.estimated_cost_inr_crore * 10_000_000
        delay_cost_per_day = max(100_000, (project_cost_inr * 0.12) / 365)
        
        # Prepare processed feature sample for data-driven simulation
        try:
            X_sample = system.pipeline.transform(raw_payload) if hasattr(system, 'pipeline') and system.pipeline is not None else None
        except Exception:
            X_sample = None
        model_inst = getattr(system, 'hybrid_predictor', None)

        prescriptive_actions = []
        for rec in raw_recs:
            try:
                roi_info = calculate_roi_for_recommendation(
                    rec,
                    project_cost=project_cost_inr,
                    delay_cost_per_day=delay_cost_per_day,
                    model=model_inst,
                    X_sample=X_sample
                )
            except Exception as roi_err:
                logging.warning("ROI calculation failed for rec %s (%s); applying fallback", rec.get('issue'), roi_err)
                roi_info = {
                    'estimated_delay_days_saved': 15.0,
                    'cost_savings': delay_cost_per_day * 15.0,
                    'roi_percentage': 150.0
                }
            
            title = rec.get('issue', 'Mitigation Action')
            if 'actions' in rec and rec['actions']:
                title = rec['actions'][0]
                desc = rec['actions'][1] if len(rec['actions']) > 1 else rec.get('issue', '')
            else:
                desc = rec.get('expected_impact', 'Operational intervention')
                
            # Allow authentic values to surface without artificial floors
            delay_saved = round(float(roi_info.get('estimated_delay_days_saved', 0.0)), 1)
            cost_savings_cr = round(float(roi_info.get('cost_savings', 0.0)) / 10_000_000, 2)
            roi_pct = round(float(roi_info.get('roi_percentage', 0.0)), 1)
            
            prescriptive_actions.append({
                "title": title,
                "description": desc,
                "issue": rec.get('issue', title),
                "actions": rec.get('actions', [title]),
                "priority": rec.get('priority', 'Medium'),
                "timeframe": rec.get('timeframe', 'Short-term'),
                "expected_impact": rec.get('expected_impact', 'Risk reduction'),
                "avoided_delay": delay_saved,
                "avoided_delay_days": delay_saved,
                "cost_savings": cost_savings_cr,
                "cost_savings_cr": cost_savings_cr,
                "roi": roi_pct,
                "roi_percentage": roi_pct,
                "roi_percent": roi_pct,
                "buffer_status": rec.get("buffer_status", "Active Schedule Path")
            })

        # Map to Frontend Schema
        frontend_response = {
            "project_id": payload.project_id,
            "predictions": {
                "delay_probability": round(result['predictions']['delay_probability'] * 100, 1),
                "calibrated_risk_tier": result['predictions']['calibrated_risk_tier'],
                "predicted_delay_days": int(result['predictions']['predicted_delay_days']),
                "median_survival_days": int(result['timeline']['median_survival_days'])
            },
            "explainability": {
                "top_risk_drivers": result['explanation']['risk_drivers'],
                "category_breakdown": result['explanation']['category_breakdown']
            },
            "recommendations": prescriptive_actions,
            "prescriptive_actions": prescriptive_actions
        }
        return frontend_response
    except Exception as e:
        logging.error("Inference pipeline failed for project %s: %s", payload.project_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

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
