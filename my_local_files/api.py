from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import os
import logging
import json
from typing import Optional, List, Dict, Any
from starlette.concurrency import run_in_threadpool

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    has_slowapi = True
except ImportError:
    has_slowapi = False
    class RateLimitExceeded(Exception):
        pass
    def _rate_limit_exceeded_handler(request, exc):
        pass
    class Limiter:
        def __init__(self, *args, **kwargs):
            pass
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    def get_remote_address(request=None):
        return "127.0.0.1"

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
if has_slowapi:
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
    for path in ["dashboard/test_dashboard.html", "dashboard/index.html", "test_dashboard.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return RedirectResponse(url="/docs")

def _prepare_df(payload_dict: dict) -> pd.DataFrame:
    # 1. Normalize and dynamically derive statutory clearance risk scores
    sia_raw = str(payload_dict.get('sia_approval_status', 'Pending')).strip()
    sia_norm = sia_raw.lower().replace(' ', '_').replace('-', '_')
    sia_score_map = {
        'approved': 0.0,
        'exempted': 0.0,
        'in_progress': 0.4,
        'pending': 0.75,
        'rejected': 1.0
    }
    payload_dict['sia_approval_status_risk_score'] = sia_score_map.get(sia_norm, 0.5)

    fc_raw = str(payload_dict.get('forest_clearance_status', 'Not_Required')).strip()
    fc_norm = fc_raw.lower().replace(' ', '_').replace('-', '_')
    fc_score_map = {
        'not_required': 0.0,
        'approved': 0.0,
        'stage_2': 0.2,
        'stage_1': 0.4,
        'stage_1_approved': 0.4,
        'in_progress': 0.6,
        'stage_1_pending': 0.8,
        'pending': 0.8,
        'rejected': 1.0
    }
    payload_dict['forest_clearance_status_risk_score'] = fc_score_map.get(fc_norm, 0.5)

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

@app.post("/ai/advisory")
@limiter.limit("20/minute")
async def get_ai_advisory(request: Request, req: AIAdvisoryRequest, api_key: str = Security(get_api_key)):
    advisor = AIAdvisor()
    res = await run_in_threadpool(advisor.generate_advisory, req.query, req.context, req.project_metadata)
    return res

@app.post("/predict")
@limiter.limit("60/minute")
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
    
    raw_payload = _prepare_df(payload_dict)
            
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
        survival_curve = _extract_survival_curve(raw_payload)

        # Meta coefficients from StackingClassifier
        meta_coefs = {}
        if system.explainer and hasattr(system.explainer, 'meta_coefficients'):
            meta_coefs = {k: round(float(v), 3) for k, v in system.explainer.meta_coefficients.items()}

        # Top full features with signed TreeSHAP impacts
        feature_labels = {
            "F_r": "Fund Disbursement Risk Ratio (F_r)",
            "C_r": "Compensation Demand Ratio (C_r)",
            "P_r": "Protest & Agitation Risk Factor (P_r)",
            "H_r": "Historical State Delay Ratio (H_r)",
            "W_r": "Weather Vulnerability Index (W_r)",
            "affected_families_count": "Affected Families Count",
            "title_dispute_rate_percent": "Title Dispute Rate (%)",
            "local_protest_flag": "Local Agitation / Protest Flag",
            "compensation_multiplier_demand": "Compensation Multiplier Demand",
            "forest_clearance_status": "Forest Clearance Status",
            "forest_clearance_status_risk_score": "Forest Clearance Risk Score",
            "sia_approval_status": "SIA Approval Status",
            "sia_approval_status_risk_score": "SIA Approval Risk Score",
            "fund_disbursement_percent": "Fund Disbursement Progress (%)",
            "terrain_type": "Terrain Complexity",
            "project_age_years": "Project Age (Years)",
            "project_start_year": "Project Start Year",
            "land_area_hectares": "Land Area (Hectares)"
        }

        full_feats = result['explanation'].get('local_explanation_full', [])
        for f in full_feats:
            f['feature_label'] = feature_labels.get(f.get('feature'), f.get('feature', '').replace('_', ' ').title())
        # Sort full features by absolute impact
        full_feats_sorted = sorted(full_feats, key=lambda x: abs(x.get('shap_impact', 0)), reverse=True)[:10]

        top_drivers = result['explanation'].get('risk_drivers', [])
        for d in top_drivers:
            d['feature_label'] = feature_labels.get(d.get('feature'), d.get('feature', '').replace('_', ' ').title())

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
        seen_titles = set()
        template_cursor = {}

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
            
            # Determine template/category key for cycling actions
            t_key = rec.get('template_key') or rec.get('category') or rec.get('source', 'default')
            actions = rec.get('actions', [])
            
            title = None
            desc = None
            
            if actions:
                cursor = template_cursor.get(t_key, 0)
                while cursor < len(actions):
                    candidate = actions[cursor].strip()
                    if candidate.lower() not in seen_titles:
                        title = candidate
                        # Use next action as description if available, otherwise issue or expected impact
                        if cursor + 1 < len(actions):
                            desc = actions[cursor + 1].strip()
                            template_cursor[t_key] = cursor + 2
                        else:
                            desc = rec.get('expected_impact') or rec.get('issue', 'Operational mitigation intervention')
                            template_cursor[t_key] = cursor + 1
                        break
                    cursor += 1
                # If all distinct actions for this category are exhausted, skip adding redundant cards
                if not title:
                    continue
            else:
                candidate_issue = rec.get('issue', 'Mitigation Action').strip()
                if candidate_issue.lower() not in seen_titles:
                    title = candidate_issue
                    desc = rec.get('expected_impact', 'Operational intervention')

            # Skip duplicate / exhausted recommendations so each card is unique and impactful
            if not title or title.lower() in seen_titles:
                continue

            seen_titles.add(title.lower())
                
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
                "delay_saved_days": int(round(delay_saved)),
                "avoided_delay": delay_saved,
                "avoided_delay_days": delay_saved,
                "cost_saved_cr": cost_savings_cr,
                "cost_savings": cost_savings_cr,
                "roi": roi_pct,
                "roi_percentage": roi_pct,
                "roi_percent": int(round(roi_pct)),
                "buffer_status": rec.get("buffer_status", "Active Schedule Path")
            })

        # Map to Frontend Schema
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
            "recommendations": prescriptive_actions,
            "prescriptive_actions": prescriptive_actions
        }
        return frontend_response
    except Exception as e:
        logging.error("Inference pipeline failed for project %s: %s", payload.project_id, e, exc_info=True)
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
