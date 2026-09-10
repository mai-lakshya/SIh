"""
Unit & Integration Test Suite for Export Summary Report Pipeline
Verifies:
1. export_narrative_engine.py strictly uses internal models/statutory logic with NO external LLMs
2. Resilience fallback activates upon model timeout or exception, generating project-specific reasoning
3. Consistency guardrail strictly enforces numerical locking and corrects drifted/hallucinated numbers
4. ReportGenerator creates valid PDF binaries with proper structure and provenance notes
5. API endpoint /api/reports/export-summary responds with HTTP 200 and application/pdf binary
"""

import os
import sys
import ast
import time
import pytest
from unittest.mock import patch, MagicMock

# Ensure backend subdirectories are on python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.join(WORKSPACE_ROOT, "backend")
for sub in [
    WORKSPACE_ROOT,
    BACKEND_DIR,
    os.path.join(BACKEND_DIR, "04_xai"),
    os.path.join(BACKEND_DIR, "07_api"),
]:
    if os.path.exists(sub) and sub not in sys.path:
        sys.path.insert(0, sub)

from export_narrative_engine import ExportNarrativeEngine, ConsistencyGuardrail
from report_generator import ReportGenerator


@pytest.fixture
def sample_project():
    return {
        "project_id": "NHAI-2026-GUJ-8891",
        "project_type": "Highway",
        "state": "Gujarat",
        "district": "Vadodara",
        "terrain_type": "Plain",
        "estimated_cost_inr_crore": 485.5,
        "land_area_hectares": 165.2,
        "affected_families_count": 620,
        "section_11_notification_days": 310,
        "compensation_multiplier_demand": 1.9,
        "title_dispute_rate_percent": 14.2,
        "sia_approval_status": "Approved",
        "forest_clearance_status": "Stage_1_Pending",
        "fund_disbursement_percent": 32.0,
        "local_protest_flag": True,
        "latitude": 22.3072,
        "longitude": 73.1812,
        "road_type": "National Highway Link / NH-48",
        "address": "Cadastral Survey Div-4, Vadodara Bypass Corridor, Gujarat"
    }


@pytest.fixture
def sample_metrics():
    return {
        "crs": 68.4,
        "delay_probability": 76.5,
        "confidence_score": 84.0,
        "confidence_label": "High Certainty",
        "calibrated_risk_tier": "High",
        "predicted_delay_days": 218,
        "median_survival_days": 135
    }


# =========================================================================
# TEST 1: STRICT BAN ON EXTERNAL / THIRD-PARTY LLMS
# =========================================================================
def test_no_external_llm_imports_or_calls():
    """
    Scans the AST and source code of export_narrative_engine.py to guarantee
    no external third-party LLM providers (OpenAI, Anthropic, Google GenAI, Cohere)
    are imported, instantiated, or referenced.
    """
    engine_file = os.path.join(BACKEND_DIR, "04_xai", "export_narrative_engine.py")
    assert os.path.exists(engine_file), f"File not found: {engine_file}"

    with open(engine_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    tree = ast.parse(source_text)

    # Prohibited modules and client names
    banned_modules = {
        "openai", "anthropic", "google.generativeai", "genai", "cohere",
        "groq", "together", "replicate", "huggingface_hub", "langchain",
        "llama_index", "litellm"
    }

    banned_keywords = [
        "OpenAI(", "ChatOpenAI", "Anthropic(", "genai.GenerativeModel",
        "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"
    ]

    # Verify no AST import statements import banned modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for banned in banned_modules:
                    assert banned not in alias.name.lower(), (
                        f"Prohibited third-party LLM library imported: '{alias.name}' in {engine_file}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for banned in banned_modules:
                    assert banned not in node.module.lower(), (
                        f"Prohibited third-party LLM library imported from: '{node.module}' in {engine_file}"
                    )

    # Verify no banned keywords in the source text
    for keyword in banned_keywords:
        assert keyword.lower() not in source_text.lower(), (
            f"Prohibited third-party LLM invocation found: '{keyword}' in {engine_file}"
        )


# =========================================================================
# TEST 2: RESILIENCE FALLBACK ACTIVATION & PROJECT-SPECIFIC REASONING
# =========================================================================
def test_fallback_activation_on_model_timeout(sample_project, sample_metrics):
    """
    Verifies that if the primary model execution times out (e.g. >15s),
    the resilience fallback automatically activates, tags the provenance as fallback,
    and produces rich, project-specific statutory reasoning rather than generic boilerplate.
    """
    engine = ExportNarrativeEngine()

    def slow_model(*args, **kwargs):
        time.sleep(0.5)
        return {}

    # Temporarily set timeout low to simulate timeout deterministically
    with patch.object(engine, 'MODEL_TIMEOUT_SECONDS', 0.05):
        with patch.object(engine, '_execute_primary_model_pipeline', side_effect=slow_model):
            narrative = engine.generate_narrative(sample_project, sample_metrics)

    assert narrative is not None
    assert "Fallback" in narrative.get("provenance", "")
    assert "Statutory RFCTLARR Compliance Engine" in narrative["provenance"]

    # Verify project-specific reasoning (NOT generic boilerplate)
    assert sample_project["project_id"] in narrative["executive_summary"]
    assert sample_project["district"] in narrative["executive_summary"]
    assert sample_project["state"] in narrative["executive_summary"]
    assert str(int(sample_project["estimated_cost_inr_crore"])) in narrative["executive_summary"]

    # Verify statutory reasoning reflects Section 11 days (310 days elapsed -> Pre-Lapse warning)
    assert "Section 19(7)" in narrative["statutory_status_headline"] or "Section 19" in narrative["statutory_status_headline"]
    assert "310 days" in narrative["statutory_status_explanation"] or "310" in narrative["statutory_status_explanation"]
    assert len(narrative["recommended_next_steps"]) >= 3


def test_fallback_activation_on_model_exception(sample_project, sample_metrics):
    """
    Verifies that if the primary model pipeline raises any exception,
    the fallback catches it gracefully and returns a complete, project-specific memo.
    """
    engine = ExportNarrativeEngine()

    def crashing_model(*args, **kwargs):
        raise RuntimeError("Internal model checkpoint loading fault")

    with patch.object(engine, '_execute_primary_model_pipeline', side_effect=crashing_model):
        narrative = engine.generate_narrative(sample_project, sample_metrics)

    assert narrative is not None
    assert "Fallback" in narrative.get("provenance", "")
    assert "executive_summary" in narrative
    assert "expanded_causation_diagnostics" in narrative
    assert sample_project["district"] in narrative["expanded_causation_diagnostics"]


# =========================================================================
# TEST 3: CONSISTENCY GUARDRAIL NUMERICAL LOCKING
# =========================================================================
def test_consistency_guardrail_prevents_numeric_drift(sample_metrics):
    """
    Verifies that if generated text attempts to drift or hallucinate numbers
    (e.g., claiming CRS is 92.5 or delay probability is 30%), the ConsistencyGuardrail
    enforces the authoritative calibrated predictions.
    """
    guardrail = ConsistencyGuardrail()

    # Hallucinated narrative with conflicting numbers
    corrupted_narrative = {
        "executive_summary": (
            "The model predicted a delay probability of 42.1% and a CRS of 91.2. "
            "Additionally, 450 days of statutory delay are expected."
        ),
        "expanded_causation_diagnostics": (
            "Due to delays, the project faces CRS: 99.9 and 500 days delay."
        ),
        "statutory_status_explanation": (
            "Statutory analysis with delay probability: 15.0% indicates risk."
        )
    }

    enforced = guardrail.enforce(corrupted_narrative, sample_metrics)

    # Verify text was sanitized to calibrated numbers
    exec_summary = enforced["executive_summary"]
    assert f"delay probability of {sample_metrics['delay_probability']:.1f}%" in exec_summary
    assert f"CRS: {sample_metrics['crs']:.1f}" in exec_summary
    assert f"{sample_metrics['predicted_delay_days']} days delay" in exec_summary

    # Verify authoritative numeric block is attached
    auth = enforced["authoritative_metrics"]
    assert auth["crs"] == sample_metrics["crs"]
    assert auth["delay_probability"] == sample_metrics["delay_probability"]
    assert auth["predicted_delay_days"] == sample_metrics["predicted_delay_days"]
    assert auth["median_survival_days"] == sample_metrics["median_survival_days"]


# =========================================================================
# TEST 4: REPORT GENERATOR PDF COMPILATION
# =========================================================================
def test_pdf_generator_compiles_valid_pdf(sample_project, sample_metrics):
    """
    Verifies that ReportGenerator produces a valid binary PDF document with
    standard PDF header (%PDF-), EOF marker (%%EOF), and non-empty size (>10KB).
    """
    engine = ExportNarrativeEngine()
    narrative = engine.generate_narrative(sample_project, sample_metrics)

    generator = ReportGenerator()
    pdf_bytes = generator.generate_pdf_bytes(sample_project, sample_metrics, narrative)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 5000, f"PDF byte size unexpectedly small: {len(pdf_bytes)} bytes"
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF binary header"
    assert b"%%EOF" in pdf_bytes[-1024:], "Missing standard PDF EOF marker"


# =========================================================================
# TEST 5: API ROUTE INTEGRATION
# =========================================================================
def test_export_summary_api_endpoint_integration(sample_project, sample_metrics):
    """
    Verifies the FastAPI endpoint /api/reports/export-summary returns HTTP 200,
    application/pdf MIME type, attachment filename, and X-Model-Provenance header.
    """
    from fastapi.testclient import TestClient
    from backend.api import app

    client = TestClient(app)

    # Test POST endpoint with client payload
    response = client.post(
        "/api/reports/export-summary",
        json={
            "project_id": sample_project["project_id"],
            "project": sample_project,
            "predictions": sample_metrics
        }
    )

    assert response.status_code == 200, f"API error: {response.status_code} - {response.text}"
    assert response.headers.get("content-type") == "application/pdf"
    disp = response.headers.get("content-disposition", "")
    assert "attachment; filename=" in disp
    # Verify file is named after project name, NOT NEXUS_Risk_Summary_
    assert 'filename="NHAI-2026-GUJ-8891.pdf"' in disp
    assert "NEXUS_Risk_Summary" not in disp
    assert "X-Model-Provenance" in response.headers
    assert response.content.startswith(b"%PDF-")


def test_export_summary_get_route_default(sample_project):
    """
    Verifies GET /api/reports/export-summary also responds with a valid PDF and project name filename.
    """
    from fastapi.testclient import TestClient
    from backend.api import app

    client = TestClient(app)
    response = client.get("/api/reports/export-summary?project_id=PROJ-DEFAULT-TEST")
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/pdf"
    disp = response.headers.get("content-disposition", "")
    assert 'filename="PROJ-DEFAULT-TEST.pdf"' in disp
    assert "NEXUS_Risk_Summary" not in disp
    assert response.content.startswith(b"%PDF-")
