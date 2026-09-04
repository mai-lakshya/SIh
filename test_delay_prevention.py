import pytest
import concurrent.futures
from typing import Dict, Any, List

from recommendation_engine import (
    TaskNode,
    CriticalChainEngine,
    RecommendationEngine,
    get_default_infrastructure_schedule,
    calculate_roi_for_recommendation
)


# =====================================================================
# 1. UNIT TESTS: CRITICAL PATH & SCHEDULE CALCULATIONS
# =====================================================================

def test_standard_schedule_critical_path():
    """
    Validate standard forward/backward pass and critical path detection.
    Graph:
      A (10d) -> B (20d) -> D (10d)  [Path 1: 40 days - Critical]
      A (10d) -> C (10d) -> D (10d)  [Path 2: 30 days - Non-critical, slack=10]
    """
    engine = CriticalChainEngine()
    tasks = [
        TaskNode(task_id="A", name="Task A", duration_days=10.0, dependencies=[]),
        TaskNode(task_id="B", name="Task B", duration_days=20.0, dependencies=["A"]),
        TaskNode(task_id="C", name="Task C", duration_days=10.0, dependencies=["A"]),
        TaskNode(task_id="D", name="Task D", duration_days=10.0, dependencies=["B", "C"]),
    ]

    res = engine.compute_schedule(tasks)
    t_map = res["tasks"]

    # Duration check
    assert res["total_duration_days"] == 40.0

    # Timing checks
    assert t_map["A"].earliest_start == 0.0
    assert t_map["A"].earliest_finish == 10.0

    assert t_map["B"].earliest_start == 10.0
    assert t_map["B"].earliest_finish == 30.0
    assert t_map["B"].slack == 0.0
    assert t_map["B"].is_critical is True

    assert t_map["C"].earliest_start == 10.0
    assert t_map["C"].earliest_finish == 20.0
    assert t_map["C"].slack == 10.0
    assert t_map["C"].is_critical is False

    assert t_map["D"].earliest_start == 30.0
    assert t_map["D"].earliest_finish == 40.0
    assert t_map["D"].is_critical is True

    # Critical path list
    assert "A" in res["critical_path_tasks"]
    assert "B" in res["critical_path_tasks"]
    assert "D" in res["critical_path_tasks"]
    assert "C" not in res["critical_path_tasks"]


def test_zero_delay_state():
    """
    Edge Case 1: When project is on schedule with positive buffer,
    the engine must identify zero-delay state and produce proactive
    buffer-preservation recommendations rather than emergency crash alerts.
    """
    engine = RecommendationEngine()
    tasks = [
        TaskNode(task_id="T1", name="Pre-demarcation", duration_days=20.0, deadline_days=50.0),
        TaskNode(task_id="T2", name="Survey", duration_days=30.0, deadline_days=100.0, dependencies=["T1"]),
    ]
    meta = {
        "schedule_tasks": [t.to_dict() for t in tasks],
        "target_completion_days": 100.0  # Finished at 50, buffer = +50 days
    }

    recs = engine.generate_recommendations(risk_drivers=[], project_metadata=meta)
    
    # Check that zero-delay recommendation is surfaced
    zero_delay_recs = [r for r in recs if r.get("is_zero_delay_state") is True]
    assert len(zero_delay_recs) == 1
    rec = zero_delay_recs[0]
    assert rec["priority"] == "Low"
    assert "buffer" in rec["actions"][0].lower()


def test_negative_buffer_times():
    """
    Edge Case 2: When milestone deadline is breached or target completion
    is less than project finish, negative buffer times must be computed and
    emergency compression mitigations generated.
    """
    engine = RecommendationEngine()
    tasks = [
        TaskNode(task_id="T1", name="Section 11 Gazette", duration_days=60.0, deadline_days=40.0),  # Slack: -20 days
    ]
    meta = {
        "schedule_tasks": [t.to_dict() for t in tasks],
        "target_completion_days": 40.0
    }

    recs = engine.generate_recommendations(risk_drivers=[], project_metadata=meta)
    neg_recs = [r for r in recs if "Negative Buffer" in r.get("buffer_status", "")]
    assert len(neg_recs) >= 1
    assert neg_recs[0]["priority"] == "Critical"
    assert "20" in neg_recs[0]["issue"] or "breach" in neg_recs[0]["issue"].lower()


def test_cascading_dependencies():
    """
    Edge Case 3: Upstream milestone delays propagate downstream when free slack is 0.
    """
    engine = CriticalChainEngine()
    # A (30d) -> B (40d) -> C (20d)
    tasks = [
        TaskNode(task_id="A", name="SIA Clearance", duration_days=30.0, dependencies=[]),
        TaskNode(task_id="B", name="Sec 11 Notice", duration_days=40.0, dependencies=["A"]),
        TaskNode(task_id="C", name="Possession", duration_days=20.0, dependencies=["B"]),
    ]

    res = engine.compute_schedule(tasks)
    # A has successors B, free slack = 0, duration = 30 -> ripple = 30
    assert res["tasks"]["A"].cascading_impact_days == 30.0
    assert "A" in res["cascading_tasks"]


def test_resource_saturation():
    """
    Edge Case 4: Concurrent milestones demanding resources beyond capacity
    must be flagged as saturated bottlenecks.
    """
    engine = CriticalChainEngine(resource_capacities={"legal_officers": 3.0})
    # T1 and T2 run concurrently from day 0 to day 30, both needing 2 legal officers (total 4 > cap 3)
    tasks = [
        TaskNode(task_id="T1", name="Hearing 1", duration_days=30.0, resources={"legal_officers": 2.0}),
        TaskNode(task_id="T2", name="Hearing 2", duration_days=30.0, resources={"legal_officers": 2.0}),
    ]

    res = engine.compute_schedule(tasks)
    assert "legal_officers" in res["resource_bottlenecks"]
    assert res["resource_bottlenecks"]["legal_officers"]["peak_demand"] == 4.0
    assert res["tasks"]["T1"].resource_saturated is True
    assert res["tasks"]["T2"].resource_saturated is True


def test_cycle_detection_graceful_fallback():
    """
    Ensure circular dependency loops (A -> B -> C -> A) are detected,
    logged, and broken gracefully without infinite recursion or hangs.
    """
    engine = CriticalChainEngine()
    tasks = [
        TaskNode(task_id="A", name="A", duration_days=10.0, dependencies=["C"]),
        TaskNode(task_id="B", name="B", duration_days=10.0, dependencies=["A"]),
        TaskNode(task_id="C", name="C", duration_days=10.0, dependencies=["B"]),
    ]

    # Must complete safely without raising an unhandled exception or hanging
    res = engine.compute_schedule(tasks)
    assert res["total_duration_days"] >= 10.0
    assert len(res["tasks"]) == 3


def test_input_validation_and_sanitization():
    """
    Verify negative durations, dangling dependency IDs, and malformed inputs
    are sanitized and clamped.
    """
    engine = CriticalChainEngine()
    raw_tasks = [
        {"task_id": "T1", "name": "Task 1", "duration_days": -15.0, "dependencies": ["NON_EXISTENT_PARENT"]},
        {"task_id": "T2", "name": "Task 2", "duration_days": "invalid", "deadline_days": -5.0},
        {"task_id": "T1", "name": "Task 1 Duplicate", "duration_days": 10.0}  # duplicate ID
    ]

    sanitized = engine.validate_and_sanitize(raw_tasks)
    assert len(sanitized) == 3
    # Duration clamped
    assert sanitized[0].duration_days == 0.0
    # Dangling parent dropped
    assert sanitized[0].dependencies == []
    # Invalid duration defaulted
    assert sanitized[1].duration_days == 0.0
    # Negative deadline clamped
    assert sanitized[1].deadline_days == 0.0
    # Duplicate ID deduplicated
    assert sanitized[2].task_id != sanitized[0].task_id


def test_deterministic_sorting():
    """
    Verify that multiple calls with identical input produce strictly identical
    recommendation order and IDs.
    """
    engine = RecommendationEngine()
    drivers = [
        ("title_dispute_rate_percent", 0.75),
        ("forest_clearance_status", 0.65),
        ("fund_disbursement_percent", 0.40)
    ]
    meta = {
        "state": "Odisha",
        "terrain_type": "Forest_Eco_Sensitive",
        "title_dispute_rate_percent": 22.0
    }

    run1 = engine.generate_recommendations(drivers, meta)
    run2 = engine.generate_recommendations(drivers, meta)

    assert len(run1) == len(run2)
    for r1, r2 in zip(run1, run2):
        assert r1["recommendation_id"] == r2["recommendation_id"]
        assert r1["priority"] == r2["priority"]
        assert r1["issue"] == r2["issue"]


# =====================================================================
# 2. CONCURRENCY & INTEGRATION TESTS
# =====================================================================

def test_concurrent_inference_thread_safety():
    """
    Verify that multiple concurrent worker threads invoking RiskAnalysisSystem.predict()
    do not encounter race conditions, dictionary mutation errors, or deadlocks.
    """
    from risk_analysis_system import RiskAnalysisSystem
    import pandas as pd

    system = RiskAnalysisSystem(
        pipeline_path='pipeline.joblib',
        ensemble_path='ensemble.joblib',
        timeline_path='timeline.joblib'
    )

    sample_row = pd.DataFrame([{
        'project_id': 'CONCUR-TEST',
        'state': 'Maharashtra',
        'terrain_type': 'Urban',
        'project_type': 'Highway',
        'estimated_cost_inr_crore': 1450.0,
        'land_area_hectares': 180.0,
        'affected_families_count': 650,
        'title_dispute_rate_percent': 15.0,
        'compensation_multiplier_demand': 2.0,
        'sia_approval_status': 'Pending',
        'forest_clearance_status': 'Pending',
        'fund_disbursement_percent': 25.0,
        'local_protest_flag': True,
        'project_start_year': 2024,
        'C_r': 0.5, 'F_r': 0.5, 'H_r': 0.5, 'W_r': 0.5, 'P_r': 0.5
    }])

    def worker_fn(idx):
        row = sample_row.copy()
        row['project_id'] = f"CONCUR-TEST-{idx}"
        res = system.predict(row)
        return res['predictions']['delay_probability'], len(res.get('recommendations', []))

    # Execute 8 concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_fn, i) for i in range(8)]
        results = [f.result() for f in futures]

    assert len(results) == 8
    # Delay probability must be consistent across runs
    probs = [r[0] for r in results]
    assert all(p == probs[0] for p in probs)
    # Recommendations generated for all
    assert all(r[1] > 0 for r in results)


def test_api_delay_prevention_integration():
    """
    Integration test using FastAPI TestClient to ensure delay-prevention
    recommendations, buffer status, and dynamic ROI surface through /predict.
    """
    from fastapi.testclient import TestClient
    from api import app, load_artifacts

    load_artifacts()
    client = TestClient(app)

    payload = {
        "project_id": "API-SCHED-INTEG-01",
        "state": "Maharashtra",
        "terrain_type": "Urban",
        "project_type": "Highway",
        "estimated_cost_inr_crore": 1450.0,
        "land_area_hectares": 180.0,
        "title_dispute_rate_percent": 25.0,
        "fund_disbursement_percent": 15.0,
        "sia_approval_status": "Pending",
        "forest_clearance_status": "Pending",
        "schedule_tasks": [
            {"task_id": "T1", "name": "Clearance 1", "duration_days": 40.0, "deadline_days": 20.0},
            {"task_id": "T2", "name": "Survey 1", "duration_days": 30.0, "dependencies": ["T1"]}
        ]
    }

    response = client.post("/predict", json=payload, headers={"X-API-Key": "super-secret-token"})
    assert response.status_code == 200

    data = response.json()
    assert "prescriptive_actions" in data
    assert len(data["prescriptive_actions"]) > 0

    first_rec = data["prescriptive_actions"][0]
    assert "title" in first_rec
    assert "description" in first_rec
    assert "avoided_delay" in first_rec
    assert "cost_savings_cr" in first_rec
    assert "roi" in first_rec
    assert "buffer_status" in first_rec
    assert first_rec["avoided_delay"] > 0
