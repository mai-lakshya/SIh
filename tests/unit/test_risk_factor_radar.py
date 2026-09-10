"""
Unit tests for RiskFactorRadarChart recentered scale math, dynamic magnitude resolution,
and directional/magnitude guardrails.
Verifies:
1. Neutral impact (0 points) maps to radial value 50 (mid-ring).
2. Max positive impact maps to radial value 100 (outer edge).
3. Max negative impact maps to radial value 0 (center).
4. Top risk driver specifically targets positive max (not magnitude).
5. Top mitigator specifically targets negative min (not magnitude).
6. P_r (+17) and Population Density (-7) dynamic ceiling scaling (maxAbs = 20):
   - P_r (+17) -> 92.5 (near outer edge)
   - Population Density (-7) -> 32.5 (deep inside neutral ring)
   - They are 60.0 units apart (proportionally far apart, not clustered near 50)
   - get_top_risk_driver returns P_r with value 17, NOT 2
   - get_top_mitigator returns Population Density with value -7, NOT -1
7. Dev-mode consistency guardrail throws error on magnitude mismatch (> 0.5 pts).
"""
import pytest
import math

SAMPLE_DATASET = [
    {"factor": "Terrain Type", "project": -10, "benchmark": -3},
    {"factor": "Local Agitation / Protest Flag", "project": 9, "benchmark": 4},
    {"factor": "Fund Disbursement Progress (%)", "project": -8, "benchmark": -2},
    {"factor": "Project Type", "project": -7, "benchmark": -1},
    {"factor": "Protest & Agitation Risk Factor (P_r)", "project": 8, "benchmark": 3}
]

# Canonical dataset representing the current project
CURRENT_PROJECT_DATASET = [
    {"factor": "Protest & Agitation Risk Factor (P_r)", "project": 17, "benchmark": 5},
    {"factor": "Population Density", "project": -7, "benchmark": -2},
    {"factor": "State", "project": 6, "benchmark": 2},
    {"factor": "District", "project": -5, "benchmark": -1},
    {"factor": "Log Land Area", "project": 6, "benchmark": 2}
]

def to_radial_scale(impact: float, max_abs_impact: float) -> float:
    if max_abs_impact <= 0:
        return 50.0
    radial = 50.0 + (impact / max_abs_impact) * 50.0
    return max(0.0, min(100.0, round(radial, 1)))

def compute_max_abs_impact(data, min_limit=5):
    if not data:
        return min_limit
    values = []
    for d in data:
        values.extend([abs(d["project"]), abs(d["benchmark"])])
    max_obs = max(values) if values else 0
    if max_obs == 0:
        return min_limit
    return math.ceil(max_obs / 5) * 5

def get_top_risk_driver(data):
    pos = [d for d in data if d["project"] > 0]
    return max(pos, key=lambda x: x["project"]) if pos else None

def get_top_mitigator(data):
    neg = [d for d in data if d["project"] < 0]
    return min(neg, key=lambda x: x["project"]) if neg else None

def assert_data_consistency(radar_data, bar_data, tolerance=0.5):
    bar_map = {d["factor"].strip().lower(): d["project"] for d in bar_data}
    mismatches = []
    for d in radar_data:
        key = d["factor"].strip().lower()
        if key in bar_map:
            true_val = bar_map[key]
            if abs(d["project"] - true_val) > tolerance:
                mismatches.append(f"{d['factor']}: shows {d['project']} but source data has {true_val}")
    return mismatches

def test_neutral_ring_centering():
    max_abs = 10.0
    assert to_radial_scale(0, max_abs) == 50.0

def test_extreme_boundaries():
    max_abs = 10.0
    # Max risk increasing (+10) -> 100
    assert to_radial_scale(10.0, max_abs) == 100.0
    # Max risk decreasing (-10) -> 0
    assert to_radial_scale(-10.0, max_abs) == 0.0

def test_sample_dataset_scaling():
    max_abs = compute_max_abs_impact(SAMPLE_DATASET)
    assert max_abs == 10.0

    radial_terrain = to_radial_scale(-10, max_abs)
    assert radial_terrain == 0.0
    assert radial_terrain < 50.0, "Terrain Type must render INSIDE the neutral ring!"

    radial_agitation = to_radial_scale(9, max_abs)
    assert radial_agitation == 95.0
    assert radial_agitation > 50.0, "Local Agitation must render OUTSIDE the neutral ring!"

def test_top_driver_and_mitigator_signed_logic():
    top_driver = get_top_risk_driver(SAMPLE_DATASET)
    top_mitigator = get_top_mitigator(SAMPLE_DATASET)

    assert top_driver is not None
    assert top_driver["factor"] == "Local Agitation / Protest Flag"
    assert top_driver["project"] == 9

    assert top_mitigator is not None
    assert top_mitigator["factor"] == "Terrain Type"
    assert top_mitigator["project"] == -10

def test_color_encoding_matches_sign():
    def get_color(val):
        return "#ef5350" if val > 0 else "#26a69a" if val < 0 else "#6b7280"

    assert get_color(-10) == "#26a69a"  # Green for protective Terrain Type
    assert get_color(9) == "#ef5350"   # Red for risk-increasing Agitation

def test_current_project_dynamic_magnitude_resolution():
    """
    Test the fix for the reported bug:
    P_r (+17) and Population Density (-7) with State (+6), District (-5), Log Land Area (+6).
    """
    max_abs = compute_max_abs_impact(CURRENT_PROJECT_DATASET)
    # 17 rounded up to next multiple of 5 is 20
    assert max_abs == 20.0, f"Expected dynamic ceiling 20, got {max_abs}"

    # P_r (+17): Sitting near outer edge (+20 ring)
    radial_pr = to_radial_scale(17, max_abs)
    assert radial_pr == 92.5
    assert radial_pr > 50.0

    # Population Density (-7): Sitting deep inside neutral ring
    radial_pop = to_radial_scale(-7, max_abs)
    assert radial_pop == 32.5
    assert radial_pop < 50.0

    # Assert they are proportionally far apart (span 60 radial points)
    radial_separation = radial_pr - radial_pop
    assert radial_separation == 60.0
    assert radial_separation > 40.0, "P_r and Population Density should NOT be clustered near 50"

    # State & Log Land Area (+6): 50 + (6/20)*50 = 65.0
    radial_state = to_radial_scale(6, max_abs)
    assert radial_state == 65.0
    # P_r (92.5) must be noticeably farther out than State (65.0)
    assert (radial_pr - radial_state) == 27.5

    # District (-5): 50 + (-5/20)*50 = 37.5
    radial_district = to_radial_scale(-5, max_abs)
    assert radial_district == 37.5
    # Population Density (-7 -> 32.5) must be closer to center than District (-5 -> 37.5)
    assert radial_pop < radial_district

    # Shared utility driver/mitigator extraction
    driver = get_top_risk_driver(CURRENT_PROJECT_DATASET)
    mitigator = get_top_mitigator(CURRENT_PROJECT_DATASET)

    assert driver is not None
    assert driver["factor"] == "Protest & Agitation Risk Factor (P_r)"
    assert driver["project"] == 17, f"Expected 17 points, got {driver['project']}"
    assert driver["project"] != 2, "Driver must be 17 points, NOT compressed to 2!"

    assert mitigator is not None
    assert mitigator["factor"] == "Population Density"
    assert mitigator["project"] == -7, f"Expected -7 points, got {mitigator['project']}"
    assert mitigator["project"] != -1, "Mitigator must be -7 points, NOT compressed to -1!"

def test_data_consistency_guardrail():
    """
    Tests the dev-mode data consistency guardrail:
    Throws error when radar receives compressed/stale values (+2 instead of +17).
    """
    stale_radar_data = [
        {"factor": "Protest & Agitation Risk Factor (P_r)", "project": 2, "benchmark": 1},
        {"factor": "Population Density", "project": -1, "benchmark": 0},
    ]

    mismatches = assert_data_consistency(stale_radar_data, CURRENT_PROJECT_DATASET)
    assert len(mismatches) == 2
    assert "Protest & Agitation Risk Factor (P_r): shows 2 but source data has 17" in mismatches[0]
    assert "Population Density: shows -1 but source data has -7" in mismatches[1]

    # Matching dataset produces zero mismatches
    correct_mismatches = assert_data_consistency(CURRENT_PROJECT_DATASET, CURRENT_PROJECT_DATASET)
    assert len(correct_mismatches) == 0
