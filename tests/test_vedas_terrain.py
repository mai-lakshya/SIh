"""
Test suite for ISRO VEDAS satellite terrain detection integration.
Tests coordinate & district classification across India, API endpoints,
and integration with the Remoteness evaluation module.
"""

import sys
import os
import requests

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from remoteness.vedas_client import VedasTerrainDetector, VALID_TERRAIN_ENUMS


def test_vedas_detector_unit():
    print("=== TEST 1: VedasTerrainDetector Unit Tests ===")
    test_cases = [
        # (lat, lon, state, district, expected_type)
        (25.8103, 93.4302, "Assam", "Karbi Anglong", "Hilly"),
        (32.2190, 76.3234, "Himachal Pradesh", "Kangra", "Hilly"),
        (34.0837, 74.7973, "Jammu and Kashmir", "Srinagar", "Hilly"),
        (23.5461, 74.4439, "Rajasthan", "Banswara", "Tribal_Schedule_V"),
        (19.0760, 72.8777, "Maharashtra", "Mumbai", "Urban"),
        (28.6139, 77.2090, "Delhi", "Central Delhi", "Urban"),
        (29.5300, 78.7747, "Uttarakhand", "Nainital", "Forest_Eco_Sensitive"),
        (28.8386, 78.7733, "Uttar Pradesh", "Moradabad", "Rural_Agri"),
    ]

    for lat, lon, state, district, expected in test_cases:
        res = VedasTerrainDetector.detect_terrain(lat, lon, state=state, district=district)
        actual = res["terrain_type"]
        print(f"  [{lat:.4f}, {lon:.4f}] {district}, {state} -> {actual} ({res['terrain_label']})")
        assert actual in VALID_TERRAIN_ENUMS, f"Invalid enum: {actual}"
        assert actual == expected, f"Expected {expected}, got {actual} for {district}"
        assert "telemetry" in res
        assert "elevation_m" in res["telemetry"]
        assert "slope_deg" in res["telemetry"]
        assert "source" in res
        assert "confidence" in res
    print(">>> VedasTerrainDetector Unit Tests PASSED!\n")


def test_gis_detect_terrain_api():
    print("=== TEST 2: GET & POST /gis/detect-terrain Endpoints ===")
    base_url = "http://localhost:8000"

    # GET request
    r_get = requests.get(f"{base_url}/gis/detect-terrain", params={
        "latitude": 25.8103,
        "longitude": 93.4302,
        "state": "Assam",
        "district": "Karbi Anglong"
    })
    print(f"  GET status: {r_get.status_code}")
    assert r_get.status_code == 200, f"GET failed with {r_get.status_code}: {r_get.text}"
    data_get = r_get.json()
    assert data_get["terrain_type"] == "Hilly"
    assert "telemetry" in data_get
    print(f"  GET response: {data_get['terrain_type']} | {data_get['telemetry']['elevation_m']}m elev | {data_get['telemetry']['slope_deg']}° slope")

    # POST request with JSON body
    r_post = requests.post(f"{base_url}/gis/detect-terrain", json={
        "latitude": 23.5461,
        "longitude": 74.4439,
        "state": "Rajasthan",
        "district": "Banswara"
    })
    print(f"  POST status: {r_post.status_code}")
    assert r_post.status_code == 200, f"POST failed with {r_post.status_code}: {r_post.text}"
    data_post = r_post.json()
    assert data_post["terrain_type"] == "Tribal_Schedule_V"
    print(f"  POST response: {data_post['terrain_type']} | Source: {data_post['source']}")

    print(">>> /gis/detect-terrain API Tests PASSED!\n")


def test_remoteness_evaluation_with_vedas():
    print("=== TEST 3: /remoteness/evaluate with VEDAS Telemetry ===")
    base_url = "http://localhost:8000"
    payload = {
        "state": "Assam",
        "district": "Karbi Anglong",
        "latitude": 25.8103,
        "longitude": 93.4302,
        "project_type": "Highway"
    }
    r = requests.post(f"{base_url}/remoteness/evaluate", json=payload)
    print(f"  Status: {r.status_code}")
    assert r.status_code == 200, f"Remoteness failed with {r.status_code}: {r.text}"
    res = r.json()
    assert "vedas_telemetry" in res, "vedas_telemetry missing from remoteness result"
    assert res["vedas_telemetry"]["terrain_type"] == "Hilly"
    print(f"  Remoteness delay: {res['remoteness_delay_days']}d | VEDAS: {res['vedas_telemetry']['terrain_label']}")
    print(">>> /remoteness/evaluate VEDAS Integration PASSED!\n")


if __name__ == "__main__":
    test_vedas_detector_unit()
    test_gis_detect_terrain_api()
    test_remoteness_evaluation_with_vedas()
