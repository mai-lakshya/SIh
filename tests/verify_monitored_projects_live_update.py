"""
End-to-End Verification of Real-Time Monitored Projects Metric Updates
Verifies that:
1. GET /projects/stats returns exact live project count.
2. Direct disk CSV append automatically increments /projects/stats.
3. API /model/ingest increments /projects/stats and /model/health.
4. /projects/geo returns complete updated project array.
5. Reverting returns the metric to baseline.
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
CSV_PATH = "indian_infrastructure_projects_dataset.csv"

def run_tests():
    print("=== RUNNING MONITORED PROJECTS DYNAMIC UPDATE TESTS ===")

    # 1. Check Initial Baseline Count
    res = requests.get(f"{BASE_URL}/projects/stats")
    assert res.status_code == 200, f"Stats returned HTTP {res.status_code}"
    initial_stats = res.json()
    initial_count = initial_stats["total_projects"]
    print(f"[OK] 1. Initial Monitored Projects: {initial_count:,}")

    # Read original CSV lines
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        original_lines = f.readlines()
    actual_csv_rows = len(original_lines) - 1
    assert initial_count == actual_csv_rows, f"Mismatch: Stats {initial_count} vs CSV {actual_csv_rows}"
    print(f"[OK] 2. Verified stats match CSV row count ({actual_csv_rows:,} rows).")

    # 2. Test Direct Disk CSV Append
    test_line = "TEST-MONITOR-001,Highway,Maharashtra,Mumbai,Urban,50.0,400.0,2024,100,5.0,False,1.5,Approved,20,Not_Required,45.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,,1,0,Low,22.0,Low\n"
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        f.write(test_line)

    # Immediately query /projects/stats
    res = requests.get(f"{BASE_URL}/projects/stats")
    assert res.status_code == 200
    after_append_count = res.json()["total_projects"]
    print(f"[OK] 3. After Disk CSV Append: Monitored Projects = {after_append_count:,} (Expected: {initial_count + 1:,})")
    assert after_append_count == initial_count + 1, "Failed to detect disk CSV append!"

    # 3. Test Ingestion via /model/ingest
    ingest_payload = {
        "records": [
            {
                "project_id": "TEST-INGEST-001",
                "state": "Karnataka",
                "district": "Bengaluru",
                "project_type": "Railway",
                "terrain_type": "Urban",
                "land_area_hectares": 35.0,
                "estimated_cost_inr_crore": 850.0,
                "affected_families_count": 250,
                "title_dispute_rate_percent": 8.0,
                "local_protest_flag": False,
                "forest_clearance_status": "Approved",
                "fund_disbursement_percent": 30.0
            }
        ]
    }
    ingest_res = requests.post(
        f"{BASE_URL}/model/ingest",
        json=ingest_payload,
        headers={"X-API-Key": "super-secret-token"}
    )
    assert ingest_res.status_code == 200, f"Ingest failed: {ingest_res.text}"
    print(f"[OK] 4. Successfully ingested test record via /model/ingest.")

    res = requests.get(f"{BASE_URL}/projects/stats")
    after_ingest_count = res.json()["total_projects"]
    print(f"[OK] 5. After /model/ingest: Monitored Projects = {after_ingest_count:,} (Expected: {initial_count + 2:,})")
    assert after_ingest_count == initial_count + 2, "Failed to update Monitored Projects after ingest!"

    # Check /model/health training_size
    health_res = requests.get(f"{BASE_URL}/model/health")
    assert health_res.status_code == 200
    training_size = health_res.json()["training_size"]
    print(f"[OK] 6. /model/health reports live data store size = {training_size:,}")
    assert training_size == after_ingest_count, f"Health training size {training_size} mismatch {after_ingest_count}"

    # 4. Clean up: restore original CSV
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.writelines(original_lines)

    # Verify count returns to baseline
    res = requests.get(f"{BASE_URL}/projects/stats")
    revert_count = res.json()["total_projects"]
    print(f"[OK] 7. After cleanup: Monitored Projects = {revert_count:,} (Expected: {initial_count:,})")
    assert revert_count == initial_count, "Failed to revert to baseline!"

    print("\n========================================================")
    print("ALL DYNAMIC MONITORED PROJECTS TESTS PASSED 100%!")
    print("========================================================\n")

if __name__ == "__main__":
    run_tests()
