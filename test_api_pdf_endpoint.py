import requests
import json

BASE_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "super-secret-token"}

def test_extract_endpoint():
    print("=== TEST 1: POST /extract-project-pdf WITH DEMO PDF ===")
    with open("templates/Land_Acquisition_Project_Data_Sheet_DEMO.pdf", "rb") as f:
        files = {"file": ("Land_Acquisition_Project_Data_Sheet_DEMO.pdf", f, "application/pdf")}
        res = requests.post(f"{BASE_URL}/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    print(f"Extracted {data['filled_count']}/{data['total_fields']} fields.")
    assert data['fields']['inp-project-id'] == 'NHAI-RJ-2023-0001'
    assert data['fields']['inp-state'] == 'Rajasthan'
    assert data['fields']['inp-district'] == 'Banswara'
    assert data['fields']['inp-cost'] == 1485.37
    assert data['fields']['inp-protest-flag'] is False
    print(">>> Endpoint Test 1 PASSED!\n")

    print("=== TEST 2: AUTH REJECTION (NO API KEY) ===")
    with open("templates/Land_Acquisition_Project_Data_Sheet_DEMO.pdf", "rb") as f:
        files = {"file": ("demo.pdf", f, "application/pdf")}
        res_no_auth = requests.post(f"{BASE_URL}/extract-project-pdf", files=files)
    print(f"Status: {res_no_auth.status_code}")
    assert res_no_auth.status_code in [401, 403], f"Expected 401/403, got {res_no_auth.status_code}"
    print(">>> Endpoint Test 2 PASSED: Correctly blocked unauthorized request!\n")

    print("=== TEST 3: BLANK TEMPLATE (EXPECT 400) ===")
    with open("templates/Form_LA-7_Blank_Template.pdf", "rb") as f:
        files = {"file": ("blank.pdf", f, "application/pdf")}
        res_blank = requests.post(f"{BASE_URL}/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res_blank.status_code}, detail: {res_blank.json().get('detail')}")
    assert res_blank.status_code == 400
    print(">>> Endpoint Test 3 PASSED: Correctly returned 400 for blank template!\n")

    print("=== TEST 4: UNRELATED PDF (EXPECT 400) ===")
    with open("templates/unrelated_document.pdf", "rb") as f:
        files = {"file": ("unrelated.pdf", f, "application/pdf")}
        res_unrelated = requests.post(f"{BASE_URL}/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res_unrelated.status_code}, detail: {res_unrelated.json().get('detail')}")
    assert res_unrelated.status_code == 400
    print(">>> Endpoint Test 4 PASSED: Correctly returned 400 for unrelated document!\n")

    print("=== TEST 5: GET /download-blank-template ===")
    res_dl = requests.get(f"{BASE_URL}/download-blank-template")
    print(f"Status: {res_dl.status_code}, Content-Type: {res_dl.headers.get('Content-Type')}, Size: {len(res_dl.content)} bytes")
    assert res_dl.status_code == 200
    assert len(res_dl.content) > 1000
    print(">>> Endpoint Test 5 PASSED: Blank template downloaded successfully!\n")

if __name__ == "__main__":
    test_extract_endpoint()
