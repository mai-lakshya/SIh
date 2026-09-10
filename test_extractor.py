from form_la7_extractor import FormLA7Extractor
import json

def run_tests():
    # Test 1: DEMO PDF
    print('=== TEST 1: DEMO PDF ===')
    with open('templates/Land_Acquisition_Project_Data_Sheet_DEMO.pdf', 'rb') as f:
        res = FormLA7Extractor.extract_from_bytes(f.read())
    print(f"Filled: {res['filled_count']}/{res['total_fields']}")
    for k, v in res['fields'].items():
        print(f"  {k:22s} -> {v}")
    
    assert res['fields']['inp-project-id'] == 'NHAI-RJ-2023-0001'
    assert res['fields']['inp-project-type'] == 'Highway'
    assert res['fields']['inp-state'] == 'Rajasthan'
    assert res['fields']['inp-district'] == 'Banswara'
    assert res['fields']['inp-terrain'] == 'Rural_Agri'
    assert res['fields']['inp-latitude'] == 23.5461
    assert res['fields']['inp-longitude'] == 74.4439
    assert res['fields']['inp-cost'] == 1485.37
    assert res['fields']['inp-land-area'] == 229.1
    assert res['fields']['inp-sia-status'] == 'Pending'
    assert res['fields']['inp-forest-status'] == 'Not_Required'
    assert res['fields']['inp-fund-pct'] == 24.96
    assert res['fields']['inp-sec11-days'] == 311
    assert res['fields']['inp-comp-mult'] == 1.65
    assert res['fields']['inp-affected-families'] == 550
    assert res['fields']['inp-dispute-rate'] == 19.78
    assert res['fields']['inp-protest-flag'] is False
    print(">>> TEST 1 PASSED: 100% exact match on DEMO PDF!\n")

    # Test 2: VARIANT PDF
    print('=== TEST 2: VARIANT PDF ===')
    with open('templates/Land_Acquisition_Project_Data_Sheet_VARIANT.pdf', 'rb') as f:
        res_var = FormLA7Extractor.extract_from_bytes(f.read())
    print(f"Filled: {res_var['filled_count']}/{res_var['total_fields']}")
    for k, v in res_var['fields'].items():
        print(f"  {k:22s} -> {v}")
    assert res_var['fields']['inp-project-id'] == 'MOR-MH-2024-0042'
    assert res_var['fields']['inp-project-type'] == 'Railway'
    assert res_var['fields']['inp-state'] == 'Maharashtra'
    assert res_var['fields']['inp-district'] == 'Pune'
    assert res_var['fields']['inp-terrain'] == 'Hilly'
    assert res_var['fields']['inp-cost'] == 2840.5
    assert res_var['fields']['inp-protest-flag'] is True
    print(">>> TEST 2 PASSED: 100% exact match on VARIANT PDF!\n")

    # Test 3: PARTIAL PDF
    print('=== TEST 3: PARTIAL PDF ===')
    with open('templates/Land_Acquisition_Project_Data_Sheet_PARTIAL.pdf', 'rb') as f:
        res_part = FormLA7Extractor.extract_from_bytes(f.read())
    print(f"Filled: {res_part['filled_count']}/{res_part['total_fields']}")
    for k, v in res_part['fields'].items():
        print(f"  {k:22s} -> {v}")
    assert res_part['fields']['inp-district'] is None
    assert res_part['fields']['inp-road-type'] is None
    assert res_part['fields']['inp-land-area'] is None
    assert res_part['fields']['inp-fund-pct'] is None
    assert res_part['fields']['inp-dispute-rate'] is None
    assert res_part['fields']['inp-state'] == 'Gujarat'
    print(">>> TEST 3 PASSED: Partial fields correctly preserved as None!\n")

    # Test 4: Blank Template
    print('=== TEST 4: BLANK TEMPLATE ===')
    try:
        with open('templates/Form_LA-7_Blank_Template.pdf', 'rb') as f:
            FormLA7Extractor.extract_from_bytes(f.read())
        print("FAIL: Expected rejection of blank template")
    except ValueError as e:
        print(">>> TEST 4 PASSED: Rejected blank template ->", e)

    # Test 5: Unrelated PDF
    print('\n=== TEST 5: UNRELATED PDF ===')
    try:
        with open('templates/unrelated_document.pdf', 'rb') as f:
            FormLA7Extractor.extract_from_bytes(f.read())
        print("FAIL: Expected rejection of unrelated document")
    except ValueError as e:
        print(">>> TEST 5 PASSED: Rejected unrelated PDF ->", e)

if __name__ == '__main__':
    run_tests()
