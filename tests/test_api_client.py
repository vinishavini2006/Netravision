"""
NetraVision API Endpoint Verification Test
Tests FastAPI endpoints directly via Starlette TestClient:
- GET /api/system-status
- GET /api/samples
- POST /api/patients
- POST /api/screen (with sample fundus image)
- GET /api/report/{screening_id}
"""

import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_api_endpoints():
    print("=== Testing NetraVision API Endpoints ===")
    
    # 1. System status
    res = client.get("/api/system-status")
    assert res.status_code == 200, f"Status failed: {res.text}"
    status = res.json()
    assert status["status"] == "online"
    print(f"[PASS] /api/system-status: {status['system']} ({status['device']})")

    # 2. Samples list
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()["samples"]
    assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
    print(f"[PASS] /api/samples: Loaded {len(samples)} clinical test samples.")

    # 3. Patient registration
    res = client.post("/api/patients", json={
        "patient_id": "TEST-PT-2026",
        "full_name": "Ramesh Kumar",
        "age": 54,
        "gender": "Male",
        "contact_number": "+91 98765 43210",
        "village_location": "Dharmapuri PHC",
        "diabetes_duration_years": 8.0,
        "known_hba1c": 8.4,
        "clinical_notes": "Referred from rural health sub-center"
    })
    assert res.status_code == 200, f"Patient reg failed: {res.text}"
    patient = res.json()["patient"]
    assert patient["full_name"] == "Ramesh Kumar"
    print(f"[PASS] /api/patients: Registered patient {patient['patient_id']}")

    # 4. Diagnostic screening endpoint
    sample_img_path = os.path.join(os.path.dirname(__file__), "..", "backend", "samples", "grade2_moderate.jpg")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/screen",
        data={
            "patient_id": "TEST-PT-2026",
            "eye": "OD",
            "colormap": "jet",
            "alpha": 0.55
        },
        files={"image": ("grade2_moderate.jpg", img_bytes, "image/jpeg")}
    )
    assert res.status_code == 200, f"Screening failed: {res.text}"
    result = res.json()
    assert result["success"] is True
    assert result["predicted_grade"] == 2
    assert "data:image/jpeg;base64" in result["images"]["gradcam_overlay"]
    assert "data:image/jpeg;base64" in result["images"]["original"]
    screening_id = result["screening_id"]
    print(f"[PASS] /api/screen: Result Grade {result['predicted_grade']} ({result['grade_label']}), Confidence: {result['confidence_percentage']}%, Screening ID: {screening_id}")

    # 5. Printable Report
    res = client.get(f"/api/report/{screening_id}")
    assert res.status_code == 200
    assert "<!DOCTYPE html>" in res.text
    assert "Ramesh Kumar" in res.text
    print(f"[PASS] /api/report/{screening_id}: Generated clinical printable report.")

    print("\n>>> ALL API ENDPOINT CHECKS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_api_endpoints()
