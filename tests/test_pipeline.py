"""
NetraVision Automated Verification Test Suite
Verifies:
1. Retinal image preprocessor and FOV circle cropping
2. Model architecture & forward inference
3. Grad-CAM hook execution, Class Activation Mapping, and heatmap normalization
4. Lesion contour extraction and quadrant localization
5. Database CRUD operations for patient registration and screening
6. Clinical report HTML builder
"""

import sys
import os
import numpy as np
import cv2

# Ensure netravision root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessor import preprocess_fundus, crop_fundus_circle, assess_image_quality
from backend.model import load_model, DR_CLASSES, TORCH_AVAILABLE
from backend.gradcam import GradCAM, create_heatmap_overlay, extract_lesion_regions
from backend.database import init_db, register_patient, get_patient_by_id, save_screening, get_screening_by_id
from backend.reports import build_clinical_report_html
from backend.generate_samples import generate_base_fundus, generate_all_samples


def test_samples_generation():
    print("[1/6] Testing sample fundus generation...")
    generate_all_samples()
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "samples")
    assert os.path.exists(os.path.join(samples_dir, "grade0_normal.jpg")), "Grade 0 sample missing"
    assert os.path.exists(os.path.join(samples_dir, "grade2_moderate.jpg")), "Grade 2 sample missing"
    assert os.path.exists(os.path.join(samples_dir, "grade4_proliferative.jpg")), "Grade 4 sample missing"
    print("  -> Sample fundus scans successfully verified.")


def test_preprocessor():
    print("[2/6] Testing retinal preprocessor and image QA...")
    dummy_img = generate_base_fundus(size=256, seed=42)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    img_bytes = encoded.tobytes()

    tensor, display_rgb, enhanced_rgb, qa = preprocess_fundus(img_bytes, target_size=(224, 224))
    
    assert display_rgb.shape == (224, 224, 3), f"Unexpected display RGB shape: {display_rgb.shape}"
    assert enhanced_rgb.shape == (224, 224, 3), f"Unexpected enhanced RGB shape: {enhanced_rgb.shape}"
    assert "sharpness_score" in qa, "Missing sharpness QA metric"
    assert "quality_grade" in qa, "Missing quality grade QA metric"
    print(f"  -> Preprocessor verified. QA Grade: {qa['quality_grade']}, Sharpness: {qa['sharpness_score']}")


def test_model_and_gradcam():
    print("[3/6] Testing CNN model and Grad-CAM explainability...")
    model, target_layer, device = load_model()
    gradcam = GradCAM(model, target_layer)

    dummy_img = generate_base_fundus(size=224, seed=42)
    heatmap, pred_class, conf = gradcam.generate_cam(dummy_img)

    assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy ndarray"
    assert heatmap.ndim == 2, f"Heatmap must be 2D, got {heatmap.ndim}D"
    assert 0 <= pred_class < 5, f"Predicted class out of bounds: {pred_class}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"
    assert np.min(heatmap) >= 0.0 and np.max(heatmap) <= 1.0, "Heatmap not normalized to [0, 1]"
    
    gradcam.remove_hooks()
    print(f"  -> Model & Grad-CAM verified. Pred class: {pred_class}, Conf: {conf:.3f}, Heatmap shape: {heatmap.shape}")


def test_lesion_localization():
    print("[4/6] Testing lesion contour detection and quadrant extraction...")
    dummy_heatmap = np.zeros((224, 224), dtype=np.float32)
    # Inject synthetic lesion activation in Superotemporal quadrant
    cv2.circle(dummy_heatmap, (160, 60), 20, 0.95, -1)
    
    regions = extract_lesion_regions(dummy_heatmap, (224, 224), threshold=0.5)
    assert len(regions) > 0, "Failed to extract lesion region"
    assert "Superior" in regions[0]["quadrant"], f"Quadrant unexpected: {regions[0]['quadrant']}"
    assert regions[0]["attention_intensity"] > 0.8, "Attention intensity should be > 0.8"
    print(f"  -> Lesion localization verified. Detected ROI in: {regions[0]['quadrant']}")


def test_database():
    print("[5/6] Testing SQLite database operations...")
    init_db()
    test_patient = {
        "patient_id": "TEST-PATIENT-001",
        "full_name": "Sita Devi",
        "age": 58,
        "gender": "Female",
        "village_location": "Rampur PHC",
        "diabetes_duration_years": 10.5,
        "known_hba1c": 8.9
    }
    registered = register_patient(test_patient)
    assert registered["patient_id"] == "TEST-PATIENT-001"
    
    fetched = get_patient_by_id("TEST-PATIENT-001")
    assert fetched["full_name"] == "Sita Devi"
    
    screening_record = save_screening({
        "patient_id": "TEST-PATIENT-001",
        "eye": "OD",
        "predicted_grade": 2,
        "grade_label": "Moderate Non-Proliferative DR",
        "confidence_score": 0.89,
        "probabilities": {0: 0.02, 1: 0.06, 2: 0.89, 3: 0.02, 4: 0.01},
        "original_image_path": "data:image/jpeg;base64,...",
        "gradcam_image_path": "data:image/jpeg;base64,...",
        "lesion_count": 2,
        "lesions": [{"region_id": 1, "quadrant": "Superotemporal", "attention_intensity": 0.88}],
        "urgency": "Moderate",
        "recommendation": "Referral within 3-6 months"
    })
    
    assert screening_record is not None
    assert screening_record["predicted_grade"] == 2
    print("  -> Database CRUD operations verified successfully.")


def test_clinical_report():
    print("[6/6] Testing clinical report generator...")
    screening = {
        "screening_id": "SCR-TEST-9999",
        "patient_id": "TEST-PATIENT-001",
        "eye": "OD",
        "predicted_grade": 2,
        "grade_label": "Moderate Non-Proliferative DR",
        "confidence_score": 0.89,
        "lesions": [{"region_id": 1, "quadrant": "Superotemporal", "attention_intensity": 0.91, "clinical_significance": "High Attention"}],
        "created_at": "2026-09-20 20:30:00"
    }
    patient = {
        "full_name": "Sita Devi",
        "patient_id": "TEST-PATIENT-001",
        "age": 58,
        "gender": "Female",
        "village_location": "Rampur PHC",
        "diabetes_duration_years": 10.5,
        "known_hba1c": 8.9
    }
    html = build_clinical_report_html(screening, patient)
    assert "<!DOCTYPE html>" in html
    assert "Sita Devi" in html
    assert "SCR-TEST-9999" in html
    assert "Moderate Non-Proliferative" in html
    print("  -> Clinical report HTML generation verified.")


if __name__ == "__main__":
    print("========================================")
    print(" NetraVision Full Pipeline Verification ")
    print("========================================")
    test_samples_generation()
    test_preprocessor()
    test_model_and_gradcam()
    test_lesion_localization()
    test_database()
    test_clinical_report()
    print("========================================")
    print(" ALL 6 VERIFICATION CHECKS PASSED!     ")
    print("========================================")
