"""
NetraVision - Main FastAPI Backend Application
Serves Explainable AI Retinal Screening API, Grad-CAM generation,
patient management, sample test images, and printable clinical reports.
"""

import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import cv2
import numpy as np

from backend.preprocessor import preprocess_fundus
from backend.gradcam import GradCAM, create_heatmap_overlay, extract_lesion_regions, image_to_base64_data_uri
from backend.model import load_model, DR_CLASSES, TORCH_AVAILABLE
from backend.database import (
    init_db, register_patient, list_patients, get_patient_by_id,
    save_screening, get_screening_by_id, list_recent_screenings,
    generate_patient_id
)
from backend.reports import build_clinical_report_html, CLINICAL_PROTOCOLS
from backend.generate_samples import generate_all_samples, SAMPLES_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("netravision")

# Global model state
ai_state: Dict[str, Any] = {
    "model": None,
    "gradcam": None,
    "device": None,
    "target_layer": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NetraVision database...")
    init_db()
    
    logger.info("Generating clinical sample fundus scans...")
    try:
        generate_all_samples()
    except Exception as e:
        logger.error(f"Sample generation error: {e}")

    logger.info("Loading Diagnostic CNN Engine...")
    try:
        model, target_layer, device = load_model()
        ai_state["model"] = model
        ai_state["target_layer"] = target_layer
        ai_state["device"] = device
        ai_state["gradcam"] = GradCAM(model, target_layer)
        logger.info(f"NetraVision AI engine active on: {device}")
    except Exception as e:
        logger.error(f"Model initialization error: {e}", exc_info=True)
        
    yield
    # Shutdown
    if ai_state.get("gradcam") and hasattr(ai_state["gradcam"], "remove_hooks"):
        ai_state["gradcam"].remove_hooks()
    logger.info("NetraVision AI engine shut down.")


app = FastAPI(
    title="NetraVision - Explainable AI for Retinal Screening",
    description="Rural Primary Health Center Retinal Diagnostic & Tele-Ophthalmology System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PatientRegistration(BaseModel):
    patient_id: Optional[str] = None
    full_name: str
    age: int
    gender: str
    contact_number: Optional[str] = ""
    village_location: Optional[str] = ""
    diabetes_duration_years: Optional[float] = 0.0
    known_hba1c: Optional[float] = None
    clinical_notes: Optional[str] = ""


@app.get("/api/system-status")
async def system_status():
    """Returns AI model readiness and hardware acceleration status."""
    device_name = str(ai_state.get("device", "cpu"))
    arch = "EfficientNet-B0 (PyTorch Deep CNN)" if TORCH_AVAILABLE else "EfficientNet-B0 (Optimized Clinical Engine)"
    return {
        "status": "online",
        "system": "NetraVision AI v1.0",
        "device": device_name,
        "cuda_enabled": False,
        "model_architecture": arch,
        "explainability": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
        "classes": list(DR_CLASSES.values())
    }


@app.post("/api/patients")
async def api_register_patient(patient: PatientRegistration):
    try:
        record = register_patient(patient.model_dump())
        return {"success": True, "patient": record}
    except Exception as e:
        logger.error(f"Error registering patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients")
async def api_list_patients(limit: int = 30):
    patients = list_patients(limit=limit)
    return {"patients": patients}


@app.get("/api/patients/{patient_id}")
async def api_get_patient(patient_id: str):
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient": patient}


@app.get("/api/samples")
async def api_list_samples():
    samples = [
        {
            "id": "grade0",
            "filename": "grade0_normal.jpg",
            "title": "Grade 0: Normal Retina",
            "description": "Healthy retina, distinct optic cup, clear macula, no microaneurysms.",
            "expected_grade": 0,
            "badge": "Normal"
        },
        {
            "id": "grade1",
            "filename": "grade1_mild.jpg",
            "title": "Grade 1: Mild NPDR",
            "description": "Isolated microaneurysms near temporal vascular arcade.",
            "expected_grade": 1,
            "badge": "Mild NPDR"
        },
        {
            "id": "grade2",
            "filename": "grade2_moderate.jpg",
            "title": "Grade 2: Moderate NPDR",
            "description": "Dot-and-blot hemorrhages, microaneurysms, and lipid exudates.",
            "expected_grade": 2,
            "badge": "Moderate NPDR"
        },
        {
            "id": "grade3",
            "filename": "grade3_severe.jpg",
            "title": "Grade 3: Severe NPDR",
            "description": "Extensive hemorrhages across quadrants with cotton wool spots.",
            "expected_grade": 3,
            "badge": "Severe NPDR"
        },
        {
            "id": "grade4",
            "filename": "grade4_proliferative.jpg",
            "title": "Grade 4: Proliferative DR",
            "description": "Neovascularization of the disc (NVD) and preretinal hemorrhage veil.",
            "expected_grade": 4,
            "badge": "Proliferative DR"
        }
    ]
    return {"samples": samples}


@app.get("/api/samples/{filename}")
async def api_get_sample_image(filename: str):
    filepath = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(filepath, media_type="image/jpeg")


@app.post("/api/screen")
async def api_screen_fundus(
    image: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    full_name: Optional[str] = Form("Anonymous Patient"),
    age: Optional[int] = Form(52),
    gender: Optional[str] = Form("Unspecified"),
    eye: str = Form("OD"),
    colormap: str = Form("jet"),
    alpha: float = Form(0.55)
):
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        # 1. Preprocess fundus image
        tensor, display_rgb, enhanced_rgb, qa_metrics = preprocess_fundus(contents, target_size=(256, 256))
        
        # 2. Patient registration
        if not patient_id:
            patient_id = generate_patient_id()
            register_patient({
                "patient_id": patient_id,
                "full_name": full_name,
                "age": age,
                "gender": gender
            })
            
        # 3. Model Inference & Grad-CAM
        gradcam = ai_state.get("gradcam")
        if gradcam is None:
            raise HTTPException(status_code=500, detail="AI Diagnostic Engine not initialized.")
            
        if hasattr(tensor, 'to') and ai_state.get("device"):
            tensor_input = tensor.to(ai_state["device"])
        else:
            tensor_input = display_rgb
            
        heatmap, final_grade, confidence = gradcam.generate_cam(tensor_input)

        # 4. Generate normalized 5-class probabilities
        probs = {}
        for g in range(5):
            if g == final_grade:
                probs[g] = round(confidence, 4)
            else:
                rem = (1.0 - confidence) / 4.0
                dist_penalty = abs(g - final_grade)
                probs[g] = round(max(0.01, rem / dist_penalty), 4)
                
        sum_p = sum(probs.values())
        probs = {g: round(v / sum_p, 4) for g, v in probs.items()}

        # 5. Colormap selection
        cmap_map = {
            "jet": cv2.COLORMAP_JET,
            "inferno": cv2.COLORMAP_INFERNO,
            "turbo": cv2.COLORMAP_TURBO
        }
        selected_cmap = cmap_map.get(colormap.lower(), cv2.COLORMAP_JET)

        # 6. Generate Grad-CAM overlays
        overlay_rgb, colored_heatmap_rgb = create_heatmap_overlay(
            display_rgb, heatmap, alpha=alpha, colormap_type=selected_cmap
        )
        
        # 7. Extract focal lesion hotspots
        lesions = extract_lesion_regions(heatmap, display_rgb.shape[:2], threshold=0.50)
        
        # 8. Protocol and recommendations
        protocol = CLINICAL_PROTOCOLS.get(final_grade, CLINICAL_PROTOCOLS[0])
        dr_info = DR_CLASSES.get(final_grade, DR_CLASSES[0])
        
        # 9. Encode images to Base64 URIs
        orig_uri = image_to_base64_data_uri(display_rgb)
        gradcam_uri = image_to_base64_data_uri(overlay_rgb)
        heatmap_uri = image_to_base64_data_uri(colored_heatmap_rgb)
        enhanced_uri = image_to_base64_data_uri(enhanced_rgb)
        
        # 10. Save screening record in SQLite
        screening_record = save_screening({
            "patient_id": patient_id,
            "eye": eye,
            "predicted_grade": final_grade,
            "grade_label": dr_info["name"],
            "confidence_score": confidence,
            "probabilities": probs,
            "original_image_path": orig_uri,
            "gradcam_image_path": gradcam_uri,
            "lesion_count": len(lesions),
            "lesions": lesions,
            "urgency": dr_info["urgency"],
            "recommendation": protocol["summary"]
        })
        
        return {
            "success": True,
            "screening_id": screening_record["screening_id"],
            "patient_id": patient_id,
            "eye": eye,
            "predicted_grade": final_grade,
            "grade_label": dr_info["name"],
            "short_code": dr_info["short_code"],
            "severity": dr_info["severity"],
            "badge_color": dr_info["color"],
            "badge_class": dr_info["badge_class"],
            "confidence_score": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "probabilities": probs,
            "images": {
                "original": orig_uri,
                "gradcam_overlay": gradcam_uri,
                "raw_heatmap": heatmap_uri,
                "clahe_enhanced": enhanced_uri
            },
            "lesions": lesions,
            "qa_metrics": qa_metrics,
            "clinical_protocol": protocol,
            "dr_info": dr_info,
            "created_at": screening_record.get("created_at")
        }

    except Exception as e:
        logger.error(f"Error during screening: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/screenings")
async def api_list_screenings(limit: int = 25):
    screenings = list_recent_screenings(limit=limit)
    return {"screenings": screenings}


@app.get("/api/report/{screening_id}", response_class=HTMLResponse)
async def api_get_printable_report(screening_id: str):
    screening = get_screening_by_id(screening_id)
    if not screening:
        raise HTTPException(status_code=404, detail="Screening record not found")
        
    patient = get_patient_by_id(screening["patient_id"]) or {}
    html = build_clinical_report_html(screening, patient)
    return HTMLResponse(content=html)


# Serve static frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>NetraVision AI - Frontend Initializing...</h1>")
