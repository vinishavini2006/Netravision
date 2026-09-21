"""
NetraVision - Retinal Fundus Preprocessor
Implements circular FOV auto-cropping, CLAHE contrast enhancement,
image quality QA scoring, and tensor normalization.
"""

import io
from typing import Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    torch = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def assess_image_quality(img_rgb: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_score = float(laplacian.var())
    
    mask = gray > 15
    if np.sum(mask) > 0:
        mean_brightness = float(np.mean(gray[mask]))
    else:
        mean_brightness = float(np.mean(gray))
        
    is_sharp = sharpness_score >= 60.0
    is_well_lit = 35.0 <= mean_brightness <= 220.0
    quality_grade = "Adequate" if (is_sharp and is_well_lit) else "Suboptimal"
    
    warning_notes = []
    if not is_sharp:
        warning_notes.append("Potential motion blur or defocus detected.")
    if mean_brightness < 35.0:
        warning_notes.append("Underexposed / low illumination fundus image.")
    elif mean_brightness > 220.0:
        warning_notes.append("Overexposed / excessive flash reflection.")
        
    return {
        "sharpness_score": round(sharpness_score, 1),
        "mean_brightness": round(mean_brightness, 1),
        "quality_grade": quality_grade,
        "is_diagnostic_quality": is_sharp and is_well_lit,
        "warnings": warning_notes
    }


def crop_fundus_circle(img_bgr: np.ndarray, tolerance: int = 15) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    max_dim = max(w, h)
    center_x = x + w // 2
    center_y = y + h // 2
    
    x1 = max(0, center_x - max_dim // 2)
    y1 = max(0, center_y - max_dim // 2)
    x2 = min(img_bgr.shape[1], center_x + max_dim // 2)
    y2 = min(img_bgr.shape[0], center_y + max_dim // 2)
    
    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return img_bgr
    return cropped


def enhance_contrast_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def preprocess_fundus(
    image_input: Any,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    if isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif hasattr(image_input, "read"):
        image = Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    cropped_bgr = crop_fundus_circle(img_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    qa_metrics = assess_image_quality(cropped_rgb)
    display_rgb = cv2.resize(cropped_rgb, target_size, interpolation=cv2.INTER_AREA)
    enhanced_rgb = enhance_contrast_clahe(display_rgb)
    
    if TORCH_AVAILABLE and torch is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(display_rgb).unsqueeze(0)
    else:
        # Standard numpy normalized tensor [1, 3, H, W]
        norm = (display_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
    
    return tensor, display_rgb, enhanced_rgb, qa_metrics
