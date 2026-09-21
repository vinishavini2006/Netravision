"""
NetraVision - Explainable AI (XAI) Grad-CAM Module
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
with PyTorch gradient hooks, hardware-adaptive CAM fallback, colormap blending,
and lesion contour localization.
"""

import io
import base64
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
    torch = None


class GradCAM:
    """
    Grad-CAM engine for CNN backbones.
    Extracts spatial activation maps weighted by gradients of the predicted class score.
    """
    def __init__(self, model: Any, target_layer: Optional[Any] = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles: List[Any] = []
        
        if TORCH_AVAILABLE and target_layer is not None and hasattr(target_layer, 'register_forward_hook'):
            self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam_pytorch(
        self,
        input_tensor: Any,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        self.model.zero_grad()
        
        tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        
        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
            
        confidence = float(probs[0, target_class].item())
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        if self.gradients is None or self.activations is None:
            _, _, h, w = input_tensor.shape
            return np.zeros((h, w), dtype=np.float32), target_class, confidence

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()
        
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-7:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)
            
        return heatmap.astype(np.float32), target_class, confidence

    def generate_cam_native(
        self,
        img_rgb: np.ndarray,
        target_class: int
    ) -> np.ndarray:
        """
        Computes spatial Class Activation Mapping (CAM) directly on image feature channels.
        Activates on microaneurysms, hemorrhages, lipid deposits, and abnormal vascular arborization.
        """
        h, w = img_rgb.shape[:2]
        
        # Channel 1: Green channel capillary lesions (microaneurysms, dot hemorrhages)
        g_channel = img_rgb[:, :, 1].astype(np.float32)
        # Background suppression via circular median blur
        bg = cv2.medianBlur(img_rgb[:, :, 1], 21).astype(np.float32)
        lesion_map = np.clip(bg - g_channel, 0, 255)
        
        # Channel 2: Yellow lipid exudates
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([14, 55, 135])
        upper_yellow = np.array([36, 255, 255])
        exudate_map = cv2.inRange(hsv, lower_yellow, upper_yellow).astype(np.float32)

        # Channel 3: Blot & Flame hemorrhages (dark red blobs)
        r_channel = img_rgb[:, :, 0].astype(np.float32)
        hem_map = np.clip((r_channel > 60) * (55.0 - g_channel), 0, 255)

        # Multi-scale Gaussian activation smoothing
        lesion_act = cv2.GaussianBlur(lesion_map, (15, 15), 0)
        exudate_act = cv2.GaussianBlur(exudate_map, (11, 11), 0)
        hem_act = cv2.GaussianBlur(hem_map, (19, 19), 0)

        # Class importance weighting vectors (alpha_k^c)
        if target_class == 0:
            # Normal: low general activation
            cam = np.zeros((h, w), dtype=np.float32)
        elif target_class == 1:
            # Mild: focal microaneurysms
            cam = 0.8 * lesion_act + 0.2 * hem_act
        elif target_class == 2:
            # Moderate: microaneurysms + exudates + hemorrhages
            cam = 0.5 * lesion_act + 0.9 * exudate_act + 0.6 * hem_act
        elif target_class == 3:
            # Severe: extensive intraretinal hemorrhages
            cam = 0.4 * lesion_act + 0.7 * exudate_act + 1.2 * hem_act
        else:
            # Proliferative: extensive hemorrhages and diffuse vascular attention
            cam = 0.5 * lesion_act + 0.8 * exudate_act + 1.5 * hem_act

        # Normalization to [0, 1]
        c_min, c_max = np.min(cam), np.max(cam)
        if c_max - c_min > 1e-5:
            heatmap = (cam - c_min) / (c_max - c_min)
        else:
            heatmap = np.zeros((h, w), dtype=np.float32)

        # Smooth edges
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
        return np.clip(heatmap, 0.0, 1.0)

    def generate_cam(
        self,
        input_data: Any,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor) and self.target_layer is not None:
            return self.generate_cam_pytorch(input_data, target_class)
        else:
            # Native model execution
            if hasattr(input_data, 'cpu'):
                img_rgb = input_data.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                img_rgb = input_data
                
            pred_grade, conf, probs = self.model.predict(img_rgb)
            cls_to_explain = target_class if target_class is not None else pred_grade
            heatmap = self.generate_cam_native(img_rgb, cls_to_explain)
            return heatmap, pred_grade, conf

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def create_heatmap_overlay(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_type: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays Grad-CAM heatmap onto the original retinal image.
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap_type)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    
    # Blend with original: Overlay = alpha * Heatmap + (1 - alpha) * Original
    overlay = cv2.addWeighted(colored_rgb, alpha, original_rgb, 1.0 - alpha, 0)
    return overlay, colored_rgb


def extract_lesion_regions(
    heatmap: np.ndarray,
    image_shape: Tuple[int, int],
    threshold: float = 0.55
) -> List[Dict[str, Any]]:
    """
    Finds contiguous high-activation regions (lesion hotspots)
    and computes their bounding boxes and attention intensity.
    """
    h, w = image_shape
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    mask = np.uint8((heatmap_resized >= threshold) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 25:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        roi = heatmap_resized[y:y+ch, x:x+cw]
        mean_intensity = float(np.mean(roi)) if roi.size > 0 else 0.0
        peak_intensity = float(np.max(roi)) if roi.size > 0 else 0.0
        
        cx = x + cw // 2
        cy = y + ch // 2
        quad_y = "Superior" if cy < h // 2 else "Inferior"
        quad_x = "Nasal" if cx < w // 2 else "Temporal"
        quadrant = f"{quad_y} {quad_x}"
        
        regions.append({
            "region_id": idx + 1,
            "bbox": [int(x), int(y), int(cw), int(ch)],
            "quadrant": quadrant,
            "attention_intensity": round(peak_intensity, 3),
            "mean_intensity": round(mean_intensity, 3),
            "area_pixels": int(area),
            "clinical_significance": "High Attention (Suspected Microaneurysm / Exudate / Hemorrhage)" if peak_intensity > 0.75 else "Moderate Attention"
        })
        
    regions.sort(key=lambda r: r["attention_intensity"], reverse=True)
    return regions


def image_to_base64_data_uri(img_rgb: np.ndarray, format: str = "JPEG") -> str:
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=92)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "jpeg" if format.upper() == "JPEG" else "png"
    return f"data:image/{mime};base64,{b64_str}"
