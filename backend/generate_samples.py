"""
NetraVision - Retinal Fundus Sample Generator
Generates medically realistic synthetic fundus images for ICDR Grades 0 through 4
so the clinical AI pipeline can be demonstrated and verified immediately without external downloads.
"""

import os
import math
import numpy as np
import cv2

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def draw_retinal_vasculature(img: np.ndarray, center_disc: tuple, width: int, height: int, rng: np.random.RandomState):
    """Draws realistic branching retinal arterial and venous arcades originating from the optic disc."""
    # Main branches: Superotemporal, Inferotemporal, Superonasal, Inferonasal
    branches = [
        # (initial_angle_rad, curvature, length, thickness, color)
        (-0.6, 0.4, width * 0.45, 5, (20, 10, 140)),   # Superotemporal vein (dark red)
        (-0.7, 0.4, width * 0.42, 3, (40, 30, 200)),   # Superotemporal artery (bright red)
        (0.6, -0.4, width * 0.45, 5, (20, 10, 140)),    # Inferotemporal vein
        (0.7, -0.4, width * 0.42, 3, (40, 30, 200)),    # Inferotemporal artery
        (-2.4, -0.3, width * 0.35, 4, (25, 15, 150)),   # Superonasal
        (2.4, 0.3, width * 0.35, 4, (25, 15, 150)),    # Inferonasal
    ]
    
    for init_angle, curve, length, thick, col in branches:
        pts = [center_disc]
        cur_x, cur_y = float(center_disc[0]), float(center_disc[1])
        angle = init_angle
        step = 10.0
        steps = int(length / step)
        
        for s in range(steps):
            angle += curve / steps + rng.normal(0, 0.04)
            cur_x += step * math.cos(angle)
            cur_y += step * math.sin(angle)
            pts.append((int(cur_x), int(cur_y)))
            
            # Add micro-branches
            if s > 5 and s % 4 == 0 and rng.rand() > 0.4:
                b_angle = angle + (0.5 if rng.rand() > 0.5 else -0.5)
                bx, by = cur_x, cur_y
                sub_pts = [(int(bx), int(by))]
                for _ in range(rng.randint(3, 7)):
                    bx += 8.0 * math.cos(b_angle)
                    by += 8.0 * math.sin(b_angle)
                    sub_pts.append((int(bx), int(by)))
                cv2.polylines(img, [np.array(sub_pts)], False, col, max(1, thick - 2), cv2.LINE_AA)

        cv2.polylines(img, [np.array(pts)], False, col, thick, cv2.LINE_AA)


def generate_base_fundus(size: int = 512, seed: int = 42) -> np.ndarray:
    """Generates the base fundus eye scan with choroid background, optic disc, and macula."""
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = (size // 2, size // 2)
    radius = int(size * 0.46)
    
    # 1. Radial gradient for choroidal illumination
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    
    # Warm retinal orange-red: BGR (10, 45, 185) center to (5, 20, 110) margin
    r_channel = (195 - norm_dist * 75 + rng.normal(0, 2, (size, size))).clip(0, 255)
    g_channel = (60 - norm_dist * 35 + rng.normal(0, 2, (size, size))).clip(0, 255)
    b_channel = (15 - norm_dist * 10 + rng.normal(0, 1, (size, size))).clip(0, 255)
    
    img = np.stack([b_channel, g_channel, r_channel], axis=-1).astype(np.uint8)
    
    # 2. Optic Disc (bright yellowish-pink ellipse on nasal side)
    disc_center = (int(size * 0.30), int(size * 0.48))
    disc_axes = (int(size * 0.08), int(size * 0.095))
    cv2.ellipse(img, disc_center, disc_axes, 0, 0, 360, (70, 160, 245), -1, cv2.LINE_AA)  # BGR
    # Physiologic cup inside disc
    cup_center = (disc_center[0] + 2, disc_center[1])
    cup_axes = (int(disc_axes[0] * 0.45), int(disc_axes[1] * 0.45))
    cv2.ellipse(img, cup_center, cup_axes, 0, 0, 360, (130, 210, 255), -1, cv2.LINE_AA)
    
    # 3. Retinal Vasculature
    draw_retinal_vasculature(img, disc_center, size, size, rng)
    
    # 4. Macula & Fovea (darker reddish circular zone on temporal side)
    macula_center = (int(size * 0.62), int(size * 0.50))
    macula_radius = int(size * 0.09)
    # Darken macula gently
    macula_mask = (np.sqrt((x - macula_center[0])**2 + (y - macula_center[1])**2) < macula_radius)
    img[macula_mask] = (img[macula_mask] * 0.85).astype(np.uint8)
    # Foveal avascular reflex (tiny darker center dot)
    cv2.circle(img, macula_center, int(size * 0.015), (5, 20, 110), -1, cv2.LINE_AA)
    
    # 5. Circular FOV mask to simulate clinical fundus camera aperture
    fov_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(fov_mask, center, radius, 255, -1, cv2.LINE_AA)
    # Smooth boundary
    fov_mask = cv2.GaussianBlur(fov_mask, (15, 15), 0)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (fov_mask / 255.0)).astype(np.uint8)
        
    return img


def add_microaneurysms(img: np.ndarray, count: int, rng: np.random.RandomState):
    """Draws tiny focal red spots (microaneurysms)."""
    size = img.shape[0]
    macula_x, macula_y = int(size * 0.60), int(size * 0.50)
    for _ in range(count):
        mx = int(macula_x + rng.normal(0, size * 0.15))
        my = int(macula_y + rng.normal(0, size * 0.15))
        r = rng.choice([1, 2, 3])
        # Dark deep red spot (BGR: 5, 5, 80)
        cv2.circle(img, (mx, my), r, (5, 5, 95), -1, cv2.LINE_AA)


def add_hemorrhages(img: np.ndarray, count: int, rng: np.random.RandomState, severe: bool = False):
    """Draws flame and dot-blot intraretinal hemorrhages."""
    size = img.shape[0]
    for _ in range(count):
        hx = rng.randint(int(size * 0.25), int(size * 0.80))
        hy = rng.randint(int(size * 0.20), int(size * 0.80))
        ax = rng.randint(4, 12 if not severe else 22)
        ay = rng.randint(2, 6 if not severe else 14)
        angle = rng.randint(0, 180)
        # Blot hemorrhage: irregular dark red blob
        cv2.ellipse(img, (hx, hy), (ax, ay), angle, 0, 360, (5, 10, 85), -1, cv2.LINE_AA)


def add_hard_exudates(img: np.ndarray, count: int, rng: np.random.RandomState):
    """Draws lipid exudates (bright yellowish glistening specks / rings)."""
    size = img.shape[0]
    center_ring = (int(size * 0.65), int(size * 0.45))
    for _ in range(count):
        ex = int(center_ring[0] + rng.normal(0, size * 0.08))
        ey = int(center_ring[1] + rng.normal(0, size * 0.08))
        r = rng.randint(2, 5)
        # Bright yellow-white lipid deposit (BGR: 120, 240, 255)
        cv2.circle(img, (ex, ey), r, (110, 240, 255), -1, cv2.LINE_AA)


def add_neovascularization(img: np.ndarray, rng: np.random.RandomState):
    """Draws delicate fronds of abnormal new vessels (NVD/NVE) for Proliferative DR."""
    size = img.shape[0]
    disc_center = (int(size * 0.30), int(size * 0.48))
    # Vessel fronds over optic disc
    for _ in range(12):
        pts = [disc_center]
        cur_x, cur_y = float(disc_center[0]), float(disc_center[1])
        angle = rng.uniform(0, 2 * math.pi)
        for _ in range(rng.randint(6, 12)):
            cur_x += 4.0 * math.cos(angle) + rng.normal(0, 1.5)
            cur_y += 4.0 * math.sin(angle) + rng.normal(0, 1.5)
            pts.append((int(cur_x), int(cur_y)))
        cv2.polylines(img, [np.array(pts)], False, (15, 15, 180), 1, cv2.LINE_AA)
        
    # Large pre-retinal hemorrhage boat/veil
    veil_center = (int(size * 0.55), int(size * 0.65))
    cv2.ellipse(img, veil_center, (int(size * 0.16), int(size * 0.08)), 15, 0, 180, (5, 5, 75), -1, cv2.LINE_AA)


def generate_all_samples():
    """Builds and caches sample images for Grades 0 through 4."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    specs = [
        ("grade0_normal.jpg", 0, "No Diabetic Retinopathy", 101),
        ("grade1_mild.jpg", 1, "Mild NPDR", 202),
        ("grade2_moderate.jpg", 2, "Moderate NPDR", 303),
        ("grade3_severe.jpg", 3, "Severe NPDR", 404),
        ("grade4_proliferative.jpg", 4, "Proliferative DR", 505),
    ]
    
    for filename, grade, label, seed in specs:
        out_path = os.path.join(SAMPLES_DIR, filename)
        if os.path.exists(out_path):
            continue
            
        rng = np.random.RandomState(seed)
        img = generate_base_fundus(size=512, seed=seed)
        
        if grade == 1:
            add_microaneurysms(img, count=12, rng=rng)
        elif grade == 2:
            add_microaneurysms(img, count=28, rng=rng)
            add_hemorrhages(img, count=10, rng=rng, severe=False)
            add_hard_exudates(img, count=18, rng=rng)
        elif grade == 3:
            add_microaneurysms(img, count=45, rng=rng)
            add_hemorrhages(img, count=32, rng=rng, severe=True)
            add_hard_exudates(img, count=30, rng=rng)
        elif grade == 4:
            add_microaneurysms(img, count=40, rng=rng)
            add_hemorrhages(img, count=25, rng=rng, severe=True)
            add_hard_exudates(img, count=25, rng=rng)
            add_neovascularization(img, rng=rng)
            
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"Generated sample: {filename} (Grade {grade}: {label})")


if __name__ == "__main__":
    generate_all_samples()
