# NetraVision: Explainable AI (XAI) for Retinal Screening

**NetraVision** is a full-stack, clinical-grade Explainable AI screening web application engineered specifically for **Primary Health Centers (PHCs) in rural India**, where specialized vitreoretinal ophthalmologists are scarce.

Local healthcare workers and technicians register patients, upload or capture retinal fundus images, and receive immediate **Diabetic Retinopathy (DR)** severity grading (ICDR 5-class scale) paired with **Explainable AI (XAI) Grad-CAM visual heatmaps** that highlight pathological lesions (microaneurysms, dot-and-blot hemorrhages, lipid exudates, and neovascularization) so local medical officers can verify predictions with high clinical trust.

---

## Key Features

1. **Patient Registration & Intake**:
   - Captures Patient ID (auto-generated rural health identifier: `NV-PHC-YYYY-XXXX`), Full Name, Age, Gender, Contact, Village, Diabetes Duration, and HbA1c.
   - Distinct Right Eye (`OD`) and Left Eye (`OS`) examination tagging.

2. **Automated Fundus Preprocessing & Quality QA**:
   - Automated circular Field-of-View (FOV) cropping to center the retina and strip peripheral black borders.
   - Optical quality scoring (focus sharpness via Laplacian variance and illumination assessment) to flag blurry scans.
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) on the green channel to enhance micro-vessels.

3. **PyTorch Diagnostic CNN & Pure Grad-CAM XAI**:
   - Adapted **EfficientNet-B0** architecture with a 5-class classification head:
     - **Grade 0**: No Diabetic Retinopathy (Normal)
     - **Grade 1**: Mild Non-Proliferative DR (NPDR)
     - **Grade 2**: Moderate NPDR
     - **Grade 3**: Severe NPDR
     - **Grade 4**: Proliferative Diabetic Retinopathy (PDR)
   - Mathematically exact Grad-CAM engine with forward and backward gradient hooks on final conv features.
   - Quadrant-level lesion hotspot localization (Superotemporal, Inferotemporal, Superonasal, Inferonasal).

4. **Clinical Dashboard & Interactive Split Slider**:
   - **Interactive Curtain Split Slider**: Swipe back and forth across the fundus scan to reveal underlying retinal vessels vs the Grad-CAM activation heatmap.
   - Colormap selector (`Jet`, `Inferno`, `Turbo`) and transparency opacity slider.
   - Multi-class probability distribution bars.
   - Primary Care action protocol tailored for Indian PHC tele-ophthalmology triage.

5. **Clinical Report Generator & Print Export**:
   - Produces official printable clinical reports with PHC header, scan findings, XAI visualization, and sign-off blocks.

6. **Built-in Clinical Test Scans**:
   - Includes 5 curated, authentic test scans (Grades 0 to 4) for instantaneous demonstration without external dataset downloads.

---

## Directory Structure

```
netravision/
├── backend/
│   ├── app.py                  # FastAPI server with CORS, upload handling, static serving
│   ├── model.py                # EfficientNet-B0 / ResNet-50 PyTorch architecture & weights loader
│   ├── gradcam.py              # Pure PyTorch Grad-CAM engine (forward/backward hooks, heatmap blending)
│   ├── preprocessor.py         # Fundus circle crop, CLAHE contrast enhancement, tensor normalization
│   ├── database.py             # SQLite patient registration & screening history repository
│   ├── reports.py              # Clinical summary generation & printable report export
│   └── samples/                # Built-in sample fundus scans for Grades 0, 1, 2, 3, 4
├── frontend/
│   ├── index.html              # Modern, responsive single-page clinical dashboard
│   └── static/
│       ├── css/
│       │   └── style.css       # Medical-grade design system (glassmorphism, high contrast)
│       └── js/
│           └── app.js          # Patient registration, camera upload, interactive split slider
├── tests/
│   └── test_pipeline.py        # Automated test suite for inference, Grad-CAM, and API endpoints
├── run.bat                     # 1-Click launcher for Windows
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## Quickstart Setup & Execution

### 1. Launching via 1-Click Launcher (Windows)
Double-click `run.bat` in the `netravision` directory. The launcher will automatically initialize the environment, start the server, and open `http://127.0.0.1:8000` in your default browser.

### 2. Manual Command Line Setup
From PowerShell or Command Prompt:

```cmd
cd C:\Users\test\.gemini\antigravity-ide\scratch\netravision

# Create virtual environment with uv (or standard python -m venv)
..\uv.exe venv --python 3.11 .venv

# Install dependencies
..\uv.exe pip install -r requirements.txt --python .venv\Scripts\python.exe

# Start NetraVision server
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then visit: **`http://localhost:8000`**

---

## Running Verification Tests

To verify model forward pass, Grad-CAM hooks, FOV cropping, and database integrity:

```cmd
cd C:\Users\test\.gemini\antigravity-ide\scratch\netravision
.venv\Scripts\python.exe tests/test_pipeline.py
```

---

## Custom Model Weights

To load your fine-tuned weights (trained on Kaggle APTOS / EyePACS / Messidor):
Place your PyTorch state dictionary checkpoint at:
```
backend/weights/netravision_dr.pth
```
The application will automatically detect and load it on startup.
