"""
NetraVision - Clinical Screening Report Generator
Generates clinical summary documentation and printable HTML reports
tailored for Rural Primary Health Centers and Tele-Ophthalmology referrals.
"""

import datetime
from typing import Dict, Any

CLINICAL_PROTOCOLS = {
    0: {
        "title": "No Apparent Retinopathy (ICDR Grade 0)",
        "summary": "Retinal fundus examination reveals clear optical media, normal cup-to-disc ratio, distinct retinal vasculature, and absence of microaneurysms or hemorrhagic lesions.",
        "primary_care_plan": [
            "Maintain routine annual diabetic eye screening (recall in 12 months).",
            "Reinforce standard diabetes lifestyle management and glycemic targets (HbA1c < 7.0%).",
            "Maintain systemic blood pressure control (< 130/80 mmHg).",
            "Advise patient to return promptly if sudden floaters, flashes, or visual blur occur."
        ],
        "referral_urgency": "Routine / Annual Recall"
    },
    1: {
        "title": "Mild Non-Proliferative Diabetic Retinopathy (ICDR Grade 1)",
        "summary": "Focal microaneurysms detected in the intraretinal capillary bed. No macular edema, hard lipid exudates, or blot hemorrhages observed.",
        "primary_care_plan": [
            "Schedule follow-up retinal screening within 9 to 12 months.",
            "Intensify glycemic control with Primary Health Center medical officer.",
            "Order baseline serum lipid profile and HbA1c assessment.",
            "Educate patient on early symptoms of diabetic maculopathy."
        ],
        "referral_urgency": "Low Priority / Primary Care Follow-up (9-12 Months)"
    },
    2: {
        "title": "Moderate Non-Proliferative Diabetic Retinopathy (ICDR Grade 2)",
        "summary": "Multiple intraretinal microaneurysms, dot-and-blot hemorrhages, and scattered lipid exudates. AI Grad-CAM attention highlights pathological vascular leakage.",
        "primary_care_plan": [
            "Referral to district hospital ophthalmology clinic within 3 to 6 months for slit-lamp biomicroscopy & OCT (Optical Coherence Tomography).",
            "Strict glycemic and hypertensive control; evaluate renal parameters (serum creatinine, microalbuminuria).",
            "Screen fellow eye immediately if not yet examined."
        ],
        "referral_urgency": "Moderate Priority / Ophthalmologist Referral (3-6 Months)"
    },
    3: {
        "title": "Severe Non-Proliferative Diabetic Retinopathy (ICDR Grade 3)",
        "summary": "Extensive intraretinal hemorrhages meeting the 4-2-1 rule criteria, definite venous beading, or microvascular abnormalities. High imminent risk of progression to proliferative DR.",
        "primary_care_plan": [
            "Urgent referral to retina specialist / vitreoretinal center within 2 to 4 weeks.",
            "Advise patient to avoid strenuous physical exertion or Valsalva maneuvers to minimize hemorrhage risk.",
            "Urgent tele-ophthalmology consultation with nodal tertiary hospital.",
            "Comprehensive evaluation for diabetic macular edema (DME)."
        ],
        "referral_urgency": "Urgent Referral / Retina Specialist (2-4 Weeks)"
    },
    4: {
        "title": "Proliferative Diabetic Retinopathy (ICDR Grade 4)",
        "summary": "Critical pathological state marked by neovascularization (disc or elsewhere) and potential vitreous/preretinal hemorrhage. Imminent threat of irreversible vision loss.",
        "primary_care_plan": [
            "EMERGENCY referral to tertiary vitreoretinal center within 24 to 48 hours.",
            "Immediate candidate for Panretinal Photocoagulation (PRP) laser or intravitreal anti-VEGF pharmacotherapy.",
            "Strict bed rest; avoid sudden head trauma or vigorous bending.",
            "Counsel family on emergency clinical protocol."
        ],
        "referral_urgency": "EMERGENCY / Immediate Tertiary Referral (24-48 Hours)"
    }
}


def build_clinical_report_html(screening_data: Dict[str, Any], patient_data: Dict[str, Any]) -> str:
    """Generates an official printable clinical report as HTML."""
    grade = int(screening_data.get("predicted_grade", 0))
    protocol = CLINICAL_PROTOCOLS.get(grade, CLINICAL_PROTOCOLS[0])
    
    screening_date = screening_data.get("created_at") or datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    original_img = screening_data.get("original_image_path") or ""
    gradcam_img = screening_data.get("gradcam_image_path") or ""
    
    # Format lesion items
    lesions = screening_data.get("lesions", [])
    lesion_rows = ""
    if lesions:
        for l in lesions[:5]:
            lesion_rows += f"""
            <tr>
                <td>Region #{l.get('region_id', '-')}</td>
                <td>{l.get('quadrant', 'Central')}</td>
                <td>{int(l.get('attention_intensity', 0) * 100)}%</td>
                <td>{l.get('clinical_significance', 'Focal Attention')}</td>
            </tr>
            """
    else:
        lesion_rows = "<tr><td colspan='4' style='text-align:center; color:#6b7280;'>No pathological lesion hotspots detected.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NetraVision Clinical Retinal Screening Report - {screening_data.get('screening_id')}</title>
    <style>
        @page {{ size: A4; margin: 15mm; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1f2937;
            background: #ffffff;
            margin: 0;
            padding: 20px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .header {{
            border-bottom: 2px solid #0284c7;
            padding-bottom: 12px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .brand {{
            font-size: 24px;
            font-weight: 800;
            color: #0369a1;
            letter-spacing: -0.5px;
        }}
        .subbrand {{
            font-size: 12px;
            color: #64748b;
            font-weight: 500;
        }}
        .report-badge {{
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            color: #0369a1;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            text-align: right;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .card-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            margin-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .data-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}
        .data-label {{ color: #64748b; font-weight: 500; }}
        .data-val {{ font-weight: 600; color: #0f172a; }}
        .diagnosis-banner {{
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .grade-0 {{ background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }}
        .grade-1 {{ background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; }}
        .grade-2 {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
        .grade-3 {{ background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; }}
        .grade-4 {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
        .visual-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .scan-frame {{
            background: #0f172a;
            border-radius: 8px;
            padding: 8px;
            color: #f8fafc;
        }}
        .scan-frame img {{
            width: 100%;
            max-height: 220px;
            object-fit: contain;
            border-radius: 6px;
            display: block;
        }}
        .scan-caption {{
            margin-top: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 6px;
        }}
        th, td {{
            padding: 6px 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
        .recommendation-list {{
            margin: 6px 0 0 0;
            padding-left: 18px;
        }}
        .recommendation-list li {{
            margin-bottom: 4px;
        }}
        .footer-signatures {{
            margin-top: 24px;
            display: flex;
            justify-content: space-between;
            padding-top: 18px;
            border-top: 1px solid #cbd5e1;
        }}
        .sig-box {{
            width: 200px;
            text-align: center;
            border-top: 1px dashed #94a3b8;
            padding-top: 6px;
            font-size: 11px;
            color: #475569;
        }}
        .print-btn {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #0284c7;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}
        @media print {{
            .print-btn {{ display: none; }}
            body {{ padding: 0; }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Print Clinical Report</button>
    
    <div class="header">
        <div>
            <div class="brand">NetraVision AI</div>
            <div class="subbrand">Primary Health Center Retinal Screening & Tele-Ophthalmology Network</div>
        </div>
        <div class="report-badge">
            <div>Report ID: {screening_data.get('screening_id')}</div>
            <div style="font-size:11px; font-weight:normal;">Date: {screening_date}</div>
        </div>
    </div>

    <!-- Patient & Screening Meta -->
    <div class="grid-2">
        <div class="card">
            <div class="card-title">Patient Demographic Record</div>
            <div class="data-row"><span class="data-label">Full Name:</span><span class="data-val">{patient_data.get('full_name', 'N/A')}</span></div>
            <div class="data-row"><span class="data-label">Patient ID:</span><span class="data-val">{patient_data.get('patient_id', 'N/A')}</span></div>
            <div class="data-row"><span class="data-label">Age / Gender:</span><span class="data-val">{patient_data.get('age', 'N/A')} Yrs / {patient_data.get('gender', 'N/A')}</span></div>
            <div class="data-row"><span class="data-label">Village / Center:</span><span class="data-val">{patient_data.get('village_location') or 'Primary Health Center'}</span></div>
        </div>
        <div class="card">
            <div class="card-title">Clinical Context & Eye Examined</div>
            <div class="data-row"><span class="data-label">Eye Examined:</span><span class="data-val">{screening_data.get('eye', 'OD')} ({'Right Eye (OD)' if screening_data.get('eye') == 'OD' else 'Left Eye (OS)'})</span></div>
            <div class="data-row"><span class="data-label">Diabetes Duration:</span><span class="data-val">{patient_data.get('diabetes_duration_years', '0')} Years</span></div>
            <div class="data-row"><span class="data-label">Latest HbA1c:</span><span class="data-val">{patient_data.get('known_hba1c') or 'Not Recorded'} %</span></div>
            <div class="data-row"><span class="data-label">AI Diagnostic Engine:</span><span class="data-val">EfficientNet-B0 + Grad-CAM XAI</span></div>
        </div>
    </div>

    <!-- Diagnostic Verdict Banner -->
    <div class="diagnosis-banner grade-{grade}">
        <div>
            <div style="font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.5px;">Diabetic Retinopathy Classification (ICDR Scale)</div>
            <div style="font-size:18px; font-weight:800; margin-top:2px;">{protocol['title']}</div>
            <div style="font-size:12px; margin-top:4px;">{protocol['summary']}</div>
        </div>
        <div style="text-align:right; min-width:140px;">
            <div style="font-size:24px; font-weight:900;">{int(float(screening_data.get('confidence_score', 0)) * 100)}%</div>
            <div style="font-size:11px; font-weight:600;">Model Confidence</div>
            <div style="margin-top:4px; font-size:11px; font-weight:700;">{protocol['referral_urgency']}</div>
        </div>
    </div>

    <!-- Scans Viewport -->
    <div class="visual-comparison">
        <div class="scan-frame">
            <img src="{original_img}" alt="Original Fundus Scan" />
            <div class="scan-caption">Original Retinal Fundus Scan ({screening_data.get('eye', 'OD')})</div>
        </div>
        <div class="scan-frame">
            <img src="{gradcam_img}" alt="Explainable AI Grad-CAM Heatmap" />
            <div class="scan-caption">Explainable AI (Grad-CAM) Lesion Heatmap</div>
        </div>
    </div>

    <!-- Lesion Attention Hotspots Table -->
    <div class="card" style="margin-bottom: 16px;">
        <div class="card-title">Explainable AI Detected Lesion Hotspots</div>
        <table>
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Retinal Quadrant</th>
                    <th>Peak Attention</th>
                    <th>Clinical Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {lesion_rows}
            </tbody>
        </table>
    </div>

    <!-- Clinical Action Plan -->
    <div class="card">
        <div class="card-title">Primary Health Center Protocol & Action Directives</div>
        <ul class="recommendation-list">
            {''.join([f"<li><strong>{act}</strong></li>" for act in protocol['primary_care_plan']])}
        </ul>
    </div>

    <!-- Signature Sign-off -->
    <div class="footer-signatures">
        <div class="sig-box">
            <div>Screening Technologist Signature</div>
            <div style="margin-top:2px; font-size:10px;">Primary Health Center Staff</div>
        </div>
        <div class="sig-box">
            <div>Medical Officer / Reviewing Doctor</div>
            <div style="margin-top:2px; font-size:10px;">District Tele-Ophthalmology Unit</div>
        </div>
    </div>
</body>
</html>"""
    return html
