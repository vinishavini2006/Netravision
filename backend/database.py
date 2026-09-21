"""
NetraVision - Local Clinical SQLite Database
Manages patient registration, screening archives, and clinical XAI records.
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "netravision.db")


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Patients table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        contact_number TEXT,
        village_location TEXT,
        diabetes_duration_years REAL,
        known_hba1c REAL,
        clinical_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Screenings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        screening_id TEXT UNIQUE NOT NULL,
        patient_id TEXT NOT NULL,
        eye TEXT NOT NULL,
        predicted_grade INTEGER NOT NULL,
        grade_label TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        probabilities_json TEXT NOT NULL,
        original_image_path TEXT,
        gradcam_image_path TEXT,
        lesion_count INTEGER DEFAULT 0,
        lesions_json TEXT,
        urgency TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    """)
    conn.commit()
    conn.close()


def generate_patient_id() -> str:
    """Generates standardized rural health ID: NV-PHC-YYYY-XXXX"""
    year = datetime.datetime.now().year
    random_suffix = hex(int(datetime.datetime.now().timestamp() * 1000) % 65536)[2:].upper().zfill(4)
    return f"NV-PHC-{year}-{random_suffix}"


def generate_screening_id() -> str:
    """Generates standardized screening ID: SCR-YYYYMMDD-XXXX"""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    random_suffix = hex(int(datetime.datetime.now().timestamp() * 1000) % 65536)[2:].upper().zfill(4)
    return f"SCR-{date_str}-{random_suffix}"


def register_patient(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    patient_id = data.get("patient_id") or generate_patient_id()
    
    cursor.execute("""
    INSERT INTO patients (
        patient_id, full_name, age, gender, contact_number,
        village_location, diabetes_duration_years, known_hba1c, clinical_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(patient_id) DO UPDATE SET
        full_name=excluded.full_name,
        age=excluded.age,
        gender=excluded.gender,
        contact_number=excluded.contact_number,
        village_location=excluded.village_location,
        diabetes_duration_years=excluded.diabetes_duration_years,
        known_hba1c=excluded.known_hba1c,
        clinical_notes=excluded.clinical_notes
    """, (
        patient_id,
        data.get("full_name", "Anonymous Patient"),
        int(data.get("age", 50)),
        data.get("gender", "Unspecified"),
        data.get("contact_number", ""),
        data.get("village_location", ""),
        float(data.get("diabetes_duration_years", 0.0)),
        float(data.get("known_hba1c", 0.0)) if data.get("known_hba1c") else None,
        data.get("clinical_notes", "")
    ))
    conn.commit()
    conn.close()
    
    return get_patient_by_id(patient_id)


def get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_patients(limit: int = 30) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_screening(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    screening_id = data.get("screening_id") or generate_screening_id()
    
    cursor.execute("""
    INSERT INTO screenings (
        screening_id, patient_id, eye, predicted_grade, grade_label,
        confidence_score, probabilities_json, original_image_path,
        gradcam_image_path, lesion_count, lesions_json, urgency, recommendation
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        screening_id,
        data["patient_id"],
        data.get("eye", "OD"),
        int(data["predicted_grade"]),
        data["grade_label"],
        float(data["confidence_score"]),
        json.dumps(data.get("probabilities", {})),
        data.get("original_image_path", ""),
        data.get("gradcam_image_path", ""),
        int(data.get("lesion_count", 0)),
        json.dumps(data.get("lesions", [])),
        data.get("urgency", "Routine"),
        data.get("recommendation", "")
    ))
    conn.commit()
    conn.close()
    
    return get_screening_by_id(screening_id)


def get_screening_by_id(screening_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.*, p.full_name, p.age, p.gender, p.village_location, p.known_hba1c
    FROM screenings s
    LEFT JOIN patients p ON s.patient_id = p.patient_id
    WHERE s.screening_id = ?
    """, (screening_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    res = dict(row)
    res["probabilities"] = json.loads(res.get("probabilities_json") or "{}")
    res["lesions"] = json.loads(res.get("lesions_json") or "[]")
    return res


def list_recent_screenings(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT s.screening_id, s.patient_id, s.eye, s.predicted_grade, s.grade_label,
           s.confidence_score, s.urgency, s.created_at, p.full_name, p.age, p.gender
    FROM screenings s
    LEFT JOIN patients p ON s.patient_id = p.patient_id
    ORDER BY s.created_at DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
