
# ============================================================================
# FILE: models/schemas.py
# Pydantic models for request/response validation
# ============================================================================

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class GenderEnum(str, Enum):
    """Valid gender options"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class SeverityEnum(str, Enum):
    """Symptom severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class RiskLevelEnum(str, Enum):
    """Disease risk levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VitalSigns(BaseModel):
    """Patient vital signs"""
    temperature: Optional[float] = Field(None, ge=35.0, le=42.0, description="°C")
    blood_pressure_systolic: Optional[int] = Field(None, ge=60, le=250, description="mmHg")
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150, description="mmHg")
    heart_rate: Optional[int] = Field(None, ge=30, le=220, description="bpm")
    respiratory_rate: Optional[int] = Field(None, ge=8, le=60, description="breaths/min")
    oxygen_saturation: Optional[float] = Field(None, ge=70.0, le=100.0, description="%")


class PatientInput(BaseModel):
    """
    Patient data input model with validation.
    
    This ensures all inputs are properly validated before processing.
    """
    patient_id: Optional[str] = Field(None, min_length=1, max_length=100)
    symptoms: Optional[str] = Field(None, min_length=3, max_length=2000)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[GenderEnum] = None
    medical_history: Optional[str] = Field(None, max_length=5000)
    duration: Optional[str] = Field(None, max_length=200)
    severity: Optional[SeverityEnum] = None
    vital_signs: Optional[VitalSigns] = None
    
    @validator('symptoms')
    def validate_symptoms(cls, v):
        """Ensure symptoms contain meaningful text"""
        if v and len(v.strip()) < 3:
            raise ValueError("Symptoms must be at least 3 characters")
        return v.strip() if v else None
    
    @validator('patient_id')
    def hash_patient_id(cls, v):
        """Hash patient ID for privacy"""
        if v:
            import hashlib
            return hashlib.sha256(v.encode()).hexdigest()[:16]
        return None


class DifferentialDiagnosis(BaseModel):
    """Single differential diagnosis"""
    disease: str
    probability: float = Field(..., ge=0, le=100)
    rationale: str
    icd10: Optional[str] = None


class SymptomHighlight(BaseModel):
    """Highlighted symptom with importance"""
    symptom: str
    category: str
    severity: int = Field(..., ge=1, le=4)
    duration: Optional[int] = None
    importance: str = Field(..., pattern="^(low|medium|high)$")


class ImageFindings(BaseModel):
    """Medical image analysis results"""
    image_type: str
    primary_finding: str
    confidence: float = Field(..., ge=0, le=1)
    requires_immediate_attention: bool
    clinical_interpretation: str


class DiagnosisOutput(BaseModel):
    """
    Complete diagnosis response model.
    
    This is what judges will see - make it comprehensive!
    """
    # Core Diagnosis
    primary_diagnosis: str
    confidence_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevelEnum
    
    # Clinical Reasoning
    reasoning: str
    knowledge_base_alignment: str
    
    # Differential Diagnoses
    differential_diagnoses: List[DifferentialDiagnosis]
    
    # Safety
    red_flags: List[str]
    immediate_actions: List[str]
    
    # Recommendations
    recommended_tests: List[str]
    clinical_notes: str
    
    # Explainability
    highlighted_symptoms: Optional[List[SymptomHighlight]] = None
    image_findings: Optional[ImageFindings] = None
    
    # Metadata
    timestamp: datetime
    processing_time_ms: float
    confidence_level: str
    data_sources: List[str]
    
    # System Info
    system_version: str = "4.0.0"
    model_versions: Dict[str, str] = {}


class HealthCheckResponse(BaseModel):
    """API health check response"""
    status: str
    version: str
    timestamp: datetime
    system_health: Dict[str, Any]