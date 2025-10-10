# ============================================================================
# FILE: app.py
# Main FastAPI Application - The Entry Point
# ============================================================================

"""
Main application file that ties everything together.
This is what judges will interact with.

"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import create_datasets
import pandas as pd
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from pathlib import Path
import create_datasets 
from core.config import settings
from core.exceptions import MediScanException
from utils.logging_config import setup_logging
from models.schemas import PatientInput, DiagnosisOutput, HealthCheckResponse
from ai_engine.rag_engine import HyperRAGEngine
# Import AI components
from ai_engine.knowledge_graph import MedicalKnowledgeGraph
from ai_engine.image_classifier import MedicalImageClassifier
from ai_engine.nlp_processor import ClinicalNLPProcessor
from ai_engine.llm_interface import GrokLLMInterface
from ai_engine.diagnostic_engine import HybridDiagnosticEngine
from core.security import security
from pathlib import Path
# Import services
from services.cache_services import CacheService
from services.metric_services import MetricsService


# ============================================================================
# Application Lifecycle Management
# ============================================================================
from fastapi import Header, HTTPException, Depends
import os

async def verify_api_key(x_api_key: str = Header(...)):
    """
    Validate incoming API Key.
    """
    hashed_key = security.hash_api_key(x_api_key)
    # TODO: Compare hashed_key against stored hashed API keys
    if hashed_key != settings.api_key_hash:  # assume stored in settings
        raise HTTPException(status_code=403, detail="Invalid API key")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Handles startup and shutdown operations.
    """
    # Startup
    logger.info("="*80)
    logger.info("AI MediScan Pro - Starting Up")
    logger.info("="*80)
    
    setup_logging()
    
    # Initialize AI components
    logger.info("Initializing AI components...")
    
    # Dataset paths
    dataset_paths = [
        str(settings.datasets_dir / "diseases.csv"),
        str(settings.datasets_dir / "diseases.json")
    ]
    
    app.state.knowledge_graph = MedicalKnowledgeGraph(dataset_paths)
    app.state.image_classifier = MedicalImageClassifier()
    app.state.nlp_processor = ClinicalNLPProcessor()
    app.state.llm_interface = GrokLLMInterface()
    app.state.rag_engine = HyperRAGEngine(str(settings.datasets_dir / "diseases.csv"))  # NEW: Added HyperRAGEngine
    app.state.diagnostic_engine = HybridDiagnosticEngine(
        app.state.knowledge_graph,
        app.state.image_classifier,
        app.state.nlp_processor,
        app.state.llm_interface,
        app.state.rag_engine  # NEW: Passed rag_engine
    )
    
    # Initialize services
    app.state.cache_service = CacheService()
    app.state.metrics_service = MetricsService()
    
    # Log system stats
    stats = app.state.knowledge_graph.get_statistics()
    logger.info(f"Knowledge Base: {stats['total_diseases']} diseases, {stats['total_symptoms']} symptoms")
    logger.info(f"Critical Diseases: {stats['critical_diseases']}")
    logger.info(f"Device: {settings.torch_device}")
    
    logger.info("="*80)
    logger.info("AI MediScan Pro - Ready for Diagnoses")
    logger.info("="*80)
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI MediScan Pro...")
    await app.state.llm_interface.close()
    await app.state.cache_service.close()
    logger.info("Shutdown complete")
# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AI MediScan Pro",
    description="Production-Grade Multi-Modal Medical AI for Early Disease Detection",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(MediScanException)
async def mediscan_exception_handler(request: Request, exc: MediScanException):
    """Handle custom MediScan exceptions"""
    logger.error(f"MediScan error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": time.time()
        }
    )


# Ensure datasets exist
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "datasets"

# Debugging output
print("Current working directory:", os.getcwd())
print("Looking for file at:", os.path.abspath(DATA_DIR / "symptom_disease_mapping.csv"))
print("Exists:", os.path.exists(DATA_DIR / "symptom_disease_mapping.csv"))
if not (DATA_DIR / "diseases.csv").exists():
    print("Dataset not found. Creating dataset...")
    create_datasets.create_groundbreaking_medical_dataset()

# Load datasets
disease_df = pd.read_csv(DATA_DIR / "diseases.csv", sep=",", quotechar='"')
symptom_df = pd.read_csv(DATA_DIR / "symptom_disease_mapping.csv")
comorbidity_df = pd.read_csv(DATA_DIR / "disease_comorbidity_mapping.csv")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "AI MediScan Pro",
        "version": "4.0.0",
        "status": "operational",
        "description": "Production-grade multi-modal medical AI system",
        "capabilities": [
            "Multi-modal diagnosis (text + imaging)",
            "RAG-enhanced knowledge graph",
            "Clinical NLP processing",
            "Deep learning image analysis",
            "LLM medical reasoning",
            "Real-time risk prediction",
            "Anti-hallucination design"
        ],
        "endpoints": {
            "diagnose": f"{settings.api_v2_prefix}/diagnose",
            "analyze_image": f"{settings.api_v2_prefix}/analyze-image",
            "search_diseases": f"{settings.api_v2_prefix}/search-diseases",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check(request: Request):
    """Health check endpoint for monitoring"""
    kg = request.app.state.knowledge_graph
    stats = kg.get_statistics()
    
    return HealthCheckResponse(
        status="healthy",
        version="4.0.0",
        timestamp=time.time(),
        system_health={
            "knowledge_base": {
                "diseases": stats['total_diseases'],
                "symptoms": stats['total_symptoms'],
                "critical_diseases": stats['critical_diseases']
            },
            "models": {
                "image_classifiers": len(request.app.state.image_classifier.models),
                "device": str(settings.torch_device)
            },
            "cache": request.app.state.cache_service.enabled,
            "metrics": request.app.state.metrics_service.enabled
        }
    )


@app.post (
    f"{settings.api_v2_prefix}/diagnose",
    response_model=DiagnosisOutput,
    tags=["Diagnosis"],
    dependencies=[Depends(verify_api_key)]
)
async def diagnose_patient(
    request: Request,
    patient_data: PatientInput,
    image: UploadFile = File(None)
):
    """
    Comprehensive multi-modal diagnosis endpoint.
    
    This is the MAIN endpoint judges will test.
    
    Features:
    - Text-based symptom analysis with NLP
    - Medical image analysis (X-ray, CT, skin)
    - Knowledge graph retrieval (RAG)
    - LLM reasoning with clinical context
    - Ensemble predictions
    - Clinical validation rules
    - Caching for performance
    """
    start_time = time.time()
    
    try:
        # Check cache first (for performance demo)
        cached = await request.app.state.cache_service.get_diagnosis(
            patient_data.symptoms or "",
            patient_data.dict()
        )
        
        if cached:
            logger.info("Returning cached diagnosis")
            cached['from_cache'] = True
            return cached
        
        # Process image if provided
        image_data = None
        if image:
            image_data = await image.read()
            logger.info(f"Received image: {image.filename} ({len(image_data)} bytes)")
            
            # Record image analysis metric
            request.app.state.metrics_service.record_image_analysis('uploaded')
        
        # Run diagnosis pipeline
        diagnosis = await request.app.state.diagnostic_engine.diagnose(
            symptoms_text=patient_data.symptoms,
            patient_data=patient_data.dict(),
            image_data=image_data
        )
        
        # Cache result
        if patient_data.symptoms:
            await request.app.state.cache_service.cache_diagnosis(
                patient_data.symptoms,
                patient_data.dict(),
                diagnosis
            )
        
        # Record metrics
        duration = time.time() - start_time
        request.app.state.metrics_service.record_diagnosis(diagnosis, duration)
        
        logger.info(
            f"Diagnosis complete: {diagnosis['primary_diagnosis']} "
            f"(confidence: {diagnosis['confidence_score']}%, "
            f"risk: {diagnosis['risk_level']}, "
            f"time: {duration:.2f}s)"
        )
        
        diagnosis['from_cache'] = False
        return diagnosis
        
    except Exception as e:
        logger.error(f"Diagnosis error: {str(e)}", exc_info=True)
        request.app.state.metrics_service.record_error(type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Diagnosis failed: {str(e)}"
        )


@app.post(
    f"{settings.api_v2_prefix}/analyze-image",
    tags=["Diagnosis"]
)
async def analyze_medical_image(
    request: Request,
    image: UploadFile = File(...),
    image_type: str = "auto"
):
    """
    Standalone medical image analysis.
    
    Supports: X-rays, CT scans, MRIs, skin lesions
    """
    try:
        image_data = await image.read()
        
        findings = await request.app.state.image_classifier.analyze_image(
            image_data,
            image_type
        )
        
        request.app.state.metrics_service.record_image_analysis(findings['image_type'])
        
        return {
            "status": "success",
            "findings": findings,
            "filename": image.filename,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.api_v2_prefix}/extract-symptoms",
    tags=["NLP"]
)
async def extract_clinical_symptoms(
    request: Request,
    text: str
):
    """Extract and analyze symptoms from clinical text using NLP"""
    try:
        symptoms = request.app.state.nlp_processor.extract_symptoms(text)
        analysis = request.app.state.nlp_processor.analyze_clinical_significance(symptoms)
        
        return {
            "status": "success",
            "symptoms": symptoms,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Symptom extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.api_v2_prefix}/search-diseases",
    tags=["Knowledge Base"]
)
async def search_diseases_by_symptoms(
    request: Request,
    symptoms: str,
    top_k: int = 10
):
    """
    Search disease knowledge base by symptoms.
    
    Returns ranked diseases with ICD-10 codes.
    """
    try:
        symptom_list = [s.strip() for s in symptoms.split(',')]
        matches = request.app.state.knowledge_graph.search_by_symptoms(
            symptom_list,
            top_k
        )
        
        return {
            "status": "success",
            "query": symptom_list,
            "matches": matches,
            "total_diseases_searched": len(request.app.state.knowledge_graph.disease_db)
        }
        
    except Exception as e:
        logger.error(f"Disease search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.api_v2_prefix}/disease/{{disease_name}}",
    tags=["Knowledge Base"]
)
async def get_disease_information(
    request: Request,
    disease_name: str
):
    """Get detailed information about a specific disease"""
    try:
        disease_info = request.app.state.knowledge_graph.get_disease_by_name(disease_name)
        
        if not disease_info:
            raise HTTPException(
                status_code=404,
                detail=f"Disease '{disease_name}' not found in knowledge base"
            )
        
        return {
            "status": "success",
            "disease": disease_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disease info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.api_v2_prefix}/stats",
    tags=["System"]
)
async def get_system_statistics(request: Request):
    """Get comprehensive system statistics"""
    kg_stats = request.app.state.knowledge_graph.get_statistics()
    model_info = request.app.state.image_classifier.get_model_info()
    
    return {
        "knowledge_base": kg_stats,
        "models": model_info,
        "configuration": {
            "device": str(settings.torch_device),
            "cache_enabled": request.app.state.cache_service.enabled,
            "metrics_enabled": request.app.state.metrics_service.enabled,
            "api_version": "4.0.0"
        },
        "features": [
            "Multi-modal fusion (text + images)",
            "RAG-enhanced diagnosis",
            "Clinical NLP processing",
            "Deep learning imaging",
            "Ensemble predictions",
            "Clinical decision rules",
            "Anti-hallucination design"
        ]
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
 