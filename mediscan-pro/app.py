
"""
AI MediScan Pro - Main Application Entry Point
===============================================

This is the main FastAPI application file.
All features are modular - add new features without rewriting this file.

Project Structure:
    app.py (THIS FILE) - Main application
    core/ - Configuration, security, exceptions
    ai_engine/ - All AI components
    services/ - Business logic services
    models/ - Data models
    utils/ - Utilities

Author: AI MediScan Pro Team
Version: 4.0.0
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

# ============================================================================
# IMPORTS - Core Configuration
# ============================================================================
from core.config import settings
from core.exceptions import MediScanException
from utils.logging_config import setup_logging

# ============================================================================
# IMPORTS - AI Engine Components
# ============================================================================
from ai_engine.knowledge_graph import MedicalKnowledgeGraph
from ai_engine.image_classifier import MedicalImageClassifier
from ai_engine.nlp_processor import ClinicalNLPProcessor
from ai_engine.llm_interface import GrokLLMInterface
from ai_engine.diagnostic_engine import HybridDiagnosticEngine

# ============================================================================
# IMPORTS - Services
# ============================================================================
from services.cache_service import CacheService
from services.metrics_service import MetricsService

# ============================================================================
# IMPORTS - API Routes (we'll add these as separate modules)
# ============================================================================
# When you add new features, create new route files and import them here
# Example: from api.v2 import diagnosis, analytics, patients


# ============================================================================
# APPLICATION STATE MANAGEMENT
# ============================================================================

class ApplicationState:
    """
    Centralized application state management.
    
    This makes it easy to add new components without cluttering app.state
    """
    
    def __init__(self):
        # AI Components
        self.knowledge_graph: MedicalKnowledgeGraph = None
        self.image_classifier: MedicalImageClassifier = None
        self.nlp_processor: ClinicalNLPProcessor = None
        self.llm_interface: GrokLLMInterface = None
        self.diagnostic_engine: HybridDiagnosticEngine = None
        
        # Services
        self.cache_service: CacheService = None
        self.metrics_service: MetricsService = None
        
        # Metadata
        self.startup_time: datetime = None
        self.version: str = "4.0.0"
        self.ready: bool = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("="*80)
        logger.info("AI MediScan Pro - Initializing System")
        logger.info("="*80)
        
        self.startup_time = datetime.now()
        
        # Setup logging
        setup_logging()
        
        try:
            # Initialize AI components
            logger.info("Loading medical knowledge base...")
            dataset_paths = self._get_dataset_paths()
            self.knowledge_graph = MedicalKnowledgeGraph(dataset_paths)
            
            logger.info("Loading image classification models...")
            self.image_classifier = MedicalImageClassifier()
            
            logger.info("Initializing clinical NLP processor...")
            self.nlp_processor = ClinicalNLPProcessor()
            
            logger.info("Connecting to LLM interface...")
            self.llm_interface = GrokLLMInterface()
            
            logger.info("Building diagnostic engine...")
            self.diagnostic_engine = HybridDiagnosticEngine(
                self.knowledge_graph,
                self.image_classifier,
                self.nlp_processor,
                self.llm_interface
            )
            
            # Initialize services
            logger.info("Starting cache service...")
            self.cache_service = CacheService()
            
            logger.info("Starting metrics service...")
            self.metrics_service = MetricsService()
            
            # Log system statistics
            stats = self.knowledge_graph.get_statistics()
            logger.info("="*80)
            logger.info("System Statistics:")
            logger.info(f"  Knowledge Base: {stats['total_diseases']} diseases")
            logger.info(f"  Symptoms Indexed: {stats['total_symptoms']}")
            logger.info(f"  Critical Diseases: {stats['critical_diseases']}")
            logger.info(f"  Device: {settings.torch_device}")
            logger.info(f"  Cache: {'Enabled' if self.cache_service.enabled else 'Disabled'}")
            logger.info(f"  Metrics: {'Enabled' if self.metrics_service.enabled else 'Disabled'}")
            logger.info("="*80)
            
            self.ready = True
            logger.info("✓ AI MediScan Pro - Ready for Medical Diagnoses")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}", exc_info=True)
            self.ready = False
            raise
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Shutting down AI MediScan Pro...")
        
        try:
            if self.llm_interface:
                await self.llm_interface.close()
            
            if self.cache_service:
                await self.cache_service.close()
            
            logger.info("✓ Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def _get_dataset_paths(self):
        """Get all available dataset paths"""
        dataset_dir = settings.datasets_dir
        paths = []
        
        # Look for common dataset files
        for pattern in ['*.csv', '*.json', '*.xlsx']:
            paths.extend([str(p) for p in dataset_dir.glob(pattern)])
        
        if not paths:
            logger.warning(f"No datasets found in {dataset_dir}")
        
        return paths
    
    def get_health_status(self):
        """Get current system health status"""
        if not self.ready:
            return {
                "status": "initializing",
                "ready": False
            }
        
        kb_stats = self.knowledge_graph.get_statistics()
        
        return {
            "status": "healthy",
            "ready": True,
            "version": self.version,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
            "components": {
                "knowledge_base": {
                    "diseases": kb_stats['total_diseases'],
                    "symptoms": kb_stats['total_symptoms'],
                    "status": "operational"
                },
                "image_classifier": {
                    "models": len(self.image_classifier.models),
                    "device": str(settings.torch_device),
                    "status": "operational"
                },
                "nlp_processor": {
                    "patterns": len(self.nlp_processor.symptom_patterns),
                    "status": "operational"
                },
                "llm_interface": {
                    "model": settings.grok_model,
                    "status": "operational"
                },
                "cache": {
                    "enabled": self.cache_service.enabled,
                    "status": "operational" if self.cache_service.enabled else "disabled"
                },
                "metrics": {
                    "enabled": self.metrics_service.enabled,
                    "status": "operational" if self.metrics_service.enabled else "disabled"
                }
            }
        }


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

# Global application state
app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Handles startup and shutdown.
    """
    # Startup
    await app_state.initialize()
    
    # Make state available to app
    app.state.mediscan = app_state
    
    yield
    
    # Shutdown
    await app_state.shutdown()


# ============================================================================
# CREATE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AI MediScan Pro",
    description=(
        "Production-Grade Multi-Modal Medical AI System for Early Disease Detection\n\n"
        "Features:\n"
        "- Medical Knowledge Graph with 50+ diseases\n"
        "- RAG-enhanced diagnosis (prevents hallucination)\n"
        "- Multi-modal analysis (text + medical images)\n"
        "- Deep learning image classifiers\n"
        "- Clinical NLP processing\n"
        "- Real-time risk assessment\n"
        "- ICD-10 validated diagnoses"
    ),
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS - Allow all origins for hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response


# ============================================================================
# EXCEPTION HANDLERS
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
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    # Record error metric
    if app_state.ready and app_state.metrics_service:
        app_state.metrics_service.record_error(type(exc).__name__)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# API ROUTES - Core Endpoints
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - API information
    
    Returns basic information about the API and available endpoints.
    """
    return {
        "service": "AI MediScan Pro",
        "version": app_state.version,
        "status": "operational" if app_state.ready else "initializing",
        "description": "Production-grade multi-modal medical AI system",
        "documentation": {
            "interactive": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "diagnose": f"{settings.api_v2_prefix}/diagnose",
            "analyze_image": f"{settings.api_v2_prefix}/analyze-image",
            "extract_symptoms": f"{settings.api_v2_prefix}/extract-symptoms",
            "search_diseases": f"{settings.api_v2_prefix}/search-diseases",
            "disease_info": f"{settings.api_v2_prefix}/disease/{{disease_name}}",
            "stats": f"{settings.api_v2_prefix}/stats"
        },
        "features": [
            "Multi-modal diagnosis (text + imaging)",
            "RAG-enhanced knowledge graph",
            "Clinical NLP processing",
            "Deep learning image analysis",
            "LLM medical reasoning",
            "Real-time risk prediction",
            "ICD-10 validated diagnoses"
        ]
    }


@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns detailed system health status for monitoring.
    """
    return app_state.get_health_status()


@app.get(f"{settings.api_v2_prefix}/stats", tags=["System"])
async def system_statistics():
    """
    Get comprehensive system statistics
    
    Returns detailed information about knowledge base, models, and configuration.
    """
    if not app_state.ready:
        return {"error": "System not ready"}
    
    kb_stats = app_state.knowledge_graph.get_statistics()
    model_info = app_state.image_classifier.get_model_info()
    
    return {
        "knowledge_base": kb_stats,
        "models": model_info,
        "configuration": {
            "device": str(settings.torch_device),
            "cache_enabled": app_state.cache_service.enabled,
            "metrics_enabled": app_state.metrics_service.enabled,
            "api_version": app_state.version,
            "grok_model": settings.grok_model
        },
        "capabilities": [
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
# IMPORT AND REGISTER ROUTE MODULES
# ============================================================================

# This is where you'll add new feature routes
# Create separate files in api/v2/ directory for different features

# Example structure:
# from api.v2.diagnosis import router as diagnosis_router
# from api.v2.analytics import router as analytics_router
# from api.v2.patients import router as patients_router
# 
# app.include_router(diagnosis_router, prefix=settings.api_v2_prefix, tags=["Diagnosis"])
# app.include_router(analytics_router, prefix=settings.api_v2_prefix, tags=["Analytics"])
# app.include_router(patients_router, prefix=settings.api_v2_prefix, tags=["Patients"])

# For now, we'll keep the main diagnosis routes here
# You can move them to separate files later

from api.v2 import diagnosis_routes, knowledge_routes

app.include_router(
    diagnosis_routes.router,
    prefix=settings.api_v2_prefix,
    tags=["Diagnosis"]
)

app.include_router(
    knowledge_routes.router,
    prefix=settings.api_v2_prefix,
    tags=["Knowledge Base"]
)


# ============================================================================
# STARTUP EVENT (Additional initialization if needed)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("FastAPI application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown tasks"""
    logger.info("FastAPI application stopped")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )