

# ============================================================================
# FILE: services/metrics_service.py
# Prometheus Metrics for Monitoring
# ============================================================================

"""
Metrics service for production monitoring.
Tracks: latency, errors, diagnosis counts, confidence scores.
"""

from prometheus_client import Counter, Histogram, Gauge
from loguru import logger

from core.config import settings


# Define metrics
diagnosis_counter = Counter(
    'mediscan_diagnoses_total',
    'Total number of diagnoses performed',
    ['risk_level']
)

diagnosis_latency = Histogram(
    'mediscan_diagnosis_duration_seconds',
    'Time spent processing diagnosis'
)

confidence_score_gauge = Gauge(
    'mediscan_confidence_score',
    'Latest diagnosis confidence score'
)

api_errors = Counter(
    'mediscan_api_errors_total',
    'Total API errors',
    ['error_type']
)

image_analysis_counter = Counter(
    'mediscan_image_analyses_total',
    'Total image analyses performed',
    ['image_type']
)


class MetricsService:
    """
    Metrics collection service.
    
    Tracks system performance and usage for monitoring.
    """
    
    def __init__(self):
        self.enabled = settings.enable_metrics
        if self.enabled:
            logger.info("Metrics service enabled")
    
    def record_diagnosis(self, diagnosis: dict, duration_seconds: float):
        """Record diagnosis metrics"""
        if not self.enabled:
            return
        
        try:
            # Increment diagnosis counter
            risk_level = diagnosis.get('risk_level', 'UNKNOWN')
            diagnosis_counter.labels(risk_level=risk_level).inc()
            
            # Record latency
            diagnosis_latency.observe(duration_seconds)
            
            # Record confidence score
            confidence = diagnosis.get('confidence_score', 0)
            confidence_score_gauge.set(confidence)
            
        except Exception as e:
            logger.error(f"Metrics recording error: {str(e)}")
    
    def record_image_analysis(self, image_type: str):
        """Record image analysis"""
        if not self.enabled:
            return
        
        try:
            image_analysis_counter.labels(image_type=image_type).inc()
        except Exception as e:
            logger.error(f"Metrics recording error: {str(e)}")
    
    def record_error(self, error_type: str):
        """Record API error"""
        if not self.enabled:
            return
        
        try:
            api_errors.labels(error_type=error_type).inc()
        except Exception as e:
            logger.error(f"Metrics recording error: {str(e)}")
