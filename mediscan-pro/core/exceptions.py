
# ============================================================================
# FILE: core/exceptions.py
# Custom exception hierarchy for better error handling
# ============================================================================

from typing import Optional, Dict, Any


class MediScanException(Exception):
    """Base exception for all MediScan errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MediScanException):
    """Input validation failed"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=400, details=details)


class ModelError(MediScanException):
    """AI model processing error"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=500, details=details)


class APIError(MediScanException):
    """External API call failed"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=503, details=details)


class DataNotFoundError(MediScanException):
    """Requested data not found"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=404, details=details)


class AuthenticationError(MediScanException):
    """Authentication failed"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)


class RateLimitError(MediScanException):
    """Rate limit exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


