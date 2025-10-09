
# ============================================================================
# FILE: core/security.py
# Security utilities: hashing, encryption, authentication
# ============================================================================

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
import hashlib
import secrets

from core.config import settings
from core.exceptions import AuthenticationError


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"msp_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_access_token(
    data: Dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Payload to encode in token
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm="HS256"
    )
    
    return encoded_jwt


def verify_token(token: str) -> Dict:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token to verify
    
    Returns:
        Decoded token payload
    
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"]
        )
        return payload
    except JWTError:
        raise AuthenticationError("Invalid or expired token")


def hash_patient_id(patient_id: str) -> str:
    """
    Hash patient ID for privacy (HIPAA compliance)
    
    Args:
        patient_id: Original patient identifier
    
    Returns:
        Hashed patient ID (first 16 chars of SHA-256)
    """
    return hashlib.sha256(patient_id.encode()).hexdigest()[:16]


def encrypt_sensitive_data(data: str) -> str:
    """
    Encrypt sensitive medical data
    
    Note: In production, use proper encryption like Fernet
    """
    from cryptography.fernet import Fernet
    
    # Use encryption key from settings
    key = settings.encryption_key.encode()[:32]  # Ensure 32 bytes
    # Pad or hash to ensure proper length
    key = hashlib.sha256(key).digest()
    key = base64.urlsafe_b64encode(key)
    
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive medical data"""
    from cryptography.fernet import Fernet
    import base64
    
    key = settings.encryption_key.encode()[:32]
    key = hashlib.sha256(key).digest()
    key = base64.urlsafe_b64encode(key)
    
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

