from __future__ import annotations
from typing import TypedDict, Any, Optional
from typing_extensions import NotRequired
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt
import hashlib
import secrets
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.config import settings
from core.exceptions import AuthenticationError

# Configure logging for audit trails (HIPAA compliance)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Password hashing context (Argon2 as default for 2025 standards)
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    deprecated="auto",
    argon2__memory_cost=65536,  # 64MB, tuned for security vs. perf
)

# Type definitions for JWT payload
class JWTPayload(TypedDict):
    sub: str  # Subject (e.g., user ID)
    exp: NotRequired[float]  # Expiration timestamp
    iat: NotRequired[float]  # Issued at timestamp
    scope: NotRequired[str]  # Optional scope

class SecurityManager:
    """Centralized security utilities for hashing, encryption, and authentication."""
    
    def __init__(self):
        self._fernet_cache: dict[str, Fernet] = {}
        self._jwt_secret = settings.jwt_secret
        self._encryption_key = settings.encryption_key
        self._token_blacklist: set[str] = set()  # In-memory for simplicity; use Redis in prod

    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2 (or fallback to bcrypt).
        
        Args:
            password: Plaintext password to hash.
            
        Returns:
            Hashed password string.
        """
        logger.debug("Hashing password")
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plaintext password against its hash.
        
        Args:
            plain_password: Plaintext password to verify.
            hashed_password: Stored hashed password.
            
        Returns:
            True if passwords match, False otherwise.
        """
        try:
            logger.debug("Verifying password")
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.warning(f"Password verification failed: {e}")
            return False

    def generate_api_key(self, prefix: str = "msp_") -> str:
        """Generate a secure API key with configurable prefix.
        
        Args:
            prefix: Custom prefix for API key (default: 'msp_').
            
        Returns:
            Secure API key string.
        """
        logger.debug("Generating API key")
        return f"{prefix}{secrets.token_urlsafe(32)}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage.
        
        Args:
            api_key: API key to hash.
            
        Returns:
            SHA-256 hash of the API key.
        """
        logger.debug("Hashing API key")
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_access_token(
        self,
        data: JWTPayload,
        expires_delta: Optional[timedelta] = None,
        scope: str = "access"
    ) -> str:
        """Create a JWT access token with expiration and scope.
        
        Args:
            data: Payload to encode (e.g., {'sub': user_id}).
            expires_delta: Optional expiration time delta.
            scope: Token scope (e.g., 'access', 'refresh').
            
        Returns:
            Encoded JWT token.
        """
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=24))
        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "scope": scope})
        token = jwt.encode(to_encode, self._jwt_secret, algorithm="HS256")
        logger.info(f"Generated JWT for sub={data.get('sub')}, scope={scope}")
        return token

    def verify_token(self, token: str) -> JWTPayload:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify.
            
        Returns:
            Decoded payload as JWTPayload.
            
        Raises:
            AuthenticationError: If token is invalid, expired, or blacklisted.
        """
        if token in self._token_blacklist:
            logger.warning(f"Attempted use of blacklisted token")
            raise AuthenticationError("Token is blacklisted")
        try:
            payload = jwt.decode(token, self._jwt_secret, algorithms=["HS256"])
            logger.debug(f"Verified JWT for sub={payload.get('sub')}")
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise AuthenticationError("Invalid or expired token")

    def blacklist_token(self, token: str) -> None:
        """Add a token to the blacklist (e.g., for logout).
        
        Args:
            token: JWT token to blacklist.
        """
        self._token_blacklist.add(token)
        logger.info(f"Blacklisted token: {token[:8]}...")

    def hash_patient_id(self, patient_id: str, salt: str = settings.patient_id_salt) -> str:
        """Hash patient ID with salt for HIPAA-compliant anonymization.
        
        Args:
            patient_id: Original patient identifier.
            salt: Salt for hashing (default from settings).
            
        Returns:
            Salted SHA-256 hash (full 64 chars to avoid collisions).
        """
        salted_id = f"{patient_id}:{salt}"
        hash_value = hashlib.sha256(salted_id.encode()).hexdigest()
        logger.debug(f"Hashed patient ID: {hash_value[:8]}...")
        return hash_value

    def _get_fernet(self) -> Fernet:
        """Derive and cache a Fernet instance for encryption."""
        key_hash = hashlib.sha256(self._encryption_key.encode()).digest()
        key = base64.urlsafe_b64encode(key_hash)
        if key not in self._fernet_cache:
            self._fernet_cache[key] = Fernet(key)
            logger.debug("Created and cached Fernet instance")
        return self._fernet_cache[key]

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive medical data using Fernet.
        
        Args:
            data: Plaintext data to encrypt.
            
        Returns:
            Encrypted data as base64 string.
            
        Raises:
            RuntimeError: If encryption fails.
        """
        try:
            fernet = self._get_fernet()
            encrypted = fernet.encrypt(data.encode()).decode()
            logger.debug("Encrypted sensitive data")
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive medical data using Fernet.
        
        Args:
            encrypted_data: Encrypted data to decrypt.
            
        Returns:
            Decrypted plaintext string.
            
        Raises:
            RuntimeError: If decryption fails.
        """
        try:
            fernet = self._get_fernet()
            decrypted = fernet.decrypt(encrypted_data.encode()).decode()
            logger.debug("Decrypted sensitive data")
            return decrypted
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise RuntimeError(f"Decryption failed: {e}")

    def rotate_encryption_key(self, new_key: str) -> None:
        """Rotate encryption key and invalidate cache.
        
        Args:
            new_key: New encryption key.
        """
        self._encryption_key = new_key
        self._fernet_cache.clear()
        logger.info("Rotated encryption key and cleared Fernet cache")

# Singleton instance for use across application
security = SecurityManager()