# ============================================================================
# FILE: core/config.py
# Centralized configuration management with validation
# ============================================================================

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
from pathlib import Path
import secrets


class Settings(BaseSettings):
    """
    Centralized application settings with validation and type checking.
    
    Features:
    - Environment variable overrides via .env
    - Auto-directory creation
    - Default secure values for secrets
    - Supports multiple environments (dev/staging/prod)
    """

    # ================================
    # API Keys
    # ================================
    grok_api_key: str = Field(..., env="GROK_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # ================================
    # Environment & Logging
    # ================================
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ================================
    # Security
    # ================================
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    api_key_hash: Optional[str] = None
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    encryption_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    patient_id_salt: str = Field(default_factory=lambda: secrets.token_urlsafe(16))  # For hashing patient IDs

    # ================================
    # Database & Cache
    # ================================
    database_url: str = Field("sqlite+aiosqlite:///./data/mediscan.db")
    redis_url: str = Field("redis://localhost:6379/0")

    # ================================
    # File Paths
    # ================================
    models_dir: Path = Field(Path("./data/models"))
    datasets_dir: Path = Field(Path("./data/datasets"))
    cache_dir: Path = Field(Path("./data/cache"))
    upload_dir: Path = Field(Path("./data/uploads"))

    # ================================
    # API Configuration
    # ================================
    api_v2_prefix: str = "/api/v2"
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 60

    # ================================
    # ML Configuration
    # ================================
    torch_device: str = "cpu"  # Auto-detect CUDA
    model_precision: str = "fp32"
    batch_size: int = 8
    max_workers: int = 4

    # ================================
    # Monitoring & Metrics
    # ================================
    enable_metrics: bool = True
    sentry_dsn: Optional[str] = None
    prometheus_port: int = 9090

    # ================================
    # Rate Limiting
    # ================================
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ================================
    # Grok API Configuration
    # ================================
    grok_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    grok_model: str = "grok-4-fast"
    grok_temperature: float = 0.3
    grok_max_tokens: int = 2000

    # ================================
    # Validators
    # ================================
    @validator("torch_device", pre=True, always=True)
    def set_torch_device(cls, v):
        """Auto-detect CUDA availability"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @validator("models_dir", "datasets_dir", "cache_dir", "upload_dir", pre=True, always=True)
    def create_directories(cls, v):
        """Ensure all directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ================================
    # Pydantic Config
    # ================================
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
