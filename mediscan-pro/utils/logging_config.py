
# ============================================================================
# FILE: utils/logging_config.py
# Structured logging configuration
# ============================================================================

from loguru import logger
import sys
from pathlib import Path
from core.config import settings


def setup_logging():
    """
    Configure structured logging for the application.
    
    Features:
    - Console output with colors
    - File rotation (10 MB per file)
    - JSON formatting for production
    - Different log levels per environment
    """
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "mediscan_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO"
    )
    
    # Error file (separate for easy debugging)
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="90 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}"
    )
    
    logger.info("Logging configured successfully")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log level: {settings.log_level}")