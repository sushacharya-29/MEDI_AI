
# ============================================================================
# FILE: services/cache_service.py
# Redis Caching for Performance Optimization
# ============================================================================

"""
Caching layer to reduce API calls and speed up repeated diagnoses.
Critical for cost control and performance during demo.
"""

import json
import hashlib
from typing import Optional, Any
from loguru import logger
import redis.asyncio as redis

from core.config import settings


class CacheService:
    """
    Redis-based caching service.
    
    Caches:
    - Knowledge graph search results
    - Image analysis results
    - LLM responses (with same symptoms)
    """
    
    def __init__(self):
        self.redis_client = None
        self.enabled = True
        self._initialize()
    
    def _initialize(self):
        """Initialize Redis connection"""
        try:
            # Parse Redis URL
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Cache service initialized (Redis)")
        except Exception as e:
            logger.warning(f"Redis not available: {str(e)}. Caching disabled.")
            self.enabled = False
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        # Convert data to string and hash
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{data_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        expire_seconds: int = 3600
    ) -> bool:
        """Set value in cache with expiration"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            value_str = json.dumps(value)
            await self.redis_client.setex(key, expire_seconds, value_str)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def get_diagnosis(self, symptoms: str, patient_data: dict) -> Optional[dict]:
        """Get cached diagnosis"""
        cache_key = self._generate_cache_key("diagnosis", {
            "symptoms": symptoms,
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender")
        })
        return await self.get(cache_key)
    
    async def cache_diagnosis(
        self,
        symptoms: str,
        patient_data: dict,
        diagnosis: dict
    ) -> bool:
        """Cache diagnosis result"""
        cache_key = self._generate_cache_key("diagnosis", {
            "symptoms": symptoms,
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender")
        })
        # Cache for 1 hour
        return await self.set(cache_key, diagnosis, expire_seconds=3600)
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
