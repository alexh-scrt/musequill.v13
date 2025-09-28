import logging
import os
import json
from typing import Optional, List, Dict, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisClient:
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:16379")
        self.client: Optional[redis.Redis] = None
        self.default_ttl = int(os.getenv("REDIS_TTL", "86400"))
        
    async def connect(self):
        if not self.client:
            try:
                self.client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.client.ping()
                logger.info(f"✅ Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                raise
    
    async def disconnect(self):
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis")
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        try:
            await self.connect()
            if ttl or self.default_ttl:
                await self.client.setex(key, ttl or self.default_ttl, value)
            else:
                await self.client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        try:
            await self.connect()
            value = await self.client.get(key)
            return value
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        try:
            await self.connect()
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def lpush(self, key: str, value: str) -> bool:
        try:
            await self.connect()
            await self.client.lpush(key, value)
            if self.default_ttl:
                await self.client.expire(key, self.default_ttl)
            return True
        except Exception as e:
            logger.error(f"Error lpush to key {key}: {e}")
            return False
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        try:
            await self.connect()
            values = await self.client.lrange(key, start, end)
            return values or []
        except Exception as e:
            logger.error(f"Error lrange key {key}: {e}")
            return []
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        try:
            await self.connect()
            await self.client.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.error(f"Error ltrim key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            await self.connect()
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False