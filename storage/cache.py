"""
High-Performance Caching System

Multi-tier caching for sub-10ms recommendation serving latency.
"""

import asyncio
import time
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import hashlib


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.total_requests)


class CacheManager:
    """
    Multi-tier cache manager for ultra-low latency serving
    
    L1: In-memory Python dict (sub-1ms)
    L2: Redis (1-5ms)
    L3: Persistent storage (5-10ms)
    """
    
    def __init__(self, size_gb: int = 32):
        """
        Initialize cache manager
        
        Args:
            size_gb: Maximum cache size in GB
        """
        self.size_gb = size_gb
        self.max_items = size_gb * 1024 * 1024 // 1024  # Estimate max items
        
        # L1 Cache - In-memory
        self.l1_cache = {}
        self.l1_access_times = {}
        self.l1_ttl = {}
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Redis connection (mock for demo)
        self.redis_client = None
        
        self.logger = logging.getLogger(__name__)
        
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with multi-tier lookup
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        start_time = time.time()
        
        try:
            # L1 Cache lookup (fastest)
            if key in self.l1_cache:
                # Check TTL
                if key in self.l1_ttl and time.time() > self.l1_ttl[key]:
                    del self.l1_cache[key]
                    del self.l1_ttl[key]
                    if key in self.l1_access_times:
                        del self.l1_access_times[key]
                else:
                    # Update access time for LRU
                    self.l1_access_times[key] = time.time()
                    
                    self.stats.hits += 1
                    self.stats.total_requests += 1
                    
                    latency_ms = (time.time() - start_time) * 1000
                    self._update_avg_latency(latency_ms)
                    
                    return self.l1_cache[key]
            
            # L2 Cache lookup (Redis) - mock implementation
            l2_value = await self._get_from_redis(key)
            if l2_value is not None:
                # Store in L1 for next time
                await self.set(key, l2_value, ttl=300, tier='l1')
                
                self.stats.hits += 1
                self.stats.total_requests += 1
                
                latency_ms = (time.time() - start_time) * 1000
                self._update_avg_latency(latency_ms)
                
                return l2_value
            
            # Cache miss
            self.stats.misses += 1
            self.stats.total_requests += 1
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            self.stats.misses += 1
            self.stats.total_requests += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 300,
        tier: str = 'all'
    ):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tier: Cache tier ('l1', 'l2', 'all')
        """
        try:
            if tier in ['l1', 'all']:
                # L1 Cache
                self._evict_if_needed()
                
                self.l1_cache[key] = value
                self.l1_access_times[key] = time.time()
                
                if ttl > 0:
                    self.l1_ttl[key] = time.time() + ttl
            
            if tier in ['l2', 'all']:
                # L2 Cache (Redis)
                await self._set_to_redis(key, value, ttl)
                
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete key from all cache tiers"""
        try:
            # Remove from L1
            if key in self.l1_cache:
                del self.l1_cache[key]
            if key in self.l1_access_times:
                del self.l1_access_times[key]
            if key in self.l1_ttl:
                del self.l1_ttl[key]
            
            # Remove from L2 (Redis)
            await self._delete_from_redis(key)
            
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user"""
        # Find all keys containing user_id
        keys_to_delete = []
        
        for key in self.l1_cache.keys():
            if user_id in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            await self.delete(key)
        
        self.logger.debug(f"Invalidated {len(keys_to_delete)} cache entries for user {user_id}")
    
    def _evict_if_needed(self):
        """Evict old entries using LRU if cache is full"""
        if len(self.l1_cache) >= self.max_items:
            # Find least recently used items
            sorted_items = sorted(
                self.l1_access_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest 10% of items
            num_to_remove = max(1, len(sorted_items) // 10)
            
            for key, _ in sorted_items[:num_to_remove]:
                if key in self.l1_cache:
                    del self.l1_cache[key]
                if key in self.l1_access_times:
                    del self.l1_access_times[key]
                if key in self.l1_ttl:
                    del self.l1_ttl[key]
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis (mock implementation)"""
        # Mock Redis lookup with simulated latency
        await asyncio.sleep(0.002)  # 2ms simulated Redis latency
        return None  # Mock miss
    
    async def _set_to_redis(self, key: str, value: Any, ttl: int):
        """Set value to Redis (mock implementation)"""
        # Mock Redis set with simulated latency
        await asyncio.sleep(0.001)  # 1ms simulated Redis latency
        pass
    
    async def _delete_from_redis(self, key: str):
        """Delete from Redis (mock implementation)"""
        await asyncio.sleep(0.001)  # 1ms simulated Redis latency
        pass
    
    def _update_avg_latency(self, latency_ms: float):
        """Update average latency using exponential moving average"""
        alpha = 0.1  # Smoothing factor
        if self.stats.avg_latency_ms == 0:
            self.stats.avg_latency_ms = latency_ms
        else:
            self.stats.avg_latency_ms = (
                alpha * latency_ms + 
                (1 - alpha) * self.stats.avg_latency_ms
            )
    
    async def ping(self) -> bool:
        """Health check for cache system"""
        try:
            # Test L1 cache
            test_key = f"ping_test_{int(time.time())}"
            test_value = "pong"
            
            await self.set(test_key, test_value, ttl=10)
            result = await self.get(test_key)
            await self.delete(test_key)
            
            return result == test_value
            
        except Exception as e:
            self.logger.error(f"Cache ping failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        # Calculate memory usage
        memory_usage_mb = 0
        for value in self.l1_cache.values():
            try:
                memory_usage_mb += len(pickle.dumps(value)) / (1024 * 1024)
            except:
                memory_usage_mb += 0.001  # Estimate
        
        self.stats.memory_usage_mb = memory_usage_mb
        
        return {
            "hit_rate": self.stats.hit_rate,
            "total_requests": self.stats.total_requests,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "avg_latency_ms": self.stats.avg_latency_ms,
            "memory_usage_mb": memory_usage_mb,
            "l1_cache_size": len(self.l1_cache),
            "max_items": self.max_items
        }
    
    def clear_stats(self):
        """Reset cache statistics"""
        self.stats = CacheStats()
    
    async def close(self):
        """Close cache connections"""
        self.logger.info("Closing cache connections...")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Clear L1 cache
        self.l1_cache.clear()
        self.l1_access_times.clear()
        self.l1_ttl.clear()
        
        self.logger.info("Cache connections closed") 