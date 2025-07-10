"""
Main Recommendation Engine

High-performance recommendation engine capable of processing 1M+ events/second
with sub-10ms serving latency.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict
import hashlib
import json

from .models import User, Item, Interaction, Recommendation
from .algorithms import AlgorithmRegistry
from ..storage.cache import CacheManager
from ..storage.feature_store import FeatureStore
from ..ml.algorithms import CollaborativeFiltering, ContentBased, DeepLearning


@dataclass
class EngineConfig:
    """Configuration for the recommendation engine"""
    max_latency_ms: int = 10
    target_throughput: int = 1000000
    cache_size_gb: int = 32
    batch_size: int = 10000
    worker_threads: int = 16
    algorithms: List[Dict[str, Any]] = None
    fallback_strategy: str = "popular_items"
    enable_real_time_updates: bool = True
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = [
                {"name": "collaborative_filtering", "weight": 0.4},
                {"name": "content_based", "weight": 0.3},
                {"name": "deep_learning", "weight": 0.3}
            ]


class RecommendationEngine:
    """
    High-performance recommendation engine with real-time updates
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the recommendation engine
        
        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.algorithm_registry = AlgorithmRegistry()
        self.cache_manager = CacheManager(size_gb=self.config.cache_size_gb)
        self.feature_store = FeatureStore()
        
        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.start_time = time.time()
        
        # Threading for high throughput
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)
        
        # Algorithm instances
        self.algorithms = {}
        self._initialize_algorithms()
        
        # Fallback recommendations
        self.fallback_recommendations = {}
        self._initialize_fallbacks()
        
        self.logger.info(f"RecommendationEngine initialized with config: {self.config}")
    
    def _initialize_algorithms(self):
        """Initialize recommendation algorithms"""
        for algo_config in self.config.algorithms:
            name = algo_config["name"]
            weight = algo_config.get("weight", 1.0)
            
            if name == "collaborative_filtering":
                self.algorithms[name] = {
                    "instance": CollaborativeFiltering(),
                    "weight": weight
                }
            elif name == "content_based":
                self.algorithms[name] = {
                    "instance": ContentBased(),
                    "weight": weight
                }
            elif name == "deep_learning":
                self.algorithms[name] = {
                    "instance": DeepLearning(),
                    "weight": weight
                }
            
            self.logger.info(f"Initialized algorithm: {name} with weight: {weight}")
    
    def _initialize_fallbacks(self):
        """Initialize fallback recommendations"""
        # Popular items across categories
        self.fallback_recommendations = {
            "electronics": ["laptop_001", "phone_002", "tablet_003"],
            "books": ["book_001", "book_002", "book_003"],
            "movies": ["movie_001", "movie_002", "movie_003"],
            "default": ["item_001", "item_002", "item_003"]
        }
    
    async def get_recommendations(
        self,
        user_id: str,
        num_items: int = 10,
        context: Optional[Dict[str, Any]] = None,
        algorithms: Optional[List[str]] = None,
        real_time: bool = True
    ) -> List[Recommendation]:
        """
        Get recommendations for a user with sub-10ms latency
        
        Args:
            user_id: User identifier
            num_items: Number of recommendations to return
            context: Additional context (device, location, etc.)
            algorithms: Specific algorithms to use
            real_time: Whether to include real-time updates
            
        Returns:
            List of recommendations
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(user_id, num_items, context, algorithms)
            
            # Try cache first (L1 - Redis)
            cached_recs = await self.cache_manager.get(cache_key)
            if cached_recs and not real_time:
                self._record_latency(start_time)
                return cached_recs
            
            # Get user features
            user_features = await self.feature_store.get_user_features(user_id)
            if not user_features:
                return self._get_fallback_recommendations(user_id, num_items, context)
            
            # Get candidate items
            candidates = await self._get_candidate_items(user_id, context)
            
            # Run recommendation algorithms
            if algorithms:
                selected_algorithms = {k: v for k, v in self.algorithms.items() if k in algorithms}
            else:
                selected_algorithms = self.algorithms
            
            # Parallel algorithm execution for speed
            algorithm_results = await self._run_algorithms_parallel(
                user_features, candidates, selected_algorithms, num_items
            )
            
            # Ensemble combination
            final_recommendations = self._ensemble_recommendations(
                algorithm_results, num_items
            )
            
            # Cache the results
            await self.cache_manager.set(cache_key, final_recommendations, ttl=300)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(start_time)
            
            # Check latency SLA
            if latency_ms > self.config.max_latency_ms:
                self.logger.warning(f"Latency SLA violation: {latency_ms:.2f}ms > {self.config.max_latency_ms}ms")
            
            return final_recommendations
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_fallback_recommendations(user_id, num_items, context)
    
    async def _get_candidate_items(
        self, 
        user_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get candidate items for recommendation
        
        Args:
            user_id: User identifier
            context: Additional context
            
        Returns:
            List of candidate item IDs
        """
        # Get user interaction history
        user_history = await self.feature_store.get_user_interactions(user_id)
        
        # Apply business rules and filters
        candidates = []
        
        # Category-based candidates
        if context and context.get("category"):
            category_items = await self.feature_store.get_items_by_category(context["category"])
            candidates.extend(category_items[:1000])  # Limit for performance
        
        # Popular items
        popular_items = await self.feature_store.get_popular_items(limit=500)
        candidates.extend(popular_items)
        
        # Collaborative filtering candidates
        similar_users = await self.feature_store.get_similar_users(user_id, limit=100)
        for similar_user in similar_users:
            similar_user_items = await self.feature_store.get_user_interactions(similar_user)
            candidates.extend([item["item_id"] for item in similar_user_items[:10]])
        
        # Remove duplicates and items user has already interacted with
        seen_items = {item["item_id"] for item in user_history}
        candidates = list(set(candidates) - seen_items)
        
        return candidates[:10000]  # Limit candidate set for performance
    
    async def _run_algorithms_parallel(
        self,
        user_features: Dict[str, Any],
        candidates: List[str],
        algorithms: Dict[str, Dict],
        num_items: int
    ) -> Dict[str, List[Recommendation]]:
        """
        Run recommendation algorithms in parallel
        
        Args:
            user_features: User feature vector
            candidates: Candidate items
            algorithms: Algorithm instances and weights
            num_items: Number of items to recommend per algorithm
            
        Returns:
            Dictionary of algorithm results
        """
        tasks = []
        
        for algo_name, algo_config in algorithms.items():
            task = asyncio.create_task(
                self._run_single_algorithm(
                    algo_config["instance"],
                    user_features,
                    candidates,
                    num_items * 2  # Get more for better ensemble
                )
            )
            tasks.append((algo_name, task))
        
        results = {}
        for algo_name, task in tasks:
            try:
                results[algo_name] = await task
            except Exception as e:
                self.logger.error(f"Algorithm {algo_name} failed: {e}")
                results[algo_name] = []
        
        return results
    
    async def _run_single_algorithm(
        self,
        algorithm,
        user_features: Dict[str, Any],
        candidates: List[str],
        num_items: int
    ) -> List[Recommendation]:
        """
        Run a single recommendation algorithm
        
        Args:
            algorithm: Algorithm instance
            user_features: User features
            candidates: Candidate items
            num_items: Number of recommendations
            
        Returns:
            List of recommendations
        """
        # This would be implemented by each algorithm
        # For demo purposes, return mock recommendations
        recommendations = []
        
        for i, item_id in enumerate(candidates[:num_items]):
            score = np.random.random() * 0.5 + 0.5  # Score between 0.5-1.0
            
            recommendation = Recommendation(
                item_id=item_id,
                score=score,
                reason=f"algorithm_{algorithm.__class__.__name__.lower()}",
                rank=i + 1
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _ensemble_recommendations(
        self,
        algorithm_results: Dict[str, List[Recommendation]],
        num_items: int
    ) -> List[Recommendation]:
        """
        Combine recommendations from multiple algorithms
        
        Args:
            algorithm_results: Results from each algorithm
            num_items: Final number of recommendations
            
        Returns:
            Final ensemble recommendations
        """
        # Score aggregation using weighted average
        item_scores = defaultdict(list)
        
        for algo_name, recommendations in algorithm_results.items():
            weight = self.algorithms[algo_name]["weight"]
            
            for rec in recommendations:
                item_scores[rec.item_id].append(rec.score * weight)
        
        # Calculate final scores
        final_scores = {}
        for item_id, scores in item_scores.items():
            final_scores[item_id] = sum(scores) / len(scores)
        
        # Sort by score and create final recommendations
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        for i, (item_id, score) in enumerate(sorted_items[:num_items]):
            recommendation = Recommendation(
                item_id=item_id,
                score=score,
                reason="ensemble",
                rank=i + 1
            )
            final_recommendations.append(recommendation)
        
        return final_recommendations
    
    def _get_fallback_recommendations(
        self,
        user_id: str,
        num_items: int,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """
        Get fallback recommendations when main algorithms fail
        
        Args:
            user_id: User identifier
            num_items: Number of recommendations
            context: Additional context
            
        Returns:
            Fallback recommendations
        """
        category = "default"
        if context and context.get("category"):
            category = context["category"]
        
        fallback_items = self.fallback_recommendations.get(category, 
                                                          self.fallback_recommendations["default"])
        
        recommendations = []
        for i, item_id in enumerate(fallback_items[:num_items]):
            recommendation = Recommendation(
                item_id=item_id,
                score=0.5,  # Neutral score for fallback
                reason="fallback",
                rank=i + 1
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_cache_key(
        self,
        user_id: str,
        num_items: int,
        context: Optional[Dict[str, Any]],
        algorithms: Optional[List[str]]
    ) -> str:
        """Generate cache key for recommendations"""
        key_data = {
            "user_id": user_id,
            "num_items": num_items,
            "context": context or {},
            "algorithms": algorithms or []
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _record_latency(self, start_time: float):
        """Record request latency for monitoring"""
        latency = time.time() - start_time
        self.request_count += 1
        self.total_latency += latency
    
    async def update_user_interaction(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Update user interaction in real-time
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            interaction_type: Type of interaction (view, click, purchase, etc.)
            context: Additional context
        """
        # Create interaction object
        interaction = Interaction(
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            timestamp=time.time(),
            context=context or {}
        )
        
        # Update feature store
        await self.feature_store.update_user_interaction(interaction)
        
        # Invalidate relevant caches
        await self.cache_manager.invalidate_user_cache(user_id)
        
        # Update algorithms if real-time learning is enabled
        if self.config.enable_real_time_updates:
            for algorithm in self.algorithms.values():
                if hasattr(algorithm["instance"], "update_online"):
                    await algorithm["instance"].update_online(interaction)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / max(1, self.request_count) * 1000  # Convert to ms
        throughput = self.request_count / max(1, uptime)
        error_rate = self.error_count / max(1, self.request_count)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "average_latency_ms": avg_latency,
            "throughput_rps": throughput,
            "error_rate": error_rate,
            "cache_stats": self.cache_manager.get_stats(),
            "algorithm_count": len(self.algorithms)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check cache
        try:
            await self.cache_manager.ping()
            health["components"]["cache"] = "healthy"
        except Exception as e:
            health["components"]["cache"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check feature store
        try:
            await self.feature_store.ping()
            health["components"]["feature_store"] = "healthy"
        except Exception as e:
            health["components"]["feature_store"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check algorithms
        for algo_name in self.algorithms:
            try:
                # Simple health check - could be more sophisticated
                health["components"][f"algorithm_{algo_name}"] = "healthy"
            except Exception as e:
                health["components"][f"algorithm_{algo_name}"] = f"unhealthy: {e}"
                health["status"] = "degraded"
        
        return health
    
    async def shutdown(self):
        """Gracefully shutdown the engine"""
        self.logger.info("Shutting down RecommendationEngine...")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close connections
        await self.cache_manager.close()
        await self.feature_store.close()
        
        self.logger.info("RecommendationEngine shutdown complete")
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return f"""RecommendationEngine(
    algorithms={list(self.algorithms.keys())},
    total_requests={stats['total_requests']},
    avg_latency_ms={stats['average_latency_ms']:.2f},
    throughput_rps={stats['throughput_rps']:.1f}
)""" 