"""
Real-time Feature Store

High-performance feature serving for recommendation systems.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from ..core.models import Interaction, FeatureVector


class FeatureStore:
    """
    Real-time feature store for serving user and item features
    with sub-millisecond latency.
    """
    
    def __init__(self):
        """Initialize the feature store"""
        
        # In-memory feature storage for ultra-fast access
        self.user_features = {}
        self.item_features = {}
        self.user_interactions = defaultdict(list)
        self.item_interactions = defaultdict(list)
        
        # Precomputed aggregations
        self.user_stats = {}
        self.item_stats = {}
        self.popular_items = []
        self.user_similarities = {}
        
        # Category mappings
        self.category_items = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
        
    async def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user features with sub-millisecond latency
        
        Args:
            user_id: User identifier
            
        Returns:
            User feature dictionary
        """
        start_time = time.time()
        
        # Check cache first
        if user_id in self.user_features:
            features = self.user_features[user_id].copy()
            
            # Add real-time computed features
            features.update(await self._compute_realtime_user_features(user_id))
            
            latency_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"User features retrieved in {latency_ms:.3f}ms")
            
            return features
        
        # Cold start - generate basic features
        return await self._generate_cold_start_user_features(user_id)
    
    async def get_item_features(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get item features with sub-millisecond latency
        
        Args:
            item_id: Item identifier
            
        Returns:
            Item feature dictionary
        """
        if item_id in self.item_features:
            features = self.item_features[item_id].copy()
            
            # Add real-time computed features
            features.update(await self._compute_realtime_item_features(item_id))
            
            return features
        
        return None
    
    async def get_user_interactions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent user interactions
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions
            
        Returns:
            List of interaction dictionaries
        """
        interactions = self.user_interactions.get(user_id, [])
        
        # Sort by timestamp (most recent first) and limit
        sorted_interactions = sorted(
            interactions, 
            key=lambda x: x.get('timestamp', 0), 
            reverse=True
        )
        
        return sorted_interactions[:limit]
    
    async def get_items_by_category(self, category: str, limit: int = 1000) -> List[str]:
        """
        Get items by category
        
        Args:
            category: Item category
            limit: Maximum number of items
            
        Returns:
            List of item IDs
        """
        items = self.category_items.get(category, [])
        return items[:limit]
    
    async def get_popular_items(self, limit: int = 500) -> List[str]:
        """
        Get popular items
        
        Args:
            limit: Maximum number of items
            
        Returns:
            List of popular item IDs
        """
        return self.popular_items[:limit]
    
    async def get_similar_users(self, user_id: str, limit: int = 100) -> List[str]:
        """
        Get similar users
        
        Args:
            user_id: User identifier
            limit: Maximum number of similar users
            
        Returns:
            List of similar user IDs
        """
        similarities = self.user_similarities.get(user_id, [])
        return [user for user, _ in similarities[:limit]]
    
    async def update_user_interaction(self, interaction: Interaction):
        """
        Update user interaction in real-time
        
        Args:
            interaction: User interaction
        """
        user_id = interaction.user_id
        item_id = interaction.item_id
        
        # Add to user interactions
        interaction_dict = interaction.to_dict()
        self.user_interactions[user_id].append(interaction_dict)
        
        # Add to item interactions
        self.item_interactions[item_id].append(interaction_dict)
        
        # Keep only recent interactions (memory optimization)
        max_interactions = 1000
        if len(self.user_interactions[user_id]) > max_interactions:
            self.user_interactions[user_id] = self.user_interactions[user_id][-max_interactions:]
        
        if len(self.item_interactions[item_id]) > max_interactions:
            self.item_interactions[item_id] = self.item_interactions[item_id][-max_interactions:]
        
        # Update real-time stats
        await self._update_user_stats(user_id)
        await self._update_item_stats(item_id)
        
        self.logger.debug(f"Updated interaction for user {user_id}, item {item_id}")
    
    async def _compute_realtime_user_features(self, user_id: str) -> Dict[str, Any]:
        """Compute real-time user features"""
        features = {}
        
        # Recent interaction count
        recent_interactions = await self.get_user_interactions(user_id, limit=50)
        features['recent_interaction_count'] = len(recent_interactions)
        
        # Activity recency
        if recent_interactions:
            last_interaction_time = max(
                interaction.get('timestamp', 0) 
                for interaction in recent_interactions
            )
            features['hours_since_last_activity'] = (
                time.time() - last_interaction_time
            ) / 3600
        else:
            features['hours_since_last_activity'] = 999999
        
        # Interaction diversity
        if recent_interactions:
            interaction_types = set(
                interaction.get('interaction_type', 'view')
                for interaction in recent_interactions
            )
            features['interaction_type_diversity'] = len(interaction_types)
        else:
            features['interaction_type_diversity'] = 0
        
        return features
    
    async def _compute_realtime_item_features(self, item_id: str) -> Dict[str, Any]:
        """Compute real-time item features"""
        features = {}
        
        # Recent interaction count
        recent_interactions = self.item_interactions.get(item_id, [])
        recent_interactions = [
            interaction for interaction in recent_interactions
            if time.time() - interaction.get('timestamp', 0) < 86400  # Last 24 hours
        ]
        
        features['recent_interaction_count'] = len(recent_interactions)
        
        # Engagement rate
        if recent_interactions:
            high_engagement_interactions = [
                interaction for interaction in recent_interactions
                if interaction.get('interaction_type') in ['purchase', 'like', 'share']
            ]
            features['engagement_rate'] = len(high_engagement_interactions) / len(recent_interactions)
        else:
            features['engagement_rate'] = 0.0
        
        return features
    
    async def _generate_cold_start_user_features(self, user_id: str) -> Dict[str, Any]:
        """Generate features for cold start users"""
        return {
            'is_new_user': True,
            'recent_interaction_count': 0,
            'hours_since_last_activity': 999999,
            'interaction_type_diversity': 0,
            'estimated_preferences': {}
        }
    
    async def _update_user_stats(self, user_id: str):
        """Update user statistics"""
        interactions = await self.get_user_interactions(user_id)
        
        if not interactions:
            return
        
        stats = {
            'total_interactions': len(interactions),
            'avg_session_length': 0,
            'favorite_categories': [],
            'purchase_frequency': 0
        }
        
        # Calculate favorite categories
        category_counts = defaultdict(int)
        purchase_count = 0
        
        for interaction in interactions:
            # Mock category extraction
            category = 'electronics'  # Would come from item metadata
            category_counts[category] += 1
            
            if interaction.get('interaction_type') == 'purchase':
                purchase_count += 1
        
        stats['favorite_categories'] = [
            category for category, _ in 
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        
        stats['purchase_frequency'] = purchase_count / len(interactions)
        
        self.user_stats[user_id] = stats
    
    async def _update_item_stats(self, item_id: str):
        """Update item statistics"""
        interactions = self.item_interactions.get(item_id, [])
        
        if not interactions:
            return
        
        stats = {
            'total_interactions': len(interactions),
            'unique_users': len(set(i.get('user_id') for i in interactions)),
            'popularity_score': 0,
            'engagement_score': 0
        }
        
        # Calculate popularity and engagement scores
        if interactions:
            recent_interactions = [
                i for i in interactions
                if time.time() - i.get('timestamp', 0) < 86400  # Last 24 hours
            ]
            
            stats['popularity_score'] = len(recent_interactions)
            
            high_engagement_count = sum(
                1 for i in recent_interactions
                if i.get('interaction_type') in ['purchase', 'like', 'share']
            )
            
            stats['engagement_score'] = (
                high_engagement_count / max(1, len(recent_interactions))
            )
        
        self.item_stats[item_id] = stats
        
        # Update popular items list
        await self._update_popular_items()
    
    async def _update_popular_items(self):
        """Update popular items ranking"""
        # Sort items by popularity score
        item_scores = []
        
        for item_id, stats in self.item_stats.items():
            score = stats.get('popularity_score', 0) * stats.get('engagement_score', 0)
            item_scores.append((item_id, score))
        
        # Sort by score and update popular items list
        item_scores.sort(key=lambda x: x[1], reverse=True)
        self.popular_items = [item_id for item_id, _ in item_scores[:1000]]
    
    async def bulk_load_features(
        self, 
        user_features: Dict[str, Dict], 
        item_features: Dict[str, Dict]
    ):
        """
        Bulk load features for initialization
        
        Args:
            user_features: Dictionary of user ID to features
            item_features: Dictionary of item ID to features
        """
        self.user_features.update(user_features)
        self.item_features.update(item_features)
        
        # Build category mappings
        for item_id, features in item_features.items():
            category = features.get('category', 'default')
            self.category_items[category].append(item_id)
        
        self.logger.info(f"Bulk loaded {len(user_features)} user features and {len(item_features)} item features")
    
    async def ping(self) -> bool:
        """Health check for feature store"""
        try:
            # Simple health check
            return True
        except Exception as e:
            self.logger.error(f"Feature store ping failed: {e}")
            return False
    
    async def close(self):
        """Close feature store connections"""
        self.logger.info("Closing feature store...")
        # Cleanup if needed
        pass 