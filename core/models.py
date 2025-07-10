"""
Data Models for the Real-time Recommendation Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import time
from datetime import datetime
from enum import Enum


class InteractionType(Enum):
    """Types of user interactions"""
    VIEW = "view"
    CLICK = "click"
    PURCHASE = "purchase"
    LIKE = "like"
    SHARE = "share"
    RATE = "rate"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    BOOKMARK = "bookmark"


@dataclass
class User:
    """User data model"""
    user_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    demographics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "features": self.features,
            "demographics": self.demographics,
            "preferences": self.preferences,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class Item:
    """Item data model"""
    item_id: str
    category: str
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    popularity_score: float = 0.0
    quality_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "features": self.features,
            "metadata": self.metadata,
            "popularity_score": self.popularity_score,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class Interaction:
    """User-item interaction model"""
    user_id: str
    item_id: str
    interaction_type: Union[str, InteractionType]
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    rating: Optional[float] = None
    duration: Optional[float] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.interaction_type, str):
            try:
                self.interaction_type = InteractionType(self.interaction_type)
            except ValueError:
                # Keep as string if not in enum
                pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "interaction_type": self.interaction_type.value if isinstance(self.interaction_type, InteractionType) else self.interaction_type,
            "timestamp": self.timestamp,
            "context": self.context,
            "rating": self.rating,
            "duration": self.duration,
            "session_id": self.session_id
        }


@dataclass
class Recommendation:
    """Recommendation result model"""
    item_id: str
    score: float
    rank: int
    reason: str = ""
    algorithm: str = ""
    confidence: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "rank": self.rank,
            "reason": self.reason,
            "algorithm": self.algorithm,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp
        }


@dataclass
class RecommendationRequest:
    """Request for recommendations"""
    user_id: str
    num_items: int = 10
    context: Dict[str, Any] = field(default_factory=dict)
    algorithms: Optional[List[str]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    real_time: bool = True
    include_metadata: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "num_items": self.num_items,
            "context": self.context,
            "algorithms": self.algorithms,
            "filters": self.filters,
            "real_time": self.real_time,
            "include_metadata": self.include_metadata
        }


@dataclass
class RecommendationResponse:
    """Response containing recommendations"""
    user_id: str
    recommendations: List[Recommendation]
    total_time_ms: float
    algorithm_times: Dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "total_time_ms": self.total_time_ms,
            "algorithm_times": self.algorithm_times,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "metadata": self.metadata
        }


@dataclass
class FeatureVector:
    """Feature vector for ML algorithms"""
    entity_id: str  # user_id or item_id
    entity_type: str  # "user" or "item"
    features: Dict[str, Union[float, int, str, List]]
    embedding: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "features": self.features,
            "embedding": self.embedding,
            "timestamp": self.timestamp
        } 