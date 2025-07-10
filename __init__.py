"""
Real-time Recommendation Engine

A high-performance, scalable recommendation system capable of processing
1M+ events per second with sub-10ms serving latency.
"""

__version__ = "1.0.0"
__author__ = "Real-time Recommendation Engine Team"
__email__ = "team@rtre.dev"

from .core.engine import RecommendationEngine
from .streaming.kafka_processor import EventProcessor
from .serving.api import RecommendationAPI
from .ml.algorithms import CollaborativeFiltering, ContentBased, DeepLearning

__all__ = [
    "RecommendationEngine",
    "EventProcessor", 
    "RecommendationAPI",
    "CollaborativeFiltering",
    "ContentBased",
    "DeepLearning"
] 