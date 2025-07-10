"""
Algorithm Registry for Recommendation Algorithms

Manages the lifecycle and execution of recommendation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import time

from .models import User, Item, Interaction, Recommendation, FeatureVector


@dataclass
class AlgorithmConfig:
    """Configuration for a recommendation algorithm"""
    name: str
    weight: float = 1.0
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}


class BaseRecommendationAlgorithm(ABC):
    """
    Base class for all recommendation algorithms
    """
    
    def __init__(self, config: Optional[AlgorithmConfig] = None):
        self.config = config or AlgorithmConfig(name=self.__class__.__name__)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_trained = False
        self.last_update = time.time()
        
    @abstractmethod
    async def predict(
        self, 
        user_features: FeatureVector,
        candidate_items: List[str],
        num_recommendations: int = 10
    ) -> List[Recommendation]:
        """
        Generate recommendations for a user
        
        Args:
            user_features: User feature vector
            candidate_items: List of candidate item IDs
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendations
        """
        pass
    
    @abstractmethod
    async def train(self, interactions: List[Interaction], **kwargs):
        """
        Train the algorithm on interaction data
        
        Args:
            interactions: List of user-item interactions
            **kwargs: Additional training parameters
        """
        pass
    
    async def update_online(self, interaction: Interaction):
        """
        Update the algorithm with a single interaction (online learning)
        
        Args:
            interaction: New user-item interaction
        """
        # Default implementation - can be overridden by algorithms that support online learning
        self.logger.debug(f"Online update not implemented for {self.__class__.__name__}")
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata"""
        return {
            "name": self.config.name,
            "is_trained": self.is_trained,
            "last_update": self.last_update,
            "config": self.config.config
        }


class AlgorithmRegistry:
    """
    Registry for managing recommendation algorithms
    """
    
    def __init__(self):
        self.algorithms: Dict[str, BaseRecommendationAlgorithm] = {}
        self.configs: Dict[str, AlgorithmConfig] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_algorithm(
        self, 
        name: str, 
        algorithm: BaseRecommendationAlgorithm,
        config: Optional[AlgorithmConfig] = None
    ):
        """
        Register a recommendation algorithm
        
        Args:
            name: Algorithm name
            algorithm: Algorithm instance
            config: Algorithm configuration
        """
        if config:
            algorithm.config = config
        
        self.algorithms[name] = algorithm
        self.configs[name] = algorithm.config
        
        self.logger.info(f"Registered algorithm: {name}")
    
    def get_algorithm(self, name: str) -> Optional[BaseRecommendationAlgorithm]:
        """Get algorithm by name"""
        return self.algorithms.get(name)
    
    def get_enabled_algorithms(self) -> Dict[str, BaseRecommendationAlgorithm]:
        """Get all enabled algorithms"""
        return {
            name: algo for name, algo in self.algorithms.items()
            if self.configs[name].enabled
        }
    
    def enable_algorithm(self, name: str):
        """Enable an algorithm"""
        if name in self.configs:
            self.configs[name].enabled = True
            self.logger.info(f"Enabled algorithm: {name}")
    
    def disable_algorithm(self, name: str):
        """Disable an algorithm"""
        if name in self.configs:
            self.configs[name].enabled = False
            self.logger.info(f"Disabled algorithm: {name}")
    
    def update_algorithm_weight(self, name: str, weight: float):
        """Update algorithm weight"""
        if name in self.configs:
            self.configs[name].weight = weight
            self.logger.info(f"Updated weight for {name}: {weight}")
    
    def get_algorithm_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all algorithms"""
        stats = {}
        for name, algo in self.algorithms.items():
            stats[name] = {
                "enabled": self.configs[name].enabled,
                "weight": self.configs[name].weight,
                "metadata": algo.get_metadata()
            }
        return stats
    
    async def train_all_algorithms(self, interactions: List[Interaction]):
        """Train all enabled algorithms"""
        enabled_algos = self.get_enabled_algorithms()
        
        for name, algo in enabled_algos.items():
            try:
                self.logger.info(f"Training algorithm: {name}")
                await algo.train(interactions)
                self.logger.info(f"Successfully trained algorithm: {name}")
            except Exception as e:
                self.logger.error(f"Failed to train algorithm {name}: {e}")
    
    def list_algorithms(self) -> List[str]:
        """List all registered algorithm names"""
        return list(self.algorithms.keys()) 