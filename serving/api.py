"""
FastAPI Serving Layer for Real-time Recommendations

High-performance API server with sub-10ms latency serving.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.engine import RecommendationEngine, EngineConfig
from ..core.models import Interaction, InteractionType, RecommendationRequest
from ..streaming.kafka_processor import EventProcessor, EventProcessorConfig


# Pydantic models for API
class RecommendationRequestModel(BaseModel):
    """Request model for recommendations"""
    user_id: str = Field(..., description="User identifier")
    num_items: int = Field(10, ge=1, le=100, description="Number of recommendations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    algorithms: Optional[List[str]] = Field(None, description="Specific algorithms to use")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Recommendation filters")
    real_time: bool = Field(True, description="Include real-time updates")


class InteractionModel(BaseModel):
    """Model for user interactions"""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier") 
    interaction_type: str = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating (1-5)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Interaction context")


class RecommendationResponseModel(BaseModel):
    """Response model for recommendations"""
    user_id: str
    recommendations: List[Dict[str, Any]]
    total_time_ms: float
    algorithm_times: Dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False
    fallback_used: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponseModel(BaseModel):
    """Health check response model"""
    status: str
    timestamp: float
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)


# Global instances
recommendation_engine: Optional[RecommendationEngine] = None
event_processor: Optional[EventProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global recommendation_engine, event_processor
    
    # Startup
    logging.info("Starting Real-time Recommendation Engine API...")
    
    # Initialize recommendation engine
    config = EngineConfig(
        max_latency_ms=10,
        target_throughput=1000000,
        cache_size_gb=16,
        worker_threads=32
    )
    recommendation_engine = RecommendationEngine(config)
    
    # Initialize event processor
    event_config = EventProcessorConfig(
        kafka_brokers=["localhost:9092"],
        topics=["user_interactions", "item_updates"],
        batch_size=5000,
        max_latency_ms=50
    )
    
    async def handle_interaction(interaction: Interaction):
        """Handle incoming interactions"""
        if recommendation_engine:
            await recommendation_engine.update_user_interaction(
                interaction.user_id,
                interaction.item_id,
                interaction.interaction_type,
                interaction.context
            )
    
    event_processor = EventProcessor(event_config, handle_interaction)
    
    # Start background tasks
    asyncio.create_task(event_processor.start())
    
    logging.info("API startup complete")
    
    yield
    
    # Shutdown
    logging.info("Shutting down API...")
    
    if event_processor:
        await event_processor.stop()
    
    if recommendation_engine:
        await recommendation_engine.shutdown()
    
    logging.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Real-time Recommendation Engine",
    description="High-performance recommendation API with sub-10ms latency",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Rate limiting middleware (simple implementation)
user_request_times = {}
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds


async def rate_limit_dependency(user_id: str = Query(...)):
    """Simple rate limiting"""
    current_time = time.time()
    
    if user_id not in user_request_times:
        user_request_times[user_id] = []
    
    # Clean old requests
    user_request_times[user_id] = [
        t for t in user_request_times[user_id] 
        if current_time - t < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(user_request_times[user_id]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    
    user_request_times[user_id].append(current_time)
    return user_id


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Real-time Recommendation Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/recommendations", response_model=RecommendationResponseModel)
async def get_recommendations(
    request: RecommendationRequestModel,
    user_id: str = Depends(rate_limit_dependency)
):
    """
    Get personalized recommendations for a user
    
    This endpoint serves recommendations with sub-10ms latency using:
    - Multi-tier caching (L1/L2/L3)
    - Parallel algorithm execution
    - Pre-computed candidate sets
    - Real-time feature updates
    """
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    start_time = time.time()
    
    try:
        # Get recommendations
        recommendations = await recommendation_engine.get_recommendations(
            user_id=request.user_id,
            num_items=request.num_items,
            context=request.context,
            algorithms=request.algorithms,
            real_time=request.real_time
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Convert to response format
        rec_dicts = [rec.to_dict() for rec in recommendations]
        
        response = RecommendationResponseModel(
            user_id=request.user_id,
            recommendations=rec_dicts,
            total_time_ms=total_time_ms,
            cache_hit=total_time_ms < 2.0,  # Assume cache hit if very fast
            metadata={
                "algorithm_count": len(request.algorithms) if request.algorithms else 3,
                "request_id": f"req_{int(time.time() * 1000)}",
                "api_version": "1.0.0"
            }
        )
        
        # Log slow requests
        if total_time_ms > 10.0:
            logger.warning(f"Slow recommendation request: {total_time_ms:.2f}ms for user {request.user_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/interactions")
async def record_interaction(interaction: InteractionModel):
    """
    Record a user interaction for real-time model updates
    
    This endpoint accepts user interactions and updates the recommendation
    models in real-time to improve future recommendations.
    """
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    try:
        # Convert to internal interaction format
        interaction_type = InteractionType(interaction.interaction_type)
        
        await recommendation_engine.update_user_interaction(
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type,
            context=interaction.context
        )
        
        return {
            "status": "success",
            "message": "Interaction recorded",
            "timestamp": time.time()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid interaction type: {e}")
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=HealthResponseModel)
async def health_check():
    """
    Comprehensive health check endpoint
    
    Checks the health of all system components:
    - Recommendation engine
    - Cache systems
    - Feature store
    - Event processor
    """
    health_status = "healthy"
    components = {}
    
    try:
        # Check recommendation engine
        if recommendation_engine:
            engine_health = await recommendation_engine.health_check()
            components["recommendation_engine"] = engine_health.get("status", "unknown")
            
            if engine_health.get("status") != "healthy":
                health_status = "degraded"
        else:
            components["recommendation_engine"] = "not_initialized"
            health_status = "unhealthy"
        
        # Check event processor
        if event_processor:
            processor_health = await event_processor.health_check()
            components["event_processor"] = processor_health.get("status", "unknown")
            
            if processor_health.get("status") != "healthy":
                health_status = "degraded"
        else:
            components["event_processor"] = "not_initialized"
        
        return HealthResponseModel(
            status=health_status,
            timestamp=time.time(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponseModel(
            status="unhealthy",
            timestamp=time.time(),
            components={"error": str(e)}
        )


@app.get("/metrics")
async def get_metrics():
    """
    Get system performance metrics
    
    Returns detailed metrics including:
    - Request latency percentiles
    - Throughput statistics
    - Cache hit rates
    - Error rates
    """
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    try:
        # Get engine statistics
        engine_stats = recommendation_engine.get_statistics()
        
        # Get event processor statistics
        processor_stats = {}
        if event_processor:
            processor_stats = event_processor.get_statistics()
        
        return {
            "timestamp": time.time(),
            "engine": engine_stats,
            "event_processor": processor_stats,
            "api": {
                "active_users": len(user_request_times),
                "rate_limit_requests": RATE_LIMIT_REQUESTS,
                "rate_limit_window": RATE_LIMIT_WINDOW
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/recommendations")
async def get_user_recommendations(
    user_id: str,
    num_items: int = Query(10, ge=1, le=100),
    algorithms: Optional[str] = Query(None, description="Comma-separated algorithm names"),
    real_time: bool = Query(True)
):
    """
    Get recommendations for a specific user (GET endpoint)
    
    Alternative endpoint that accepts parameters as query strings
    for easier integration with caching layers and CDNs.
    """
    algorithm_list = None
    if algorithms:
        algorithm_list = [alg.strip() for alg in algorithms.split(",")]
    
    request = RecommendationRequestModel(
        user_id=user_id,
        num_items=num_items,
        algorithms=algorithm_list,
        real_time=real_time
    )
    
    return await get_recommendations(request, user_id)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": time.time(),
            "path": str(request.url)
        }
    )


class RecommendationAPI:
    """Wrapper class for the recommendation API"""
    
    def __init__(self, engine: RecommendationEngine, port: int = 8000):
        self.engine = engine
        self.port = port
        global recommendation_engine
        recommendation_engine = engine
    
    def run(self, host: str = "0.0.0.0", workers: int = 1):
        """Run the API server"""
        uvicorn.run(
            "serving.api:app",
            host=host,
            port=self.port,
            workers=workers,
            log_level="info",
            access_log=True
        )


def main():
    """Main entry point for running the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Recommendation Engine API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "serving.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main() 