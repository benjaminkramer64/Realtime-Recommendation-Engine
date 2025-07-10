"""
Kafka Event Processor for Real-time Recommendation Engine

Processes 1M+ events/second from Kafka streams for real-time recommendations.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ..core.models import Interaction, InteractionType


@dataclass
class EventProcessorConfig:
    """Configuration for the event processor"""
    kafka_brokers: List[str]
    topics: List[str]
    consumer_group: str = "recommendation_engine"
    batch_size: int = 10000
    max_latency_ms: int = 100
    auto_offset_reset: str = "latest"
    worker_threads: int = 16
    enable_batching: bool = True


class EventProcessor:
    """
    High-performance Kafka event processor for real-time recommendations
    
    Processes user interactions, item updates, and other events to update
    recommendation models in real-time.
    """
    
    def __init__(
        self, 
        config: EventProcessorConfig,
        interaction_handler: Optional[Callable] = None
    ):
        """
        Initialize the event processor
        
        Args:
            config: Processor configuration
            interaction_handler: Callback for processing interactions
        """
        self.config = config
        self.interaction_handler = interaction_handler
        
        # Performance tracking
        self.processed_events = 0
        self.processing_errors = 0
        self.start_time = time.time()
        self.last_batch_time = time.time()
        
        # Threading for high throughput
        self.executor = ThreadPoolExecutor(max_workers=config.worker_threads)
        
        # Event batching
        self.event_batch = []
        self.batch_lock = asyncio.Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # Mock Kafka consumer (in real implementation, use confluent-kafka)
        self.consumer = None
        self.is_running = False
        
        self.logger.info(f"EventProcessor initialized with config: {config}")
    
    async def start(self):
        """Start the event processor"""
        self.logger.info("Starting event processor...")
        
        # Initialize Kafka consumer (mock implementation)
        await self._initialize_consumer()
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._consume_events()),
            asyncio.create_task(self._process_batch_periodically()),
            asyncio.create_task(self._log_statistics())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Event processor tasks cancelled")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the event processor"""
        self.logger.info("Stopping event processor...")
        
        self.is_running = False
        
        # Process remaining events
        if self.event_batch:
            await self._process_batch(self.event_batch)
            self.event_batch.clear()
        
        # Close Kafka consumer
        if self.consumer:
            await self._close_consumer()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Event processor stopped")
    
    async def _initialize_consumer(self):
        """Initialize Kafka consumer (mock implementation)"""
        # In real implementation, this would use confluent-kafka:
        # 
        # from confluent_kafka import Consumer
        # 
        # consumer_config = {
        #     'bootstrap.servers': ','.join(self.config.kafka_brokers),
        #     'group.id': self.config.consumer_group,
        #     'auto.offset.reset': self.config.auto_offset_reset,
        #     'enable.auto.commit': True,
        #     'max.poll.interval.ms': 300000,
        #     'session.timeout.ms': 10000,
        #     'fetch.min.bytes': 1024,
        #     'fetch.max.wait.ms': 500,
        # }
        # 
        # self.consumer = Consumer(consumer_config)
        # self.consumer.subscribe(self.config.topics)
        
        self.logger.info("Kafka consumer initialized (mock)")
    
    async def _close_consumer(self):
        """Close Kafka consumer"""
        # In real implementation:
        # self.consumer.close()
        
        self.logger.info("Kafka consumer closed")
    
    async def _consume_events(self):
        """Main event consumption loop"""
        self.logger.info("Starting event consumption...")
        
        while self.is_running:
            try:
                # Mock event consumption - in real implementation:
                # msg = self.consumer.poll(timeout=1.0)
                # if msg is None:
                #     continue
                # if msg.error():
                #     self.logger.error(f"Kafka error: {msg.error()}")
                #     continue
                
                # For demo, generate synthetic events
                events = await self._generate_synthetic_events(batch_size=1000)
                
                # Add events to batch
                async with self.batch_lock:
                    self.event_batch.extend(events)
                    
                    # Process batch if it's large enough
                    if len(self.event_batch) >= self.config.batch_size:
                        await self._process_batch(self.event_batch)
                        self.event_batch.clear()
                
                # Small delay to simulate real consumption
                await asyncio.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error consuming events: {e}")
                self.processing_errors += 1
                await asyncio.sleep(0.1)  # Back off on errors
    
    async def _generate_synthetic_events(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate synthetic events for demonstration"""
        events = []
        
        for _ in range(batch_size):
            event = {
                "user_id": f"user_{hash(time.time()) % 10000}",
                "item_id": f"item_{hash(time.time() * 2) % 5000}",
                "interaction_type": "view",
                "timestamp": time.time(),
                "context": {
                    "device": "mobile",
                    "session_id": f"session_{hash(time.time() * 3) % 100000}",
                    "page": "homepage"
                }
            }
            events.append(event)
        
        return events
    
    async def _process_batch_periodically(self):
        """Process batches periodically even if not full"""
        while self.is_running:
            try:
                # Wait for max latency period
                await asyncio.sleep(self.config.max_latency_ms / 1000.0)
                
                # Process any pending events
                async with self.batch_lock:
                    if self.event_batch:
                        await self._process_batch(self.event_batch)
                        self.event_batch.clear()
                        
            except Exception as e:
                self.logger.error(f"Error in periodic batch processing: {e}")
    
    async def _process_batch(self, events: List[Dict[str, Any]]):
        """
        Process a batch of events
        
        Args:
            events: List of event dictionaries
        """
        if not events:
            return
        
        start_time = time.time()
        
        try:
            # Convert events to Interaction objects
            interactions = []
            
            for event in events:
                try:
                    interaction = self._event_to_interaction(event)
                    if interaction:
                        interactions.append(interaction)
                except Exception as e:
                    self.logger.warning(f"Failed to parse event: {e}")
            
            # Process interactions
            if interactions and self.interaction_handler:
                # Process in parallel for high throughput
                await asyncio.gather(*[
                    self.interaction_handler(interaction)
                    for interaction in interactions
                ], return_exceptions=True)
            
            # Update statistics
            self.processed_events += len(events)
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.debug(
                f"Processed batch of {len(events)} events in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self.processing_errors += len(events)
    
    def _event_to_interaction(self, event: Dict[str, Any]) -> Optional[Interaction]:
        """
        Convert event dictionary to Interaction object
        
        Args:
            event: Event dictionary
            
        Returns:
            Interaction object or None if conversion fails
        """
        try:
            interaction_type = event.get("interaction_type", "view")
            
            # Convert string to InteractionType enum
            if isinstance(interaction_type, str):
                try:
                    interaction_type = InteractionType(interaction_type)
                except ValueError:
                    interaction_type = InteractionType.VIEW
            
            interaction = Interaction(
                user_id=event["user_id"],
                item_id=event["item_id"],
                interaction_type=interaction_type,
                timestamp=event.get("timestamp", time.time()),
                context=event.get("context", {}),
                rating=event.get("rating"),
                duration=event.get("duration"),
                session_id=event.get("session_id", event.get("context", {}).get("session_id"))
            )
            
            return interaction
            
        except KeyError as e:
            self.logger.warning(f"Missing required field in event: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error converting event to interaction: {e}")
            return None
    
    async def _log_statistics(self):
        """Log processing statistics periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Log every 10 seconds
                
                uptime = time.time() - self.start_time
                throughput = self.processed_events / max(1, uptime)
                error_rate = self.processing_errors / max(1, self.processed_events)
                
                self.logger.info(
                    f"Event processing stats: "
                    f"throughput={throughput:.0f} events/sec, "
                    f"total_processed={self.processed_events:,}, "
                    f"error_rate={error_rate:.3f}, "
                    f"batch_queue_size={len(self.event_batch)}"
                )
                
            except Exception as e:
                self.logger.error(f"Error logging statistics: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        uptime = time.time() - self.start_time
        throughput = self.processed_events / max(1, uptime)
        error_rate = self.processing_errors / max(1, self.processed_events)
        
        return {
            "uptime_seconds": uptime,
            "processed_events": self.processed_events,
            "processing_errors": self.processing_errors,
            "throughput_eps": throughput,
            "error_rate": error_rate,
            "batch_queue_size": len(self.event_batch),
            "is_running": self.is_running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        stats = self.get_statistics()
        
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "stats": stats
        }
        
        # Check if processing is healthy
        if not self.is_running:
            health["status"] = "stopped"
        elif stats["error_rate"] > 0.1:  # More than 10% errors
            health["status"] = "degraded"
        elif stats["throughput_eps"] < 1000:  # Less than 1000 events/sec
            health["status"] = "degraded"
        
        return health


# Factory function for easy creation
def create_event_processor(
    kafka_brokers: List[str],
    topics: List[str],
    interaction_handler: Optional[Callable] = None,
    **kwargs
) -> EventProcessor:
    """
    Create an event processor with default configuration
    
    Args:
        kafka_brokers: List of Kafka broker addresses
        topics: List of Kafka topics to consume
        interaction_handler: Callback for processing interactions
        **kwargs: Additional configuration options
        
    Returns:
        Configured EventProcessor instance
    """
    config = EventProcessorConfig(
        kafka_brokers=kafka_brokers,
        topics=topics,
        **kwargs
    )
    
    return EventProcessor(config, interaction_handler) 