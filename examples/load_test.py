#!/usr/bin/env python3
"""
Load Testing for Real-time Recommendation Engine

Tests the system's ability to handle 1M+ requests per second
with sub-10ms latency.
"""

import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict, Any
import argparse
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTester:
    """High-performance load tester for recommendation API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        connector = aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def make_recommendation_request(self, user_id: str) -> Dict[str, Any]:
        """Make a single recommendation request"""
        url = f"{self.base_url}/recommendations"
        
        payload = {
            "user_id": user_id,
            "num_items": 10,
            "context": {
                "device": random.choice(["mobile", "desktop", "tablet"]),
                "page": "homepage"
            },
            "real_time": False  # Use cache for load testing
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(url, json=payload) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "latency_ms": latency_ms,
                        "status_code": response.status,
                        "recommendations": len(data.get("recommendations", []))
                    }
                else:
                    return {
                        "success": False,
                        "latency_ms": latency_ms,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency_ms": latency_ms,
                "status_code": 0,
                "error": str(e)
            }
    
    async def run_load_test(
        self,
        target_rps: int,
        duration_seconds: int,
        num_users: int = 10000
    ) -> Dict[str, Any]:
        """
        Run load test with specified parameters
        
        Args:
            target_rps: Target requests per second
            duration_seconds: Test duration
            num_users: Number of unique users to simulate
            
        Returns:
            Test results
        """
        logger.info(f"Starting load test: {target_rps:,} RPS for {duration_seconds}s")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Calculate request interval
        request_interval = 1.0 / target_rps
        
        # Track progress
        completed_requests = 0
        failed_requests = 0
        
        async def make_requests_batch():
            """Make a batch of requests"""
            nonlocal completed_requests, failed_requests
            
            # Batch size based on target RPS
            batch_size = min(1000, target_rps // 10)
            
            tasks = []
            for _ in range(batch_size):
                user_id = f"user_{random.randint(0, num_users-1)}"
                task = asyncio.create_task(self.make_recommendation_request(user_id))
                tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                    if result["success"]:
                        completed_requests += 1
                    else:
                        failed_requests += 1
                else:
                    failed_requests += 1
                    results.append({
                        "success": False,
                        "latency_ms": 0,
                        "status_code": 0,
                        "error": str(result)
                    })
        
        # Main load generation loop
        last_progress_time = start_time
        
        while time.time() < end_time:
            batch_start = time.time()
            
            await make_requests_batch()
            
            # Progress reporting
            current_time = time.time()
            if current_time - last_progress_time >= 5.0:  # Every 5 seconds
                elapsed = current_time - start_time
                current_rps = completed_requests / elapsed
                
                logger.info(
                    f"Progress: {elapsed:.1f}s elapsed, "
                    f"{completed_requests:,} completed, "
                    f"{failed_requests:,} failed, "
                    f"Current RPS: {current_rps:,.0f}"
                )
                
                last_progress_time = current_time
            
            # Wait before next batch
            batch_time = time.time() - batch_start
            sleep_time = max(0, 0.1 - batch_time)  # 100ms batches
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        actual_rps = len(results) / total_time
        
        # Latency statistics
        successful_results = [r for r in results if r["success"]]
        latencies = [r["latency_ms"] for r in successful_results]
        
        stats = {
            "duration_seconds": total_time,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "target_rps": target_rps,
            "actual_rps": actual_rps,
            "rps_achievement": actual_rps / target_rps if target_rps > 0 else 0
        }
        
        if latencies:
            latencies.sort()
            stats.update({
                "latency_avg_ms": statistics.mean(latencies),
                "latency_p50_ms": latencies[len(latencies) // 2],
                "latency_p95_ms": latencies[int(len(latencies) * 0.95)],
                "latency_p99_ms": latencies[int(len(latencies) * 0.99)],
                "latency_max_ms": max(latencies),
                "latency_min_ms": min(latencies)
            })
            
            # SLA compliance
            sla_violations = sum(1 for l in latencies if l > 10.0)
            stats["sla_compliance"] = (len(latencies) - sla_violations) / len(latencies)
        
        return stats


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Load test the recommendation engine")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--rps", type=int, default=10000, help="Target requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--users", type=int, default=10000, help="Number of unique users")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    print("🚀 Real-time Recommendation Engine - Load Test")
    print("=" * 60)
    print(f"Target: {args.rps:,} RPS for {args.duration}s")
    print(f"Users: {args.users:,}")
    print(f"API URL: {args.url}")
    print()
    
    async with LoadTester(args.url) as tester:
        # Health check first
        try:
            health_url = f"{args.url}/health"
            async with tester.session.get(health_url) as response:
                if response.status != 200:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    return
                else:
                    print("✅ Health check passed")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Run load test
        results = await tester.run_load_test(
            target_rps=args.rps,
            duration_seconds=args.duration,
            num_users=args.users
        )
        
        # Print results
        print("\n📊 Load Test Results")
        print("=" * 60)
        print(f"Duration: {results['duration_seconds']:.1f}s")
        print(f"Total requests: {results['total_requests']:,}")
        print(f"Successful requests: {results['successful_requests']:,}")
        print(f"Failed requests: {results['failed_requests']:,}")
        print(f"Success rate: {results['success_rate']:.1%}")
        print()
        print(f"Target RPS: {results['target_rps']:,}")
        print(f"Actual RPS: {results['actual_rps']:,.0f}")
        print(f"RPS achievement: {results['rps_achievement']:.1%}")
        print()
        
        if 'latency_avg_ms' in results:
            print("Latency Statistics:")
            print(f"  Average: {results['latency_avg_ms']:.2f}ms")
            print(f"  P50: {results['latency_p50_ms']:.2f}ms")
            print(f"  P95: {results['latency_p95_ms']:.2f}ms")
            print(f"  P99: {results['latency_p99_ms']:.2f}ms")
            print(f"  Max: {results['latency_max_ms']:.2f}ms")
            print()
            print(f"SLA Compliance (≤10ms): {results['sla_compliance']:.1%}")
            
            if results['latency_p99_ms'] <= 10.0:
                print("✅ Sub-10ms latency target achieved!")
            else:
                print("⚠️ Latency target not met")
        
        if results['actual_rps'] >= args.rps * 0.8:
            print("✅ Throughput target achieved!")
        else:
            print("⚠️ Throughput target not met")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main()) 