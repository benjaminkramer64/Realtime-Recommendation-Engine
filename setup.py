#!/usr/bin/env python3
"""
Setup configuration for Real-time Recommendation Engine
"""

from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="realtime-recommendation-engine",
    version="1.0.0",
    author="Real-time Recommendation Engine Team",
    author_email="team@rtre.dev",
    description="A high-performance, scalable recommendation system capable of processing 1M+ events per second with sub-10ms serving latency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/realtime-recommendation-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core Dependencies
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        
        # Streaming & Event Processing
        "kafka-python>=2.0.2",
        "confluent-kafka>=2.2.0",
        "redis>=4.6.0",
        
        # Machine Learning & Recommendations
        "torch>=2.0.0",
        "tensorflow>=2.13.0",
        "lightgbm>=4.0.0",
        "implicit>=0.7.0",
        "surprise>=1.1.3",
        "faiss-cpu>=1.7.4",
        "annoy>=1.17.0",
        
        # High-Performance Serving
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "aiohttp>=3.8.0",
        
        # Storage & Databases
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.28.0",
        "pymongo>=4.4.0",
        
        # Caching & Performance
        "aiocache>=0.12.0",
        "cachetools>=5.3.0",
        "msgpack>=1.0.5",
        "orjson>=3.9.0",
        
        # Monitoring & Observability
        "prometheus-client>=0.17.0",
        "structlog>=23.1.0",
        
        # Data Processing
        "pyarrow>=12.0.0",
        "polars>=0.18.0",
        
        # Web & API
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        
        # Utilities
        "click>=8.1.0",
        "rich>=13.4.0",
        "pydantic>=2.0.0",
        "tenacity>=8.2.0",
        "httpx>=0.24.0",
        "aiofiles>=23.1.0",
        
        # Configuration
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "locust>=2.15.0",
            "hypothesis>=6.82.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "flake8>=6.0.0",
        ],
        "streaming": [
            "apache-flink>=1.17.0",
            "pyflink>=1.17.0",
            "ray[default]>=2.6.0",
            "dask>=2023.6.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "tensorflow-gpu>=2.13.0",
            "faiss-gpu>=1.7.4",
        ],
        "monitoring": [
            "statsd>=4.0.1",
            "opencensus>=0.11.0",
            "jaeger-client>=4.8.0",
            "dash>=2.11.0",
        ],
        "all": [
            "apache-flink>=1.17.0",
            "pyflink>=1.17.0",
            "ray[default]>=2.6.0",
            "statsd>=4.0.1",
            "dash>=2.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "rtre-server=serving.api:main",
            "rtre-demo=examples.demo:main",
            "rtre-benchmark=examples.load_test:main",
            "rtre-train=ml.training:main",
        ],
    },
    include_package_data=True,
    package_data={
        "realtime_recommendation_engine": [
            "config/*.yaml",
            "config/*.json",
            "examples/*.py",
            "docs/*.md",
        ],
    },
    zip_safe=False,
    keywords="recommendation-engine real-time machine-learning streaming high-performance low-latency",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/realtime-recommendation-engine/issues",
        "Source": "https://github.com/yourusername/realtime-recommendation-engine",
        "Documentation": "https://realtime-recommendation-engine.readthedocs.io/",
    },
) 