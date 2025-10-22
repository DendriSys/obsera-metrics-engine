"""
Metrics AI Engine - FastAPI Application
Main API server for AI-powered metrics analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional
import time
from loguru import logger

from ..prometheus.client import PrometheusClient, PrometheusConfig, TimeSeriesData
from ..models.metric_data import (
    MetricIngestRequest,
    MetricQueryRequest,
    MetricAnalyzeRequest,
    MetricAnalyzeResponse,
    HealthResponse,
    MetricIdentifier,
    MetricSeries,
    MetricPoint,
    TimeRange,
    MetricFeatures
)
from ..models.feature_extractor import FeatureExtractor
from ..vector.ollama_client import OllamaClient, OllamaConfig
from ..vector.vector_store import VectorStore
from ..anomaly.detector import AnomalyDetectorEnsemble


# Global state
class AppState:
    """Application state"""
    prometheus: Optional[PrometheusClient] = None
    ollama: Optional[OllamaClient] = None
    vector_store: Optional[VectorStore] = None
    feature_extractor: Optional[FeatureExtractor] = None
    anomaly_detector: Optional[AnomalyDetectorEnsemble] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    logger.info("Starting Metrics AI Engine...")
    
    # Initialize Prometheus client
    prom_config = PrometheusConfig(
        url="http://prometheus:9090",  # Adjust as needed
        timeout=30
    )
    state.prometheus = PrometheusClient(prom_config)
    
    # Initialize Ollama client
    ollama_config = OllamaConfig(
        url="http://ollama:11434",  # Adjust as needed
        embedding_model="nomic-embed-text",
        generation_model="codellama:13b"
    )
    state.ollama = OllamaClient(ollama_config)
    
    # Initialize vector store
    state.vector_store = VectorStore(dimension=768, storage_path="./vector_store")
    
    # Initialize feature extractor
    state.feature_extractor = FeatureExtractor()
    
    # Initialize anomaly detector
    state.anomaly_detector = AnomalyDetectorEnsemble(min_consensus=2)
    
    logger.info("Metrics AI Engine started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Metrics AI Engine...")
    
    if state.prometheus:
        await state.prometheus.close()
    
    if state.ollama:
        await state.ollama.close()
    
    if state.vector_store:
        state.vector_store.save()
    
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Metrics AI Engine",
    description="AI-powered metrics analysis and anomaly detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions

def parse_time_range(time_range_str: str) -> tuple[datetime, datetime, str]:
    """
    Parse time range string to start/end datetimes
    
    Args:
        time_range_str: String like '2w', '24h', '30d'
    
    Returns:
        Tuple of (start, end, step)
    """
    end = datetime.now()
    
    # Parse duration
    unit = time_range_str[-1]
    value = int(time_range_str[:-1])
    
    if unit == 'h':
        start = end - timedelta(hours=value)
        step = "1m" if value <= 6 else "5m"
    elif unit == 'd':
        start = end - timedelta(days=value)
        step = "5m" if value <= 1 else "1h"
    elif unit == 'w':
        start = end - timedelta(weeks=value)
        step = "1h" if value <= 1 else "6h"
    elif unit == 'm':  # months
        start = end - timedelta(days=value * 30)
        step = "1d"
    else:
        raise ValueError(f"Invalid time unit: {unit}")
    
    return start, end, step


def build_promql(metric_name: str, labels: Optional[dict] = None) -> str:
    """Build PromQL query from metric name and labels"""
    if not labels:
        return metric_name
    
    label_selectors = ",".join([f'{k}="{v}"' for k, v in labels.items()])
    return f"{metric_name}{{{label_selectors}}}"


# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "Metrics AI Engine",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns system health status
    """
    try:
        # Check Prometheus
        prom_health = await state.prometheus.health_check() if state.prometheus else False
        
        # Check Ollama
        ollama_health = await state.ollama.health_check() if state.ollama else False
        
        # Check vector store
        vector_health = state.vector_store is not None
        
        overall_status = "healthy" if all([prom_health, ollama_health, vector_health]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            prometheus_connected=prom_health,
            ollama_connected=ollama_health,
            vector_db_ready=vector_health,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/ingest", tags=["Metrics"])
async def ingest_metrics(request: MetricIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest metrics from Prometheus and store in vector database
    
    This endpoint:
    1. Queries Prometheus for the specified metric
    2. Extracts features (statistics, trends, patterns)
    3. Generates embeddings using Ollama
    4. Stores in FAISS vector database for semantic search
    """
    try:
        logger.info(f"Ingesting metric: {request.metric_name}")
        
        # Build PromQL query
        query = build_promql(request.metric_name, request.labels)
        
        # Query Prometheus
        time_series_list = await state.prometheus.query_range(
            query=query,
            start=request.start_time,
            end=request.end_time,
            step=request.step
        )
        
        if not time_series_list:
            raise HTTPException(status_code=404, detail="No data found for query")
        
        ingested_count = 0
        
        for ts_data in time_series_list:
            # Convert to MetricSeries
            metric = MetricIdentifier(
                name=ts_data.metric_name or request.metric_name,
                labels=ts_data.labels
            )
            
            points = [
                MetricPoint(timestamp=p.timestamp, value=p.value)
                for p in ts_data.values
            ]
            
            series = MetricSeries(metric=metric, points=points)
            
            # Extract features
            time_range = TimeRange(
                start=request.start_time,
                end=request.end_time,
                step=request.step
            )
            features = state.feature_extractor.extract_features(series, time_range)
            
            # Detect anomalies
            anomalies = state.anomaly_detector.detect(series, features)
            features.anomalies = anomalies
            
            # Generate embedding
            text = features.to_text()
            embedding = await state.ollama.generate_embedding(text)
            
            # Store in vector database
            metric_id = f"{metric.name}_{hash(str(metric.labels))}"
            state.vector_store.add(
                metric_id=metric_id,
                metric=metric,
                embedding=embedding,
                features=features,
                timestamp=time.time()
            )
            
            ingested_count += 1
        
        # Save vector store in background
        background_tasks.add_task(state.vector_store.save)
        
        return {
            "status": "success",
            "ingested": ingested_count,
            "metric": request.metric_name,
            "time_range": f"{request.start_time} to {request.end_time}"
        }
        
    except Exception as e:
        logger.error(f"Error ingesting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/query", tags=["Metrics"])
async def query_metrics(request: MetricQueryRequest):
    """
    Semantic search for metrics using natural language
    
    Examples:
    - "high CPU usage patterns"
    - "memory spikes in payment service"
    - "metrics showing increasing trends"
    """
    try:
        logger.info(f"Querying metrics: {request.query}")
        
        # Generate embedding for query
        query_embedding = await state.ollama.generate_embedding(request.query)
        
        # Search vector store
        results = state.vector_store.search(
            query_embedding=query_embedding,
            k=request.top_k,
            filters=request.filters
        )
        
        # Format response
        response = []
        for metadata, distance in results:
            response.append({
                "metric": {
                    "name": metadata.metric.name,
                    "labels": metadata.metric.labels
                },
                "similarity": float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                "summary": metadata.features_summary[:200] + "...",
                "timestamp": metadata.timestamp
            })
        
        return {
            "query": request.query,
            "results": response,
            "count": len(response)
        }
        
    except Exception as e:
        logger.error(f"Error querying metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/analyze", response_model=MetricAnalyzeResponse, tags=["Metrics"])
async def analyze_metrics(request: MetricAnalyzeRequest):
    """
    Analyze metric trends, detect anomalies, and generate AI insights
    
    Performs complete analysis:
    - Trend analysis (direction, strength, change %)
    - Seasonal pattern detection
    - Anomaly detection (multiple algorithms)
    - AI-generated insights and recommendations
    """
    try:
        logger.info(f"Analyzing metric: {request.metric_name}")
        
        # Parse time range
        start, end, step = parse_time_range(request.time_range)
        
        # Build query
        query = build_promql(request.metric_name, request.labels)
        
        # Query Prometheus
        time_series_list = await state.prometheus.query_range(
            query=query,
            start=start,
            end=end,
            step=step
        )
        
        if not time_series_list:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Analyze first series (can be extended for multiple)
        ts_data = time_series_list[0]
        
        metric = MetricIdentifier(
            name=ts_data.metric_name or request.metric_name,
            labels=ts_data.labels
        )
        
        points = [
            MetricPoint(timestamp=p.timestamp, value=p.value)
            for p in ts_data.values
        ]
        
        series = MetricSeries(metric=metric, points=points)
        
        # Extract features
        time_range = TimeRange(start=start, end=end, step=step)
        features = state.feature_extractor.extract_features(series, time_range)
        
        # Detect anomalies
        if "anomaly" in request.analysis_types:
            anomalies = state.anomaly_detector.detect(series, features)
            features.anomalies = anomalies
        
        # Generate AI insights
        ai_insights = None
        recommendations = []
        
        if "insights" in request.analysis_types or True:  # Always generate
            context = features.to_text()
            ai_insights = await state.ollama.generate_insights(context)
            
            # Generate recommendations based on anomalies
            if features.anomalies:
                recommendations.append(f"Investigate {len(features.anomalies)} detected anomalies")
            
            if features.trend and features.trend.direction == "increasing":
                recommendations.append("Monitor for potential capacity issues")
            elif features.trend and features.trend.direction == "decreasing":
                recommendations.append("Verify if decreasing trend is expected")
        
        return MetricAnalyzeResponse(
            metric=metric,
            features=features,
            ai_insights=ai_insights,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error analyzing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/stats", tags=["Metrics"])
async def get_stats():
    """Get vector store statistics"""
    try:
        stats = state.vector_store.stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
