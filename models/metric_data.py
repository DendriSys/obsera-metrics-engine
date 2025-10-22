"""
Data models for metrics AI engine
Pydantic models for API requests/responses and internal data structures
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class IntentType(str, Enum):
    """Types of query intents"""
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CURRENT_VALUE = "current_value"
    COMPARISON = "comparison"
    FORECAST = "forecast"
    CORRELATION = "correlation"


class AnomalyType(str, Enum):
    """Types of anomalies"""
    SPIKE = "spike"
    DROP = "drop"
    LEVEL_SHIFT = "level_shift"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    UNKNOWN = "unknown"


class MetricIdentifier(BaseModel):
    """Identifies a specific metric with labels"""
    name: str = Field(..., description="Metric name (e.g., 'cpu_usage')")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels/tags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "cpu_usage",
                "labels": {"service": "payment", "instance": "prod-1"}
            }
        }


class TimeRange(BaseModel):
    """Time range for queries"""
    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")
    step: str = Field(default="1m", description="Query resolution (e.g., '1m', '5m', '1h')")
    
    @validator("step")
    def validate_step(cls, v):
        """Validate step format"""
        valid_units = ["s", "m", "h", "d"]
        if not any(v.endswith(unit) for unit in valid_units):
            raise ValueError(f"Step must end with one of: {valid_units}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "start": "2025-10-21T00:00:00Z",
                "end": "2025-10-22T00:00:00Z",
                "step": "5m"
            }
        }


class MetricPoint(BaseModel):
    """Single metric data point"""
    timestamp: float = Field(..., description="Unix timestamp")
    value: float = Field(..., description="Metric value")


class MetricSeries(BaseModel):
    """Complete metric time-series"""
    metric: MetricIdentifier
    points: List[MetricPoint]
    
    @property
    def timestamps(self) -> List[float]:
        """Get all timestamps"""
        return [p.timestamp for p in self.points]
    
    @property
    def values(self) -> List[float]:
        """Get all values"""
        return [p.value for p in self.points]


class StatisticalSummary(BaseModel):
    """Statistical summary of metric data"""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float


class TrendInfo(BaseModel):
    """Trend information"""
    direction: str = Field(..., description="'increasing', 'decreasing', or 'stable'")
    strength: float = Field(..., description="Strength of trend (0-1)")
    slope: float = Field(..., description="Rate of change per time unit")
    change_percent: float = Field(..., description="Percentage change over period")


class SeasonalPattern(BaseModel):
    """Detected seasonal pattern"""
    period: str = Field(..., description="Period of seasonality (e.g., 'daily', 'weekly')")
    strength: float = Field(..., description="Strength of pattern (0-1)")
    peaks: List[int] = Field(default_factory=list, description="Time indices of peaks")
    troughs: List[int] = Field(default_factory=list, description="Time indices of troughs")


class Anomaly(BaseModel):
    """Detected anomaly"""
    timestamp: float
    value: float
    expected_value: float
    deviation: float
    deviation_percent: float
    anomaly_type: AnomalyType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    context: Optional[str] = None


class MetricFeatures(BaseModel):
    """Extracted features from metric data"""
    metric: MetricIdentifier
    time_range: TimeRange
    statistics: StatisticalSummary
    trend: Optional[TrendInfo] = None
    seasonal: Optional[SeasonalPattern] = None
    anomalies: List[Anomaly] = Field(default_factory=list)
    
    def to_text(self) -> str:
        """
        Convert features to text representation for vector embedding
        
        Returns:
            Human-readable text description of metric features
        """
        parts = []
        
        # Metric identification
        parts.append(f"Metric: {self.metric.name}")
        if self.metric.labels:
            label_str = ", ".join([f"{k}={v}" for k, v in self.metric.labels.items()])
            parts.append(f"Labels: {label_str}")
        
        # Statistical summary
        parts.append(f"\nStatistics:")
        parts.append(f"- Average: {self.statistics.mean:.2f}")
        parts.append(f"- Range: {self.statistics.min_value:.2f} to {self.statistics.max_value:.2f}")
        parts.append(f"- Std Dev: {self.statistics.std_dev:.2f}")
        parts.append(f"- 95th percentile: {self.statistics.percentile_95:.2f}")
        
        # Trend information
        if self.trend:
            parts.append(f"\nTrend: {self.trend.direction}")
            parts.append(f"- Change: {self.trend.change_percent:+.1f}%")
            parts.append(f"- Strength: {self.trend.strength:.2f}")
        
        # Seasonal patterns
        if self.seasonal:
            parts.append(f"\nSeasonality: {self.seasonal.period}")
            parts.append(f"- Pattern strength: {self.seasonal.strength:.2f}")
        
        # Anomalies
        if self.anomalies:
            parts.append(f"\nAnomalies detected: {len(self.anomalies)}")
            for anomaly in self.anomalies[:3]:  # Top 3 anomalies
                parts.append(
                    f"- {anomaly.anomaly_type.value}: "
                    f"{anomaly.deviation_percent:+.0f}% deviation "
                    f"(confidence: {anomaly.confidence:.2f})"
                )
        
        return "\n".join(parts)


# API Request/Response Models

class MetricIngestRequest(BaseModel):
    """Request to ingest metrics from Prometheus"""
    metric_name: str = Field(..., description="Metric to query")
    labels: Optional[Dict[str, str]] = Field(None, description="Label filters")
    start_time: datetime = Field(..., description="Start of time range")
    end_time: datetime = Field(..., description="End of time range")
    step: str = Field(default="1m", description="Query resolution")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metric_name": "cpu_usage",
                "labels": {"service": "payment"},
                "start_time": "2025-10-21T00:00:00Z",
                "end_time": "2025-10-22T00:00:00Z",
                "step": "5m"
            }
        }


class MetricQueryRequest(BaseModel):
    """Request to query metrics semantically"""
    query: str = Field(..., description="Natural language query or metric description")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    filters: Optional[Dict[str, str]] = Field(None, description="Additional filters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "high CPU usage patterns in payment service",
                "top_k": 10,
                "filters": {"service": "payment"}
            }
        }


class MetricAnalyzeRequest(BaseModel):
    """Request to analyze metric trends"""
    metric_name: str
    labels: Optional[Dict[str, str]] = None
    time_range: str = Field(..., description="Time range (e.g., '2w', '24h', '30d')")
    analysis_types: List[str] = Field(
        default=["trend", "anomaly", "seasonal"],
        description="Types of analysis to perform"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "metric_name": "cpu_usage",
                "labels": {"service": "payment"},
                "time_range": "2w",
                "analysis_types": ["trend", "anomaly"]
            }
        }


class MetricAnalyzeResponse(BaseModel):
    """Response from metric analysis"""
    metric: MetricIdentifier
    features: MetricFeatures
    ai_insights: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    prometheus_connected: bool
    ollama_connected: bool
    vector_db_ready: bool
    version: str = "1.0.0"
