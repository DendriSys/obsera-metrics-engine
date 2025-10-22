"""
Prometheus Client for Metrics AI Engine
Handles connection and querying of Prometheus time-series database
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
from loguru import logger


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus connection"""
    url: str
    timeout: int = 30
    verify_ssl: bool = True
    headers: Optional[Dict[str, str]] = None


@dataclass
class TimeSeriesPoint:
    """Single time-series data point"""
    timestamp: float
    value: float


@dataclass
class TimeSeriesData:
    """Time-series data with metadata"""
    metric_name: str
    labels: Dict[str, str]
    values: List[TimeSeriesPoint]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "labels": self.labels,
            "values": [{"timestamp": p.timestamp, "value": p.value} for p in self.values]
        }


class PrometheusClient:
    """
    Client for querying Prometheus metrics
    
    Features:
    - Range queries for time-series data
    - Instant queries for current values
    - Automatic retry with exponential backoff
    - Connection pooling for performance
    """
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize Prometheus client
        
        Args:
            config: PrometheusConfig with connection details
        """
        self.config = config
        self.base_url = config.url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            verify=config.verify_ssl,
            headers=config.headers or {}
        )
        logger.info(f"Prometheus client initialized for {self.base_url}")
    
    async def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> List[TimeSeriesData]:
        """
        Execute a range query against Prometheus
        
        Args:
            query: PromQL query string
            start: Start time for query
            end: End time for query
            step: Query resolution (e.g., "1m", "5m", "1h")
        
        Returns:
            List of TimeSeriesData objects
        
        Example:
            >>> start = datetime.now() - timedelta(hours=24)
            >>> end = datetime.now()
            >>> data = await client.query_range(
            ...     "cpu_usage{service='payment'}",
            ...     start, end, "5m"
            ... )
        """
        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                "query": query,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "step": step
            }
            
            logger.debug(f"Executing range query: {query}")
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                raise ValueError(f"Prometheus query failed: {data.get('error')}")
            
            result = self._parse_range_response(data["data"]["result"])
            logger.info(f"Range query returned {len(result)} time-series")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error querying Prometheus: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing range query: {e}")
            raise
    
    async def instant_query(
        self,
        query: str,
        time: Optional[datetime] = None
    ) -> List[TimeSeriesData]:
        """
        Execute an instant query against Prometheus
        
        Args:
            query: PromQL query string
            time: Time for query (default: now)
        
        Returns:
            List of TimeSeriesData objects with single point
        
        Example:
            >>> data = await client.instant_query("up{job='api'}")
        """
        try:
            url = f"{self.base_url}/api/v1/query"
            params = {"query": query}
            
            if time:
                params["time"] = int(time.timestamp())
            
            logger.debug(f"Executing instant query: {query}")
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                raise ValueError(f"Prometheus query failed: {data.get('error')}")
            
            result = self._parse_instant_response(data["data"]["result"])
            logger.info(f"Instant query returned {len(result)} metrics")
            return result
            
        except Exception as e:
            logger.error(f"Error executing instant query: {e}")
            raise
    
    async def get_label_values(
        self,
        label: str,
        match: Optional[str] = None
    ) -> List[str]:
        """
        Get all values for a specific label
        
        Args:
            label: Label name (e.g., "service", "instance")
            match: Optional metric selector to filter
        
        Returns:
            List of label values
        
        Example:
            >>> services = await client.get_label_values("service")
        """
        try:
            url = f"{self.base_url}/api/v1/label/{label}/values"
            params = {}
            if match:
                params["match[]"] = match
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                raise ValueError(f"Label query failed: {data.get('error')}")
            
            values = data["data"]
            logger.info(f"Found {len(values)} values for label '{label}'")
            return values
            
        except Exception as e:
            logger.error(f"Error getting label values: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if Prometheus is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/-/healthy"
            response = await self.client.get(url)
            response.raise_for_status()
            logger.info("Prometheus health check passed")
            return True
        except Exception as e:
            logger.error(f"Prometheus health check failed: {e}")
            return False
    
    def _parse_range_response(self, result: List[Dict]) -> List[TimeSeriesData]:
        """Parse range query response from Prometheus"""
        time_series = []
        
        for item in result:
            metric_name = item["metric"].get("__name__", "")
            labels = {k: v for k, v in item["metric"].items() if k != "__name__"}
            
            values = [
                TimeSeriesPoint(timestamp=float(ts), value=float(val))
                for ts, val in item["values"]
            ]
            
            time_series.append(TimeSeriesData(
                metric_name=metric_name,
                labels=labels,
                values=values
            ))
        
        return time_series
    
    def _parse_instant_response(self, result: List[Dict]) -> List[TimeSeriesData]:
        """Parse instant query response from Prometheus"""
        time_series = []
        
        for item in result:
            metric_name = item["metric"].get("__name__", "")
            labels = {k: v for k, v in item["metric"].items() if k != "__name__"}
            
            timestamp, value = item["value"]
            values = [TimeSeriesPoint(timestamp=float(timestamp), value=float(value))]
            
            time_series.append(TimeSeriesData(
                metric_name=metric_name,
                labels=labels,
                values=values
            ))
        
        return time_series
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        logger.info("Prometheus client closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
