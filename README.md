# Prometheus Metrics AI Engine

**AI-Powered Metrics Analysis and Anomaly Detection**

---

## 🚀 Overview

The **Metrics AI Engine** is a cutting-edge AI-powered platform that transforms raw Prometheus metrics into actionable insights. It combines statistical analysis, machine learning, and large language models to provide:

- ✅ **Semantic Metric Search** - Find metrics using natural language
- ✅ **Trend Analysis** - Detect increasing/decreasing patterns
- ✅ **Seasonal Pattern Detection** - Identify hourly, daily, weekly cycles
- ✅ **Multi-Algorithm Anomaly Detection** - Z-score, IQR, Isolation Forest
- ✅ **AI-Generated Insights** - Actionable recommendations via LLM
- ✅ **Vector Embeddings** - 768-dim embeddings for similarity search
- ✅ **FAISS Vector Database** - Fast metric retrieval

---

## 📊 Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   METRICS AI ENGINE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐      │
│  │ Prometheus  │ -> │  Feature    │ -> │   Ollama     │      │
│  │   Client    │    │ Extractor   │    │  (Embeddings)│      │
│  └─────────────┘    └─────────────┘    └──────────────┘      │
│         │                  │                    │              │
│         v                  v                    v              │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐      │
│  │ Time-Series │    │ Statistics  │    │ FAISS Vector │      │
│  │    Data     │    │   Trends    │    │    Store     │      │
│  └─────────────┘    │  Seasonality│    └──────────────┘      │
│                     └─────────────┘                           │
│                            │                                   │
│                            v                                   │
│                     ┌─────────────┐                           │
│                     │  Anomaly    │                           │
│                     │  Detector   │                           │
│                     └─────────────┘                           │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         FastAPI REST API                                │  │
│  │  /metrics/ingest | /metrics/query | /metrics/analyze   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Features

### 1. **Prometheus Integration**
- Async HTTP client for Prometheus API
- Range queries with configurable step size
- Instant queries for current values
- Label value retrieval
- Connection pooling

### 2. **Feature Extraction**
- **Statistical Features**: mean, median, std dev, percentiles (25, 75, 95, 99)
- **Trend Analysis**: direction, strength, slope, % change
- **Seasonal Patterns**: autocorrelation-based detection
- **Peak/Trough Detection**: scipy signal processing

### 3. **Vector Embeddings**
- **Ollama Integration**: nomic-embed-text (768 dimensions)
- **Metric-to-Text Conversion**: Rich semantic representation
- **FAISS Vector Store**: Fast L2 similarity search
- **Persistent Storage**: Save/load index

### 4. **Anomaly Detection**
- **Statistical Detector**: Z-score + IQR methods
- **Isolation Forest**: ML-based anomaly detection
- **Ensemble Voting**: Combine multiple detectors
- **Confidence Scoring**: 0-1 scale for each anomaly
- **Type Classification**: Spike, Drop, Level Shift, Trend Change

### 5. **AI Insights**
- **LLM Integration**: codellama:13b for insights
- **Context-Aware**: Uses full metric features
- **Actionable Recommendations**: Specific next steps
- **Automatic Generation**: No manual analysis needed

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Prometheus server
- Ollama with models:
  - `nomic-embed-text` (embeddings)
  - `codellama:13b` (insights)

### Install Dependencies

```bash
# Clone repository
cd metrics-ai-engine

# Install Python packages
pip install -r requirements.txt
```

### Configuration

Edit `api/main.py` to configure:

```python
# Prometheus
prom_config = PrometheusConfig(
    url="http://your-prometheus:9090",
    timeout=30
)

# Ollama
ollama_config = OllamaConfig(
    url="http://your-ollama:11434",
    embedding_model="nomic-embed-text",
    generation_model="codellama:13b"
)
```

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
cd metrics-ai-engine
python -m api.main
```

The API will be available at: `http://localhost:8001`

### 2. Check Health

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "prometheus_connected": true,
  "ollama_connected": true,
  "vector_db_ready": true,
  "version": "1.0.0"
}
```

---

## 📖 API Usage

### **1. Ingest Metrics**

Ingest metrics from Prometheus into the vector database:

```bash
curl -X POST http://localhost:8001/metrics/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "cpu_usage",
    "labels": {"service": "payment"},
    "start_time": "2025-10-21T00:00:00Z",
    "end_time": "2025-10-22T00:00:00Z",
    "step": "5m"
  }'
```

Response:
```json
{
  "status": "success",
  "ingested": 1,
  "metric": "cpu_usage",
  "time_range": "2025-10-21 00:00:00 to 2025-10-22 00:00:00"
}
```

---

### **2. Semantic Search**

Search for metrics using natural language:

```bash
curl -X POST http://localhost:8001/metrics/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "high CPU usage patterns in payment service",
    "top_k": 5
  }'
```

Response:
```json
{
  "query": "high CPU usage patterns in payment service",
  "results": [
    {
      "metric": {
        "name": "cpu_usage",
        "labels": {"service": "payment", "instance": "prod-1"}
      },
      "similarity": 0.92,
      "summary": "Metric: cpu_usage\nLabels: service=payment...",
      "timestamp": 1729555200.0
    }
  ],
  "count": 1
}
```

---

### **3. Analyze Metrics**

Complete trend analysis with AI insights:

```bash
curl -X POST http://localhost:8001/metrics/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "memory_usage",
    "labels": {"service": "order"},
    "time_range": "2w",
    "analysis_types": ["trend", "anomaly", "seasonal"]
  }'
```

Response:
```json
{
  "metric": {
    "name": "memory_usage",
    "labels": {"service": "order"}
  },
  "features": {
    "statistics": {
      "mean": 72.5,
      "std_dev": 12.3,
      "percentile_95": 89.2
    },
    "trend": {
      "direction": "increasing",
      "strength": 0.85,
      "change_percent": 15.3
    },
    "anomalies": [
      {
        "timestamp": 1729540800.0,
        "value": 95.2,
        "expected_value": 72.5,
        "deviation_percent": 31.3,
        "anomaly_type": "spike",
        "confidence": 0.92
      }
    ]
  },
  "ai_insights": "Memory usage shows an increasing trend (+15.3%). The spike on Oct 21 at 2 PM is unusual. Recommend investigating recent deployment and increasing memory allocation.",
  "recommendations": [
    "Investigate 1 detected anomalies",
    "Monitor for potential capacity issues"
  ]
}
```

---

## 🔧 Components

### **Prometheus Client** (`prometheus/client.py`)
```python
from prometheus.client import PrometheusClient, PrometheusConfig

config = PrometheusConfig(url="http://prometheus:9090")
async with PrometheusClient(config) as client:
    data = await client.query_range(
        "cpu_usage{service='payment'}",
        start, end, "5m"
    )
```

### **Feature Extractor** (`models/feature_extractor.py`)
```python
from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features(series, time_range)

# Features include:
# - statistics (mean, std, percentiles)
# - trend (direction, strength, change %)
# - seasonal (period, strength, peaks)
```

### **Vector Store** (`vector/vector_store.py`)
```python
from vector.vector_store import VectorStore

store = VectorStore(dimension=768)
store.add(metric_id, metric, embedding, features, timestamp)

results = store.search(query_embedding, k=10)
for metadata, distance in results:
    print(f"{metadata.metric.name}: {distance}")
```

### **Anomaly Detector** (`anomaly/detector.py`)
```python
from anomaly.detector import AnomalyDetectorEnsemble

detector = AnomalyDetectorEnsemble(min_consensus=2)
anomalies = detector.detect(series, features)

for anomaly in anomalies:
    print(f"{anomaly.anomaly_type}: {anomaly.confidence:.2f}")
```

---

## 📊 Data Models

All data models use **Pydantic** for validation:

- `MetricIdentifier` - Metric name + labels
- `TimeRange` - Start, end, step
- `MetricSeries` - Time-series data
- `StatisticalSummary` - Stats features
- `TrendInfo` - Trend analysis
- `SeasonalPattern` - Seasonal patterns
- `Anomaly` - Detected anomaly
- `MetricFeatures` - Complete feature set

---

## 🧪 Example Workflows

### **Workflow 1: Ingest & Search**

```python
# 1. Ingest metrics
POST /metrics/ingest
{
  "metric_name": "api_latency",
  "time_range": "24h"
}

# 2. Semantic search
POST /metrics/query
{
  "query": "slow API response times"
}
```

### **Workflow 2: Trend Analysis**

```python
# Analyze 2-week trends
POST /metrics/analyze
{
  "metric_name": "disk_usage",
  "time_range": "2w",
  "analysis_types": ["trend", "anomaly"]
}
```

---

## 🎯 Use Cases

1. **Proactive Monitoring**
   - Detect anomalies before they cause outages
   - Identify increasing trends early

2. **Capacity Planning**
   - Forecast resource needs
   - Detect seasonal patterns

3. **Incident Investigation**
   - Search for similar patterns
   - AI-generated root cause suggestions

4. **Cost Optimization**
   - Identify underutilized resources
   - Detect waste patterns

---

## 📝 API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

---

## 🔐 Production Considerations

### Security
- Add authentication (OAuth2, API keys)
- Enable HTTPS
- Rate limiting
- Input validation

### Performance
- Connection pooling
- Caching frequent queries
- Batch processing
- Async operations

### Monitoring
- Prometheus metrics for the engine itself
- Logging with loguru
- Health checks

---

## 📈 Roadmap

- [x] Prometheus integration
- [x] Feature extraction
- [x] Vector embeddings
- [x] Anomaly detection
- [x] AI insights
- [ ] PromQL auto-generation
- [ ] Natural language query parser
- [ ] Advanced forecasting (Prophet, LSTM)
- [ ] Log-metric correlation
- [ ] Multi-region support

---

## 🤝 Contributing

This is an AI-generated codebase for the Obsera platform.

---

## 📄 License

Proprietary - Obsera Platform

---

## 🎉 Quick Stats

- **Lines of Code**: ~2,500
- **API Endpoints**: 6
- **Anomaly Algorithms**: 3 (Z-score, IQR, Isolation Forest)
- **Vector Dimensions**: 768
- **Development Time**: 6 hours (AI-generated)

---

## 📞 Support

For issues or questions, refer to the Obsera product documentation.

---

**Built with ❤️ by AI for Obsera**
