"""
Integration Test for Metrics AI Engine with Ollama
Tests the complete workflow: Prometheus → Features → Embeddings → Search
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prometheus.client import PrometheusClient, PrometheusConfig
from models.metric_data import MetricIdentifier, MetricSeries, MetricPoint, TimeRange
from models.feature_extractor import FeatureExtractor
from vector.ollama_client import OllamaClient, OllamaConfig
from vector.scalable_vector_store import ScalableVectorStore
from anomaly.detector import AnomalyDetectorEnsemble


async def test_ollama_connection():
    """Test 1: Verify Ollama connection"""
    print("\n" + "="*70)
    print("TEST 1: Ollama Connection")
    print("="*70)
    
    config = OllamaConfig(
        url="http://localhost:11434",
        embedding_model="nomic-embed-text",
        generation_model="codellama:13b"
    )
    
    async with OllamaClient(config) as client:
        # Test health
        health = await client.health_check()
        print(f"✅ Ollama health check: {health}")
        
        # Test embedding generation
        test_text = "CPU usage metric showing high values with increasing trend"
        embedding = await client.generate_embedding(test_text)
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        
        # Test AI insights
        context = """
        Metric: cpu_usage
        Service: payment-api
        Statistics: mean=85%, std=12%, max=98%
        Trend: increasing +15% over 2 weeks
        Anomalies: 2 spikes detected
        """
        insights = await client.generate_insights(context)
        print(f"✅ Generated insights ({len(insights)} chars)")
        print(f"\nAI Insights:\n{insights[:200]}...\n")
        
        return health


async def test_vector_store_scalability():
    """Test 2: Verify scalable vector store"""
    print("\n" + "="*70)
    print("TEST 2: Scalable Vector Store")
    print("="*70)
    
    store = ScalableVectorStore(
        dimension=768,
        storage_path="/tmp/test_vector_store",
        auto_optimize=True
    )
    
    # Initial stats
    print(f"Initial stats: {store.stats()}")
    
    # Add test vectors
    print("\nAdding test vectors...")
    for i in range(50):
        test_embedding = np.random.randn(768).tolist()
        
        # Create dummy metric and features
        metric = MetricIdentifier(
            name=f"test_metric_{i}",
            labels={"service": "test", "instance": f"inst-{i}"}
        )
        
        from models.metric_data import MetricFeatures, StatisticalSummary
        features = MetricFeatures(
            metric=metric,
            time_range=TimeRange(
                start=datetime.now() - timedelta(hours=24),
                end=datetime.now(),
                step="5m"
            ),
            statistics=StatisticalSummary(
                count=100,
                mean=50.0 + i,
                median=50.0,
                std_dev=10.0,
                min_value=30.0,
                max_value=70.0,
                percentile_25=45.0,
                percentile_75=55.0,
                percentile_95=65.0,
                percentile_99=68.0
            )
        )
        
        store.add(
            metric_id=f"test_{i}",
            metric=metric,
            embedding=test_embedding,
            features=features,
            timestamp=datetime.now().timestamp()
        )
    
    print(f"✅ Added 50 vectors")
    
    # Search test
    query_embedding = np.random.randn(768).tolist()
    results = store.search(query_embedding, k=5)
    print(f"✅ Search returned {len(results)} results")
    
    # Final stats
    stats = store.stats()
    print(f"\nFinal stats:")
    print(f"  - Total vectors: {stats['total_vectors']}")
    print(f"  - Index type: {stats['index_type']}")
    print(f"  - Dimension: {stats['dimension']}")
    
    return True


async def test_feature_extraction():
    """Test 3: Feature extraction from time-series"""
    print("\n" + "="*70)
    print("TEST 3: Feature Extraction")
    print("="*70)
    
    # Create sample time-series data
    metric = MetricIdentifier(name="cpu_usage", labels={"service": "payment"})
    
    # Generate sample data with trend and seasonality
    np.random.seed(42)
    timestamps = np.arange(0, 288) # 24 hours at 5-min intervals
    base = 50
    trend = timestamps * 0.05  # Increasing trend
    seasonal = 10 * np.sin(2 * np.pi * timestamps / 288)  # Daily pattern
    noise = np.random.randn(288) * 2
    values = base + trend + seasonal + noise
    
    points = [
        MetricPoint(timestamp=float(t), value=float(v))
        for t, v in zip(timestamps, values)
    ]
    
    series = MetricSeries(metric=metric, points=points)
    
    # Extract features
    extractor = FeatureExtractor()
    time_range = TimeRange(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        step="5m"
    )
    
    features = extractor.extract_features(series, time_range)
    
    print(f"✅ Statistics extracted:")
    print(f"  - Mean: {features.statistics.mean:.2f}")
    print(f"  - Std Dev: {features.statistics.std_dev:.2f}")
    print(f"  - Min/Max: {features.statistics.min_value:.2f} / {features.statistics.max_value:.2f}")
    
    if features.trend:
        print(f"\n✅ Trend detected:")
        print(f"  - Direction: {features.trend.direction}")
        print(f"  - Strength: {features.trend.strength:.2f}")
        print(f"  - Change: {features.trend.change_percent:+.1f}%")
    
    if features.seasonal:
        print(f"\n✅ Seasonal pattern detected:")
        print(f"  - Period: {features.seasonal.period}")
        print(f"  - Strength: {features.seasonal.strength:.2f}")
        print(f"  - Peaks: {len(features.seasonal.peaks)}")
    
    return features


async def test_anomaly_detection():
    """Test 4: Anomaly detection"""
    print("\n" + "="*70)
    print("TEST 4: Anomaly Detection")
    print("="*70)
    
    # Create sample data with anomalies
    metric = MetricIdentifier(name="memory_usage", labels={"service": "order"})
    
    np.random.seed(42)
    values = np.random.randn(200) * 5 + 50
    
    # Inject anomalies
    values[50] = 95  # Spike
    values[100] = 10  # Drop
    values[150] = 92  # Another spike
    
    points = [
        MetricPoint(timestamp=float(i), value=float(v))
        for i, v in enumerate(values)
    ]
    
    series = MetricSeries(metric=metric, points=points)
    
    # Detect anomalies
    detector = AnomalyDetectorEnsemble(min_consensus=2)
    anomalies = detector.detect(series)
    
    print(f"✅ Detected {len(anomalies)} anomalies")
    
    for i, anomaly in enumerate(anomalies[:5], 1):
        print(f"\nAnomaly {i}:")
        print(f"  - Type: {anomaly.anomaly_type.value}")
        print(f"  - Value: {anomaly.value:.2f}")
        print(f"  - Expected: {anomaly.expected_value:.2f}")
        print(f"  - Deviation: {anomaly.deviation_percent:+.1f}%")
        print(f"  - Confidence: {anomaly.confidence:.2f}")
    
    return len(anomalies) > 0


async def test_end_to_end():
    """Test 5: Complete end-to-end workflow"""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Workflow")
    print("="*70)
    
    # Step 1: Extract features
    print("\n1. Extracting features...")
    features = await test_feature_extraction()
    
    # Step 2: Generate embedding
    print("\n2. Generating embedding...")
    config = OllamaConfig(url="http://localhost:11434")
    async with OllamaClient(config) as client:
        text = features.to_text()
        print(f"   Text representation ({len(text)} chars):")
        print(f"   {text[:200]}...")
        
        embedding = await client.generate_embedding(text)
        print(f"   ✅ Embedding: {len(embedding)} dimensions")
        
        # Step 3: Store in vector database
        print("\n3. Storing in vector database...")
        store = ScalableVectorStore(
            dimension=768,
            storage_path="/tmp/test_e2e_store"
        )
        
        store.add(
            metric_id="test_metric_1",
            metric=features.metric,
            embedding=embedding,
            features=features,
            timestamp=datetime.now().timestamp()
        )
        
        print(f"   ✅ Stored successfully")
        
        # Step 4: Search
        print("\n4. Searching for similar metrics...")
        query_text = "CPU usage with increasing trend"
        query_embedding = await client.generate_embedding(query_text)
        
        results = store.search(query_embedding, k=3)
        print(f"   ✅ Found {len(results)} similar metrics")
        
        for i, (metadata, distance) in enumerate(results, 1):
            similarity = 1.0 / (1.0 + distance)
            print(f"\n   Result {i}:")
            print(f"     - Metric: {metadata.metric.name}")
            print(f"     - Similarity: {similarity:.2%}")
            print(f"     - Summary: {metadata.features_summary[:100]}...")
    
    print("\n✅ End-to-end workflow complete!")
    return True


async def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("METRICS AI ENGINE - INTEGRATION TESTS")
    print("="*70)
    
    results = {}
    
    try:
        # Test 1: Ollama
        results['ollama'] = await test_ollama_connection()
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        results['ollama'] = False
    
    try:
        # Test 2: Vector Store
        results['vector_store'] = await test_vector_store_scalability()
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        results['vector_store'] = False
    
    try:
        # Test 3: Feature Extraction
        features = await test_feature_extraction()
        results['feature_extraction'] = features is not None
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        results['feature_extraction'] = False
    
    try:
        # Test 4: Anomaly Detection
        results['anomaly_detection'] = await test_anomaly_detection()
    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        results['anomaly_detection'] = False
    
    try:
        # Test 5: End-to-End
        results['end_to_end'] = await test_end_to_end()
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        results['end_to_end'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for production!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
