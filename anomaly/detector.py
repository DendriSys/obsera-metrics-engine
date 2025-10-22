"""
Anomaly Detection for Metrics
Statistical and ML-based anomaly detection algorithms
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest
from loguru import logger

from ..models.metric_data import (
    Anomaly,
    AnomalyType,
    MetricSeries,
    MetricFeatures
)


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using Z-score and IQR methods
    
    Methods:
    - Z-score: Detects points beyond N standard deviations
    - IQR (Interquartile Range): Detects outliers beyond Q1-1.5*IQR and Q3+1.5*IQR
    - Threshold-based: Custom thresholds
    """
    
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_anomaly_confidence: float = 0.6
    ):
        """
        Initialize detector
        
        Args:
            z_score_threshold: Number of std devs for anomaly (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
            min_anomaly_confidence: Minimum confidence to report (default: 0.6)
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_anomaly_confidence = min_anomaly_confidence
        logger.info(
            f"Statistical detector initialized: "
            f"z={z_score_threshold}, iqr={iqr_multiplier}"
        )
    
    def detect(
        self,
        series: MetricSeries,
        features: Optional[MetricFeatures] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies in metric series
        
        Args:
            series: MetricSeries with data points
            features: Optional pre-computed features
        
        Returns:
            List of detected anomalies
        """
        try:
            values = np.array(series.values)
            timestamps = np.array(series.timestamps)
            
            if len(values) < 3:
                logger.warning("Not enough data points for anomaly detection")
                return []
            
            # Remove NaN
            mask = ~np.isnan(values)
            clean_values = values[mask]
            clean_timestamps = timestamps[mask]
            
            if len(clean_values) < 3:
                return []
            
            anomalies = []
            
            # Z-score based detection
            z_anomalies = self._detect_zscore(clean_timestamps, clean_values)
            anomalies.extend(z_anomalies)
            
            # IQR based detection
            iqr_anomalies = self._detect_iqr(clean_timestamps, clean_values)
            anomalies.extend(iqr_anomalies)
            
            # Remove duplicates and sort by timestamp
            unique_anomalies = self._deduplicate_anomalies(anomalies)
            
            # Filter by confidence
            filtered = [
                a for a in unique_anomalies
                if a.confidence >= self.min_anomaly_confidence
            ]
            
            logger.info(f"Detected {len(filtered)} anomalies")
            return filtered
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _detect_zscore(
        self,
        timestamps: np.ndarray,
        values: np.ndarray
    ) -> List[Anomaly]:
        """
        Detect anomalies using Z-score method
        
        Args:
            timestamps: Array of timestamps
            values: Array of values
        
        Returns:
            List of anomalies
        """
        try:
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return []
            
            # Calculate Z-scores
            z_scores = np.abs((values - mean) / std)
            
            # Find anomalies
            anomaly_indices = np.where(z_scores > self.z_score_threshold)[0]
            
            anomalies = []
            for idx in anomaly_indices:
                value = values[idx]
                z_score = z_scores[idx]
                
                # Determine anomaly type
                if value > mean:
                    anomaly_type = AnomalyType.SPIKE
                else:
                    anomaly_type = AnomalyType.DROP
                
                # Calculate confidence based on Z-score
                # Confidence increases with Z-score magnitude
                confidence = min(1.0, z_score / (self.z_score_threshold * 2))
                
                anomaly = Anomaly(
                    timestamp=float(timestamps[idx]),
                    value=float(value),
                    expected_value=float(mean),
                    deviation=float(value - mean),
                    deviation_percent=float(((value - mean) / mean * 100) if mean != 0 else 0),
                    anomaly_type=anomaly_type,
                    confidence=float(confidence),
                    context=f"Z-score: {z_score:.2f}"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in Z-score detection: {e}")
            return []
    
    def _detect_iqr(
        self,
        timestamps: np.ndarray,
        values: np.ndarray
    ) -> List[Anomaly]:
        """
        Detect anomalies using IQR method
        
        Args:
            timestamps: Array of timestamps
            values: Array of values
        
        Returns:
            List of anomalies
        """
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                return []
            
            # Calculate bounds
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            # Find anomalies
            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            
            anomalies = []
            median = np.median(values)
            
            for idx in anomaly_indices:
                value = values[idx]
                
                # Determine anomaly type
                if value > upper_bound:
                    anomaly_type = AnomalyType.SPIKE
                    expected = upper_bound
                else:
                    anomaly_type = AnomalyType.DROP
                    expected = lower_bound
                
                # Calculate confidence based on distance from bounds
                deviation_from_bound = abs(value - expected)
                max_deviation = abs(median - expected)
                confidence = min(1.0, deviation_from_bound / max_deviation) if max_deviation > 0 else 0.8
                
                anomaly = Anomaly(
                    timestamp=float(timestamps[idx]),
                    value=float(value),
                    expected_value=float(median),
                    deviation=float(value - median),
                    deviation_percent=float(((value - median) / median * 100) if median != 0 else 0),
                    anomaly_type=anomaly_type,
                    confidence=float(confidence),
                    context=f"IQR outlier (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in IQR detection: {e}")
            return []
    
    def _deduplicate_anomalies(
        self,
        anomalies: List[Anomaly]
    ) -> List[Anomaly]:
        """
        Remove duplicate anomalies (same timestamp)
        Keep the one with higher confidence
        
        Args:
            anomalies: List of anomalies
        
        Returns:
            Deduplicated list
        """
        if not anomalies:
            return []
        
        # Group by timestamp
        by_timestamp = {}
        for anomaly in anomalies:
            ts = anomaly.timestamp
            if ts not in by_timestamp or anomaly.confidence > by_timestamp[ts].confidence:
                by_timestamp[ts] = anomaly
        
        # Sort by timestamp
        result = sorted(by_timestamp.values(), key=lambda a: a.timestamp)
        return result


class IsolationForestDetector:
    """
    ML-based anomaly detection using Isolation Forest
    
    Isolation Forest is effective for:
    - High-dimensional data
    - Non-parametric anomaly detection
    - Detecting novel patterns
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies (default: 0.1 = 10%)
            n_estimators: Number of trees (default: 100)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
        logger.info(
            f"Isolation Forest initialized: "
            f"contamination={contamination}, trees={n_estimators}"
        )
    
    def detect(
        self,
        series: MetricSeries,
        features: Optional[MetricFeatures] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            series: MetricSeries with data points
            features: Optional pre-computed features
        
        Returns:
            List of detected anomalies
        """
        try:
            values = np.array(series.values)
            timestamps = np.array(series.timestamps)
            
            if len(values) < 10:
                logger.warning("Not enough data for Isolation Forest")
                return []
            
            # Remove NaN
            mask = ~np.isnan(values)
            clean_values = values[mask]
            clean_timestamps = timestamps[mask]
            
            if len(clean_values) < 10:
                return []
            
            # Create features (value + rate of change)
            features_matrix = self._create_features(clean_values)
            
            # Fit and predict
            predictions = self.model.fit_predict(features_matrix)
            anomaly_scores = self.model.score_samples(features_matrix)
            
            # -1 means anomaly, 1 means normal
            anomaly_indices = np.where(predictions == -1)[0]
            
            anomalies = []
            mean = np.mean(clean_values)
            
            for idx in anomaly_indices:
                value = clean_values[idx]
                score = anomaly_scores[idx]
                
                # Determine type
                if value > mean:
                    anomaly_type = AnomalyType.SPIKE
                else:
                    anomaly_type = AnomalyType.DROP
                
                # Convert anomaly score to confidence (more negative = higher confidence)
                confidence = min(1.0, abs(score) / 0.5)
                
                anomaly = Anomaly(
                    timestamp=float(clean_timestamps[idx]),
                    value=float(value),
                    expected_value=float(mean),
                    deviation=float(value - mean),
                    deviation_percent=float(((value - mean) / mean * 100) if mean != 0 else 0),
                    anomaly_type=anomaly_type,
                    confidence=float(confidence),
                    context=f"Isolation Forest score: {score:.3f}"
                )
                anomalies.append(anomaly)
            
            logger.info(f"Isolation Forest detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            return []
    
    def _create_features(self, values: np.ndarray) -> np.ndarray:
        """
        Create feature matrix for Isolation Forest
        
        Features:
        - Current value
        - Rate of change
        - Rolling mean deviation
        
        Args:
            values: Array of values
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        n = len(values)
        features = np.zeros((n, 3))
        
        # Feature 1: Normalized value
        features[:, 0] = (values - np.mean(values)) / (np.std(values) + 1e-10)
        
        # Feature 2: Rate of change
        rate_of_change = np.diff(values, prepend=values[0])
        features[:, 1] = rate_of_change
        
        # Feature 3: Deviation from rolling mean
        window = min(5, n // 3)
        if window > 1:
            rolling_mean = np.convolve(values, np.ones(window)/window, mode='same')
            features[:, 2] = values - rolling_mean
        
        return features


class AnomalyDetectorEnsemble:
    """
    Ensemble of multiple anomaly detectors
    Combines results using consensus voting
    """
    
    def __init__(self, min_consensus: int = 2):
        """
        Initialize ensemble
        
        Args:
            min_consensus: Minimum detectors that must agree (default: 2)
        """
        self.statistical = StatisticalAnomalyDetector()
        self.isolation_forest = IsolationForestDetector()
        self.min_consensus = min_consensus
        logger.info(f"Ensemble initialized with {min_consensus} min consensus")
    
    def detect(
        self,
        series: MetricSeries,
        features: Optional[MetricFeatures] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies using ensemble of detectors
        
        Args:
            series: MetricSeries with data points
            features: Optional pre-computed features
        
        Returns:
            List of anomalies with consensus voting
        """
        try:
            # Run all detectors
            statistical_anomalies = self.statistical.detect(series, features)
            ml_anomalies = self.isolation_forest.detect(series, features)
            
            # Combine and vote
            all_anomalies = statistical_anomalies + ml_anomalies
            
            if not all_anomalies:
                return []
            
            # Group by timestamp (with small tolerance)
            consensus_anomalies = self._vote(all_anomalies)
            
            logger.info(
                f"Ensemble detected {len(consensus_anomalies)} anomalies "
                f"(from {len(all_anomalies)} candidates)"
            )
            
            return consensus_anomalies
            
        except Exception as e:
            logger.error(f"Error in ensemble detection: {e}")
            return []
    
    def _vote(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """
        Consensus voting for anomalies
        
        Args:
            anomalies: All candidate anomalies
        
        Returns:
            Anomalies with sufficient consensus
        """
        if not anomalies:
            return []
        
        # Group by timestamp (tolerance: 60 seconds)
        groups = {}
        for anomaly in anomalies:
            ts = int(anomaly.timestamp / 60) * 60  # Round to minute
            if ts not in groups:
                groups[ts] = []
            groups[ts].append(anomaly)
        
        # Keep groups with enough votes
        result = []
        for ts, group in groups.items():
            if len(group) >= self.min_consensus:
                # Average confidence
                avg_confidence = np.mean([a.confidence for a in group])
                # Use highest confidence anomaly as template
                best = max(group, key=lambda a: a.confidence)
                best.confidence = avg_confidence
                best.context = f"Ensemble ({len(group)} detectors agree)"
                result.append(best)
        
        return sorted(result, key=lambda a: a.timestamp)
