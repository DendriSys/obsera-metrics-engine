"""
Feature Extraction for Metrics
Extracts statistical, temporal, and pattern features from time-series data
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
from loguru import logger

from .metric_data import (
    MetricSeries,
    StatisticalSummary,
    TrendInfo,
    SeasonalPattern,
    MetricFeatures,
    TimeRange
)


class FeatureExtractor:
    """
    Extract features from metric time-series data
    
    Features extracted:
    - Statistical: mean, median, std, percentiles
    - Trend: direction, strength, slope
    - Seasonal: patterns, peaks, troughs
    """
    
    def extract_features(
        self,
        series: MetricSeries,
        time_range: TimeRange
    ) -> MetricFeatures:
        """
        Extract all features from a metric series
        
        Args:
            series: MetricSeries with data points
            time_range: Time range for the data
        
        Returns:
            MetricFeatures with complete feature set
        """
        try:
            values = np.array(series.values)
            
            if len(values) == 0:
                raise ValueError("Empty metric series")
            
            logger.debug(f"Extracting features from {len(values)} data points")
            
            # Extract statistical features
            statistics = self._extract_statistics(values)
            
            # Extract trend
            trend = self._extract_trend(values)
            
            # Extract seasonal patterns
            seasonal = self._extract_seasonal(values)
            
            features = MetricFeatures(
                metric=series.metric,
                time_range=time_range,
                statistics=statistics,
                trend=trend,
                seasonal=seasonal,
                anomalies=[]  # Will be filled by anomaly detector
            )
            
            logger.info(f"Feature extraction complete for {series.metric.name}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_statistics(self, values: np.ndarray) -> StatisticalSummary:
        """
        Extract statistical summary from values
        
        Args:
            values: Numpy array of metric values
        
        Returns:
            StatisticalSummary with statistical features
        """
        try:
            # Remove NaN values
            clean_values = values[~np.isnan(values)]
            
            if len(clean_values) == 0:
                raise ValueError("All values are NaN")
            
            summary = StatisticalSummary(
                count=len(clean_values),
                mean=float(np.mean(clean_values)),
                median=float(np.median(clean_values)),
                std_dev=float(np.std(clean_values)),
                min_value=float(np.min(clean_values)),
                max_value=float(np.max(clean_values)),
                percentile_25=float(np.percentile(clean_values, 25)),
                percentile_75=float(np.percentile(clean_values, 75)),
                percentile_95=float(np.percentile(clean_values, 95)),
                percentile_99=float(np.percentile(clean_values, 99))
            )
            
            logger.debug(
                f"Statistics: mean={summary.mean:.2f}, "
                f"std={summary.std_dev:.2f}, "
                f"range=[{summary.min_value:.2f}, {summary.max_value:.2f}]"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error extracting statistics: {e}")
            raise
    
    def _extract_trend(self, values: np.ndarray) -> Optional[TrendInfo]:
        """
        Extract trend information using linear regression
        
        Args:
            values: Numpy array of metric values
        
        Returns:
            TrendInfo or None if trend cannot be determined
        """
        try:
            if len(values) < 3:
                return None
            
            # Remove NaN
            clean_values = values[~np.isnan(values)]
            if len(clean_values) < 3:
                return None
            
            # Linear regression
            x = np.arange(len(clean_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_values)
            
            # Determine direction
            threshold = np.std(clean_values) * 0.1  # 10% of std dev
            if abs(slope) < threshold:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            # Calculate percentage change
            start_value = clean_values[0]
            end_value = clean_values[-1]
            if start_value != 0:
                change_percent = ((end_value - start_value) / abs(start_value)) * 100
            else:
                change_percent = 0.0
            
            trend = TrendInfo(
                direction=direction,
                strength=abs(r_value),  # R-squared shows trend strength
                slope=float(slope),
                change_percent=float(change_percent)
            )
            
            logger.debug(
                f"Trend: {direction}, "
                f"strength={trend.strength:.2f}, "
                f"change={change_percent:+.1f}%"
            )
            
            return trend
            
        except Exception as e:
            logger.warning(f"Could not extract trend: {e}")
            return None
    
    def _extract_seasonal(
        self,
        values: np.ndarray,
        min_period: int = 12
    ) -> Optional[SeasonalPattern]:
        """
        Extract seasonal patterns using autocorrelation
        
        Args:
            values: Numpy array of metric values
            min_period: Minimum period to detect (default: 12 points)
        
        Returns:
            SeasonalPattern or None if no pattern detected
        """
        try:
            if len(values) < min_period * 2:
                return None
            
            clean_values = values[~np.isnan(values)]
            if len(clean_values) < min_period * 2:
                return None
            
            # Detrend the data
            x = np.arange(len(clean_values))
            slope, intercept = np.polyfit(x, clean_values, 1)
            detrended = clean_values - (slope * x + intercept)
            
            # Find peaks and troughs
            peaks, peak_properties = find_peaks(detrended, distance=min_period)
            troughs, trough_properties = find_peaks(-detrended, distance=min_period)
            
            # Check if we have enough peaks/troughs for seasonality
            if len(peaks) < 2 and len(troughs) < 2:
                return None
            
            # Calculate period
            if len(peaks) >= 2:
                peak_diffs = np.diff(peaks)
                avg_period = int(np.mean(peak_diffs))
            elif len(troughs) >= 2:
                trough_diffs = np.diff(troughs)
                avg_period = int(np.mean(trough_diffs))
            else:
                return None
            
            # Calculate strength (variance explained by seasonality)
            seasonal_variance = np.var(detrended)
            total_variance = np.var(clean_values)
            strength = float(seasonal_variance / total_variance) if total_variance > 0 else 0.0
            
            # Determine period name
            if avg_period <= 60:  # Assume 1 min resolution
                period_name = "hourly"
            elif avg_period <= 288:  # 5 min resolution, 24h
                period_name = "daily"
            elif avg_period <= 2016:  # 5 min resolution, 7 days
                period_name = "weekly"
            else:
                period_name = "monthly"
            
            seasonal = SeasonalPattern(
                period=period_name,
                strength=min(strength, 1.0),
                peaks=peaks.tolist(),
                troughs=troughs.tolist()
            )
            
            logger.debug(
                f"Seasonal pattern: {period_name}, "
                f"strength={strength:.2f}, "
                f"peaks={len(peaks)}, troughs={len(troughs)}"
            )
            
            return seasonal
            
        except Exception as e:
            logger.warning(f"Could not extract seasonal pattern: {e}")
            return None


class MetricAggregator:
    """
    Aggregate metrics over time windows
    """
    
    @staticmethod
    def rolling_average(
        values: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Calculate rolling average
        
        Args:
            values: Array of values
            window: Window size
        
        Returns:
            Array of rolling averages
        """
        if len(values) < window:
            return values
        
        return np.convolve(values, np.ones(window)/window, mode='valid')
    
    @staticmethod
    def rolling_std(
        values: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Calculate rolling standard deviation
        
        Args:
            values: Array of values
            window: Window size
        
        Returns:
            Array of rolling std devs
        """
        if len(values) < window:
            return np.array([np.std(values)])
        
        result = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i+window]
            result.append(np.std(window_values))
        
        return np.array(result)
    
    @staticmethod
    def resample(
        timestamps: np.ndarray,
        values: np.ndarray,
        target_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample time-series to fixed number of points
        
        Args:
            timestamps: Array of timestamps
            values: Array of values
            target_points: Desired number of points
        
        Returns:
            Tuple of (resampled_timestamps, resampled_values)
        """
        if len(values) <= target_points:
            return timestamps, values
        
        indices = np.linspace(0, len(values) - 1, target_points, dtype=int)
        return timestamps[indices], values[indices]
