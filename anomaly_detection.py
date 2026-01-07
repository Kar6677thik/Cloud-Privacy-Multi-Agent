"""
Anomaly-Aware Clustering Module
Extends spectral clustering with outlier detection.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist


class AnomalyAwareClustering:
    """
    Wrapper that adds outlier detection to spectral clustering results.
    Outliers are assigned to a special "noise" cluster (-1).
    """
    
    def __init__(self, method: str = "lof", contamination: float = 0.1):
        """
        Args:
            method: 'lof' (Local Outlier Factor) or 'iforest' (Isolation Forest)
            contamination: Expected proportion of outliers (0.0 to 0.5)
        """
        self.method = method
        self.contamination = min(max(contamination, 0.01), 0.5)
        self.outlier_mask = None
        self.outlier_scores = None
        
    def detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers in the dataset.
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            
        Returns:
            Boolean mask where True = outlier
        """
        if self.method == "lof":
            detector = LocalOutlierFactor(
                n_neighbors=min(20, len(data) - 1),
                contamination=self.contamination
            )
            # LOF returns -1 for outliers, 1 for inliers
            predictions = detector.fit_predict(data)
            self.outlier_scores = -detector.negative_outlier_factor_
            
        elif self.method == "iforest":
            detector = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            predictions = detector.fit_predict(data)
            self.outlier_scores = -detector.score_samples(data)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.outlier_mask = predictions == -1
        return self.outlier_mask
    
    def adjust_labels(self, original_labels: np.ndarray, data: np.ndarray) -> tuple:
        """
        Adjust cluster labels by marking outliers as -1 (noise cluster).
        
        Args:
            original_labels: Original cluster assignments
            data: Original data for outlier detection
            
        Returns:
            (adjusted_labels, outlier_info_dict)
        """
        # Detect outliers if not already done
        if self.outlier_mask is None:
            self.detect_outliers(data)
        
        # Create adjusted labels
        adjusted_labels = original_labels.copy()
        adjusted_labels[self.outlier_mask] = -1  # Mark outliers as noise
        
        # Compute outlier statistics
        n_outliers = np.sum(self.outlier_mask)
        n_total = len(original_labels)
        
        # Distribution of outliers across original clusters
        outlier_distribution = {}
        for orig_cluster in np.unique(original_labels):
            mask = (original_labels == orig_cluster) & self.outlier_mask
            outlier_distribution[int(orig_cluster)] = int(np.sum(mask))
        
        info = {
            "n_outliers": n_outliers,
            "outlier_percentage": float(n_outliers / n_total * 100),
            "outlier_indices": np.where(self.outlier_mask)[0].tolist(),
            "outlier_distribution": outlier_distribution,
            "detection_method": self.method,
            "contamination": self.contamination
        }
        
        return adjusted_labels, info
    
    def get_outlier_severity(self) -> np.ndarray:
        """
        Returns outlier severity scores (higher = more anomalous).
        Only available after calling detect_outliers.
        """
        if self.outlier_scores is None:
            return None
        return self.outlier_scores


class ClusteringWithAnomalies:
    """
    Complete pipeline that integrates anomaly detection with clustering.
    """
    
    def __init__(self, enable_anomaly: bool = True, 
                 anomaly_method: str = "lof",
                 contamination: float = 0.1):
        self.enable = enable_anomaly
        self.anomaly_detector = AnomalyAwareClustering(anomaly_method, contamination) if enable_anomaly else None
        self.outlier_info = None
        
    def process(self, labels: np.ndarray, data: np.ndarray) -> tuple:
        """
        Process clustering results with optional anomaly detection.
        
        Args:
            labels: Original cluster assignments from spectral clustering
            data: Original data or spectral embedding
            
        Returns:
            (final_labels, outlier_info or None)
        """
        if not self.enable or self.anomaly_detector is None:
            return labels, None
        
        adjusted_labels, info = self.anomaly_detector.adjust_labels(labels, data)
        self.outlier_info = info
        
        return adjusted_labels, info
    
    def get_cluster_summary(self, labels: np.ndarray) -> dict:
        """Get summary of clusters including outliers."""
        unique_labels = np.unique(labels)
        
        summary = {
            "n_clusters": len(unique_labels[unique_labels >= 0]),
            "has_outliers": -1 in labels,
            "cluster_sizes": {}
        }
        
        for label in unique_labels:
            if label == -1:
                summary["cluster_sizes"]["outliers"] = int(np.sum(labels == label))
            else:
                summary["cluster_sizes"][f"cluster_{label}"] = int(np.sum(labels == label))
        
        return summary
