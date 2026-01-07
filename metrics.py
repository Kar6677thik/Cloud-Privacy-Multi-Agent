
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
import numpy as np

def calculate_metrics(X, labels, true_labels=None):
    """
    Calculates clustering quality metrics.
    
    Args:
        X: The data points (or embedding) used for clustering.
        labels: The predicted cluster labels.
        true_labels: Ground truth labels (if available).
        
    Returns:
        dict: Dictionary of metric names and values.
    """
    metrics = {}
    
    # Internal Metrics (No ground truth needed)
    if len(set(labels)) > 1: # Silhouette needs > 1 cluster
        try:
            metrics["Silhouette Score"] = silhouette_score(X, labels)
            metrics["Davies-Bouldin Index"] = davies_bouldin_score(X, labels)
        except Exception as e:
            metrics["Silhouette Score"] = float('nan')
            metrics["Davies-Bouldin Index"] = float('nan')
    
    # External Metrics (Ground truth needed)
    if true_labels is not None:
        metrics["Adjusted Rand Index (ARI)"] = adjusted_rand_score(true_labels, labels)
        
    return metrics
