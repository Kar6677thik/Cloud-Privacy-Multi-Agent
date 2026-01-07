
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def find_optimal_k(data, max_k=10):
    """
    Evaluates clustering performance for k=2..max_k.
    Uses Inertia (Elbow) and Silhouette Score.
    
    Returns:
        results: Dict containing lists of 'k', 'distortion', 'silhouette'
        best_k: Suggested k based on max silhouette
    """
    ks = list(range(2, max_k + 1))
    distortions = []
    silhouettes = []
    
    # Use standard K-Means for speed estimation involved in tuning
    # (Spectral is O(N^3), checking range is expensive)
    # Ideally we use Spectral, but heuristic K-Means on raw data often gives good hint for k
    
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        
        distortions.append(model.inertia_)
        silhouettes.append(silhouette_score(data, labels))
        
    # Heuristic for Best K: Max Silhouette is reliable
    best_k_idx = np.argmax(silhouettes)
    best_k = ks[best_k_idx]
    
    return {
        "k": ks,
        "distortion": distortions,
        "silhouette": silhouettes
    }, best_k
