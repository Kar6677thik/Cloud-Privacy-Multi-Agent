"""
AI-Powered Dataset Insights Module
Provides pre-clustering analysis and recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.spatial.distance import pdist
import warnings

warnings.filterwarnings('ignore')


class DatasetInsights:
    """
    Analyzes dataset characteristics before clustering.
    Provides insights on structure, outliers, and recommendations.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.insights = {}
        
    def analyze_all(self) -> dict:
        """Run all analyses and return comprehensive insights."""
        self.insights = {
            "basic_stats": self._basic_statistics(),
            "cluster_tendency": self._hopkins_statistic(),
            "outliers": self._detect_outliers(),
            "feature_correlation": self._feature_analysis(),
            "density_estimation": self._density_analysis(),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        self.insights["recommendations"] = self._generate_recommendations()
        
        return self.insights
    
    def _basic_statistics(self) -> dict:
        """Compute basic dataset statistics."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "mean_per_feature": np.mean(self.data, axis=0).tolist(),
            "std_per_feature": np.std(self.data, axis=0).tolist(),
            "min_per_feature": np.min(self.data, axis=0).tolist(),
            "max_per_feature": np.max(self.data, axis=0).tolist(),
            "has_missing": bool(np.isnan(self.data).any()),
            "variance_ratio": float(np.var(self.data) / (np.mean(np.abs(self.data)) + 1e-10))
        }
    
    def _hopkins_statistic(self) -> dict:
        """
        Calculate Hopkins statistic to assess cluster tendency.
        Values close to 1 indicate clusterable data.
        Values close to 0.5 indicate random/uniform data.
        """
        if self.n_samples < 10:
            return {"hopkins": 0.5, "interpretation": "Too few samples for reliable Hopkins test"}
        
        try:
            n = min(self.n_samples // 5, 50)  # Sample size
            
            # Random sample from data
            sample_idx = np.random.choice(self.n_samples, n, replace=False)
            sample = self.data[sample_idx]
            
            # Generate random points in same space
            mins = np.min(self.data, axis=0)
            maxs = np.max(self.data, axis=0)
            random_points = np.random.uniform(mins, maxs, (n, self.n_features))
            
            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(self.data)
            
            # Distance from random points to nearest neighbor in data
            u_distances, _ = nn.kneighbors(random_points)
            u = np.sum(u_distances[:, 1])  # Skip self
            
            # Distance from sample points to nearest neighbor in data
            w_distances, _ = nn.kneighbors(sample)
            w = np.sum(w_distances[:, 1])
            
            hopkins = u / (u + w + 1e-10)
            
            if hopkins > 0.75:
                interpretation = "Strong cluster tendency - data has clear structure"
            elif hopkins > 0.5:
                interpretation = "Moderate cluster tendency - some structure present"
            else:
                interpretation = "Low cluster tendency - data may be uniform/random"
                
            return {"hopkins": float(hopkins), "interpretation": interpretation}
            
        except Exception as e:
            return {"hopkins": 0.5, "interpretation": f"Could not compute: {str(e)}"}
    
    def _detect_outliers(self) -> dict:
        """Detect outliers using multiple methods."""
        outliers = {}
        
        # Method 1: Z-score based
        z_scores = np.abs(zscore(self.data, axis=0))
        zscore_outliers = np.any(z_scores > 3, axis=1)
        outliers["zscore_count"] = int(np.sum(zscore_outliers))
        outliers["zscore_indices"] = np.where(zscore_outliers)[0].tolist()[:20]  # Limit to 20
        
        # Method 2: IQR based
        Q1 = np.percentile(self.data, 25, axis=0)
        Q3 = np.percentile(self.data, 75, axis=0)
        IQR = Q3 - Q1
        iqr_outliers = np.any((self.data < Q1 - 1.5 * IQR) | (self.data > Q3 + 1.5 * IQR), axis=1)
        outliers["iqr_count"] = int(np.sum(iqr_outliers))
        outliers["iqr_indices"] = np.where(iqr_outliers)[0].tolist()[:20]
        
        # Combined (union)
        combined = zscore_outliers | iqr_outliers
        outliers["combined_count"] = int(np.sum(combined))
        outliers["outlier_percentage"] = float(np.sum(combined) / self.n_samples * 100)
        
        return outliers
    
    def _feature_analysis(self) -> dict:
        """Analyze feature correlations and importance."""
        if self.n_features < 2:
            return {"correlation_matrix": None, "high_correlations": []}
        
        # Correlation matrix
        corr_matrix = np.corrcoef(self.data.T)
        
        # Find highly correlated pairs (|r| > 0.8)
        high_corr = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr.append({
                        "feature_1": i,
                        "feature_2": j,
                        "correlation": float(corr_matrix[i, j])
                    })
        
        # PCA for variance explanation
        pca = PCA()
        pca.fit(self.data)
        
        return {
            "correlation_matrix": corr_matrix.tolist(),
            "high_correlations": high_corr,
            "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components_95_var": int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1)
        }
    
    def _density_analysis(self) -> dict:
        """Analyze data density and distribution."""
        try:
            # K-distance graph for DBSCAN eps estimation
            k = min(5, self.n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(self.data)
            distances, _ = nn.kneighbors(self.data)
            
            # Average k-th neighbor distance
            k_distances = np.sort(distances[:, -1])
            
            # Estimate density variation
            density_variation = float(np.std(k_distances) / (np.mean(k_distances) + 1e-10))
            
            # Suggested sigma for Gaussian kernel (median heuristic)
            pairwise_dists = pdist(self.data[:min(100, self.n_samples)])  # Sample for efficiency
            suggested_sigma = float(np.median(pairwise_dists))
            
            return {
                "avg_neighbor_distance": float(np.mean(k_distances)),
                "density_variation": density_variation,
                "suggested_sigma": suggested_sigma,
                "density_interpretation": "High variation" if density_variation > 1 else "Uniform density"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations based on analysis."""
        recs = []
        
        # Cluster tendency recommendation
        hopkins = self.insights.get("cluster_tendency", {}).get("hopkins", 0.5)
        if hopkins > 0.7:
            recs.append({
                "type": "success",
                "message": "Data shows strong natural clustering tendency. Spectral clustering should work well!"
            })
        elif hopkins < 0.5:
            recs.append({
                "type": "warning",
                "message": "Low cluster tendency detected. Consider if clustering is appropriate for this data."
            })
        
        # Outlier recommendations
        outlier_pct = self.insights.get("outliers", {}).get("outlier_percentage", 0)
        if outlier_pct > 10:
            recs.append({
                "type": "warning",
                "message": f"High outlier rate ({outlier_pct:.1f}%). Consider enabling Anomaly-Aware mode."
            })
        elif outlier_pct > 5:
            recs.append({
                "type": "info",
                "message": f"{outlier_pct:.1f}% outliers detected. Results may be affected."
            })
        
        # Feature correlation recommendations
        high_corr = self.insights.get("feature_correlation", {}).get("high_correlations", [])
        if len(high_corr) > 0:
            recs.append({
                "type": "info",
                "message": f"{len(high_corr)} highly correlated feature pair(s) found. Consider PCA preprocessing."
            })
        
        # Sigma recommendation
        suggested_sigma = self.insights.get("density_estimation", {}).get("suggested_sigma", 1.0)
        recs.append({
            "type": "tip",
            "message": f"Suggested Gaussian sigma based on data density: {suggested_sigma:.2f}"
        })
        
        # Sample size recommendations
        if self.n_samples < 30:
            recs.append({
                "type": "warning",
                "message": "Small sample size. Clustering results may be unstable."
            })
        elif self.n_samples > 500:
            recs.append({
                "type": "info",
                "message": "Large dataset. Computation may take longer with real HE backend."
            })
        
        return recs


def get_insights_summary(data: np.ndarray) -> dict:
    """
    Convenience function to get dataset insights.
    
    Args:
        data: numpy array
        
    Returns:
        dict with all insights
    """
    analyzer = DatasetInsights(data)
    return analyzer.analyze_all()
