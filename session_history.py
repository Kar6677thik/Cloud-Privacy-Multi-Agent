"""
Clustering Session History Module
Tracks and allows replay of clustering experiments.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any
import tempfile
import hashlib


class SessionHistory:
    """
    Manages clustering session history with save/load functionality.
    Stores results in session state for Streamlit compatibility.
    """
    
    MAX_HISTORY_SIZE = 20  # Keep last 20 experiments
    
    def __init__(self):
        self.history: List[Dict] = []
        self._current_id = 0
        
    def add_run(self, 
                config: Dict,
                metrics: Dict,
                labels: np.ndarray,
                embedding: np.ndarray,
                data_hash: str,
                notes: str = "") -> int:
        """
        Add a clustering run to history.
        
        Args:
            config: Dict with k, sigma, epsilon, backend, etc.
            metrics: Dict with Silhouette, ARI, etc.
            labels: Cluster assignments
            embedding: Spectral embedding Y
            data_hash: Hash of input data for comparison
            notes: Optional user notes
            
        Returns:
            Run ID
        """
        run_id = self._current_id
        self._current_id += 1
        
        entry = {
            "id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "labels": labels.tolist(),
            "embedding": embedding.tolist(),
            "data_hash": data_hash,
            "notes": notes,
            "n_samples": len(labels),
            "n_clusters": len(set(labels) - {-1})  # Exclude outliers
        }
        
        self.history.append(entry)
        
        # Trim old entries
        if len(self.history) > self.MAX_HISTORY_SIZE:
            self.history = self.history[-self.MAX_HISTORY_SIZE:]
        
        return run_id
    
    def get_run(self, run_id: int) -> Optional[Dict]:
        """Retrieve a specific run by ID."""
        for entry in self.history:
            if entry["id"] == run_id:
                return entry
        return None
    
    def get_all_runs(self) -> List[Dict]:
        """Get all runs in reverse chronological order."""
        return list(reversed(self.history))
    
    def get_run_summary(self, run_id: int) -> str:
        """Get a human-readable summary of a run."""
        entry = self.get_run(run_id)
        if not entry:
            return "Run not found"
        
        config = entry["config"]
        metrics = entry["metrics"]
        
        summary = f"Run #{run_id} @ {entry['timestamp'][:19]}\n"
        summary += f"  k={config.get('k', '?')}, σ={config.get('sigma', '?')}, ε={config.get('epsilon', 'None')}\n"
        summary += f"  Backend: {config.get('backend', 'mock')}\n"
        
        if metrics:
            sil = metrics.get("Silhouette Score", "N/A")
            if isinstance(sil, float):
                summary += f"  Silhouette: {sil:.4f}\n"
            else:
                summary += f"  Silhouette: {sil}\n"
        
        return summary
    
    def compare_runs(self, run_id_1: int, run_id_2: int) -> Dict:
        """Compare two runs and return differences."""
        run1 = self.get_run(run_id_1)
        run2 = self.get_run(run_id_2)
        
        if not run1 or not run2:
            return {"error": "One or both runs not found"}
        
        comparison = {
            "run_1": run_id_1,
            "run_2": run_id_2,
            "config_diff": {},
            "metric_diff": {},
            "same_data": run1["data_hash"] == run2["data_hash"]
        }
        
        # Config differences
        for key in set(run1["config"].keys()) | set(run2["config"].keys()):
            v1 = run1["config"].get(key)
            v2 = run2["config"].get(key)
            if v1 != v2:
                comparison["config_diff"][key] = {"run_1": v1, "run_2": v2}
        
        # Metric differences
        for key in set(run1["metrics"].keys()) | set(run2["metrics"].keys()):
            v1 = run1["metrics"].get(key, 0)
            v2 = run2["metrics"].get(key, 0)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                comparison["metric_diff"][key] = {
                    "run_1": v1,
                    "run_2": v2,
                    "delta": v2 - v1,
                    "improvement": v2 > v1
                }
        
        return comparison
    
    def clear_history(self):
        """Clear all history."""
        self.history = []
        self._current_id = 0
        
    def to_json(self) -> str:
        """Export history as JSON string."""
        return json.dumps(self.history, indent=2)
    
    def from_json(self, json_str: str):
        """Import history from JSON string."""
        self.history = json.loads(json_str)
        if self.history:
            self._current_id = max(h["id"] for h in self.history) + 1


def compute_data_hash(data_dict: dict) -> str:
    """
    Compute a hash of the input data for comparison.
    
    Args:
        data_dict: Dict of {name: numpy_array}
        
    Returns:
        Hash string
    """
    hasher = hashlib.md5()
    
    for name in sorted(data_dict.keys()):
        arr = data_dict[name]
        hasher.update(name.encode())
        hasher.update(arr.tobytes())
    
    return hasher.hexdigest()[:12]


# Singleton for Streamlit session state
_session_history = None

def get_session_history() -> SessionHistory:
    """Get or create the session history singleton."""
    global _session_history
    if _session_history is None:
        _session_history = SessionHistory()
    return _session_history
