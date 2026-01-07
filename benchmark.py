
import time
import numpy as np
import pandas as pd
from system_sim import simulate_multi_user_system

def run_benchmark(max_users=5, samples_per_user=10, step_size=1, he_type="mock"):
    """
    Runs the system with increasing number of users/data size
    and records execution time.
    
    Args:
        max_users: Maximum number of users to simulate
        samples_per_user: Data points per user
        step_size: User increment step
        he_type: Backend
        
    Returns:
        pd.DataFrame containing results
    """
    results = []
    
    # Features (Iris-like)
    n_features = 4
    
    for n_users in range(1, max_users + 1, step_size):
        # Generate dummy data
        data_owners = {}
        for i in range(n_users):
            # Random data
            data = np.random.rand(samples_per_user, n_features)
            data_owners[f"User_{i}"] = data
            
        start_time = time.time()
        
        try:
            # Run System
            _ = simulate_multi_user_system(data_owners, k=3, sigma=1.0, he_type=he_type)
            duration = time.time() - start_time
            success = True
        except Exception as e:
            duration = time.time() - start_time
            success = False
            print(f"Benchmark failed at {n_users} users: {e}")
            
        total_samples = n_users * samples_per_user
        
        results.append({
            "Users": n_users,
            "Total Samples": total_samples,
            "Time (s)": duration,
            "Success": success,
            "Backend": he_type
        })
        
    return pd.DataFrame(results)
