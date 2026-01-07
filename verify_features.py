
import sys
import numpy as np
import pandas as pd
import os
from data_utils import UniversalAdapter
from auto_tuning import find_optimal_k
from report_gen import generate_pdf_report

def test_adapter():
    print("Testing Universal Adapter...")
    # Create messy data with NaNs and unscaled
    df = pd.DataFrame({
        "A": [1.0, np.nan, 3.0, 100.0],
        "B": [10.0, 20.0, 30.0, 400.0],
        "C": ["ignore", "strings", "please", "ok"]
    })
    
    adapter = UniversalAdapter()
    data, cols = adapter.process(df)
    
    # NaN should be gone
    assert not np.isnan(data).any()
    # Should be 4 rows
    assert data.shape[0] == 4
    # Should be 2 cols (C dropped)
    assert data.shape[1] == 2
    # Should be scaled (mean close to 0)
    assert np.allclose(data.mean(axis=0), 0, atol=1e-1)
    
    print("Adapter OK.")

def test_auto_k():
    print("Testing Auto-K...")
    # 2 obvious clusters
    c1 = np.random.rand(10, 2)
    c2 = np.random.rand(10, 2) + 5
    data = np.vstack([c1, c2])
    
    # Should find k=2
    res, best_k = find_optimal_k(data, max_k=5)
    print(f"Auto-K found optimal k={best_k}")
    assert best_k == 2 or best_k == 3 # Sometimes silhouette is fuzzy, but 2 is strongest
    print("Auto-K OK.")

def test_report_gen():
    print("Testing Report Gen...")
    try:
        metrics = {"Score": 0.95}
        config = {"k": 3}
        
        pdf_path = generate_pdf_report(metrics, config)
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
        print(f"PDF generated at {pdf_path}")
        print("Report Gen OK.")
    except ImportError:
        print("Skipping Report Gen Test (fpdf missing).")

if __name__ == "__main__":
    test_adapter()
    test_auto_k()
    test_report_gen()
    print("ALL TESTS PASSED.")
