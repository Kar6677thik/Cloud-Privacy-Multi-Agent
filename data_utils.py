
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class UniversalAdapter:
    """
    Adapts any raw tabular data for spectral clustering.
    Handles:
    - Missing values (Imputation)
    - Scale differences (Standardization)
    - Categorical Coercion (tries to convert or drops)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def process(self, df):
        """
        Input: Pandas DataFrame
        Output: Numpy Array (processed)
        """
        # 1. Drop completely empty rows/cols
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # 2. Select Numerical Columns
        df_num = df.select_dtypes(include=[np.number])
        
        # Check if we lost everything (maybe data was all strings?)
        if df_num.empty:
            raise ValueError("No numeric data found in dataset.")
            
        data = df_num.values
        
        # 3. Impute Missing Values
        if np.isnan(data).any():
            data = self.imputer.fit_transform(data)
            
        # 4. Standardize (Z-Score)
        # Spectral clustering relies on Euclidean distance. 
        # Large features (Salary) should not dominate small ones (Age).
        data = self.scaler.fit_transform(data)
        
        return data, df_num.columns.tolist()
