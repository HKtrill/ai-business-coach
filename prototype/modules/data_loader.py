# =============================================================================
# /content/churn_pipeline/modules/data_loader.py (FIXED)
# =============================================================================
import pandas as pd
import hashlib

class DataLoader:
    """Handles loading of raw data and drops the customerID identifier."""
    
    def __init__(self):
        self.raw_data_hash = None
    
    def compute_data_hash(self, df):
        """Compute hash of raw data for version tracking"""
        # Ensure a consistent hash by sorting the DataFrame before hashing
        df_sorted = df.sort_index(axis=1).sort_values(by=list(df.columns), ignore_index=True)
        data_string = df_sorted.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()[:12]
    
    def load_raw_data(self, filepath):
        """
        Load data from CSV and drop the customerID column.
        """
        print("=== LOADING RAW DATA ===")
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        
        # === THIS IS THE CRITICAL LINE THAT WAS MISSING ===
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
        # ==================================================
        
        # ============================================
        # DEBUGGING: Print the head of the raw data
        # ============================================
        print("\n=== HEAD OF RAW DATA ===")
        print(df.head().to_string())
        print("=========================\n")
        # ============================================
        
        # Compute and store data hash for tracking
        self.raw_data_hash = self.compute_data_hash(df)
        
        print(f"Data shape after cleaning: {df.shape}")
        print(f"Data hash: {self.raw_data_hash}")
        
        return df