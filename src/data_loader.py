"""
Data Loader Module for Iris Dataset
Handles loading and initial inspection of the Iris dataset.
"""

import pandas as pd
import os


def load_dataset(csv_path):
    """
    Load the Iris dataset from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        Exception: If loading fails
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print("✓ Dataset loaded successfully!")
        
        return df
    
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")


def inspect_dataset(df):
    """
    Display comprehensive information about the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to inspect
    """
    print("\n" + "="*60)
    print("DATASET INSPECTION")
    print("="*60)
    
    # Dataset shape
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   - Rows: {df.shape[0]}")
    print(f"   - Columns: {df.shape[1]}")
    
    # Column names
    print(f"\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    # Data types
    print(f"\n🔤 Data Types:")
    print(df.dtypes.to_string())
    
    # First 5 rows
    print(f"\n👀 First 5 Rows:")
    print(df.head().to_string())
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(df.describe().to_string())
    
    # Missing values check
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values found!")
    else:
        print(missing.to_string())
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # This is a placeholder for testing
    # Actual CSV path will be provided by dataset_downloader
    print("This module is designed to be imported.")
    print("Use dataset_downloader.py to download the dataset first.")
