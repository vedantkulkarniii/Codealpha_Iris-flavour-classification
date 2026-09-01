"""
Integrated Dataset Setup Script
Combines dataset downloading, CSV discovery, and initial data loading.
"""

from dataset_downloader import download_iris_dataset, find_csv_file
from data_loader import load_dataset, inspect_dataset


def setup_iris_dataset():
    """
    Complete pipeline to download, locate, and load the Iris dataset.
    
    Returns:
        tuple: (dataframe, csv_path)
    """
    try:
        print("="*60)
        print("IRIS DATASET SETUP PIPELINE")
        print("="*60)
        
        # Step 1: Download dataset
        print("\n[Step 1/3] Downloading dataset...")
        dataset_path = download_iris_dataset()
        
        # Step 2: Find CSV file
        print("\n[Step 2/3] Locating CSV file...")
        csv_filename, csv_path = find_csv_file(dataset_path)
        
        # Step 3: Load dataset
        print("\n[Step 3/3] Loading dataset...")
        df = load_dataset(csv_path)
        
        # Inspect dataset
        inspect_dataset(df)
        
        print("\n✅ Dataset setup completed successfully!")
        print(f"Dataset ready for analysis with {df.shape[0]} samples")
        
        return df, csv_path
    
    except Exception as e:
        print(f"\n❌ Dataset setup failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        df, csv_path = setup_iris_dataset()
        
        # Additional summary
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        print(f"✓ Dataset Path: {csv_path}")
        print(f"✓ Total Samples: {df.shape[0]}")
        print(f"✓ Total Features: {df.shape[1]}")
        print(f"✓ Columns: {', '.join(df.columns.tolist())}")
        print("="*60)
        
    except Exception as e:
        print(f"\nSetup failed. Please check the error above.")
