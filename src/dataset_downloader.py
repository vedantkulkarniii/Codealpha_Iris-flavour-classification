"""
Iris Dataset Downloader using KaggleHub
Downloads the Iris dataset from Kaggle and locates the CSV file automatically.
"""

import os
import kagglehub


def download_iris_dataset():
    """
    Download the Iris dataset from Kaggle using KaggleHub.
    
    Returns:
        str: Path to the downloaded dataset directory
    
    Raises:
        Exception: If dataset download fails
    """
    try:
        print("Downloading Iris dataset from Kaggle...")
        path = kagglehub.dataset_download("saurabh00007/iriscsv")
        print(f"✓ Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        return path
    except Exception as e:
        raise Exception(f"Failed to download dataset: {str(e)}")


def find_csv_file(dataset_path):
    """
    Automatically locate the CSV file in the downloaded dataset directory.
    
    Args:
        dataset_path (str): Path to the dataset directory
    
    Returns:
        tuple: (csv_filename, full_csv_path)
    
    Raises:
        FileNotFoundError: If no CSV file is found
    """
    try:
        # List all files in the dataset directory
        files = os.listdir(dataset_path)
        print(f"\nFiles in dataset directory: {files}")
        
        # Find CSV files
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the dataset directory")
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found: {csv_files}")
            print(f"Using the first one: {csv_files[0]}")
        
        csv_filename = csv_files[0]
        full_csv_path = os.path.join(dataset_path, csv_filename)
        
        print(f"\n✓ CSV file located: {csv_filename}")
        print(f"Full path: {full_csv_path}")
        
        return csv_filename, full_csv_path
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error locating CSV file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error while finding CSV: {str(e)}")


if __name__ == "__main__":
    try:
        # Download dataset
        dataset_path = download_iris_dataset()
        
        # Find CSV file
        csv_filename, csv_path = find_csv_file(dataset_path)
        
        print("\n" + "="*50)
        print("Dataset Download Summary")
        print("="*50)
        print(f"Dataset Directory: {dataset_path}")
        print(f"CSV Filename: {csv_filename}")
        print(f"CSV Path: {csv_path}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
