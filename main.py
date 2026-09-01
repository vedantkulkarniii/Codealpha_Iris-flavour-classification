"""
Main entry point for the Iris Flower Classification Project
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from setup_dataset import setup_iris_dataset


def main():
    """Main function to run the Iris classification pipeline."""
    
    print("="*70)
    print("IRIS FLOWER SPECIES CLASSIFICATION")
    print("="*70)
    print("\nDay 1: Project Setup & Dataset\n")
    
    try:
        # Setup and load dataset
        df, csv_path = setup_iris_dataset()
        
        print("\n✅ Day 1 Complete!")
        print("   Dataset successfully downloaded and loaded")
        print(f"   Ready for Day 2: Data Understanding & Cleaning")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error message above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
