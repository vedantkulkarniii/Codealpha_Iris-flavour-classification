"""
Day 1 Verification Script
Tests all Day 1 functionality to ensure proper setup.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from dataset_downloader import download_iris_dataset, find_csv_file
from data_loader import load_dataset, inspect_dataset


def verify_day1_setup():
    """Verify all Day 1 components are working correctly."""
    
    print("="*70)
    print("DAY 1 VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Dataset Download
    print("\n[Test 1] Testing dataset download...")
    try:
        dataset_path = download_iris_dataset()
        print("   ✅ Dataset download successful")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dataset download failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: CSV File Discovery
    print("\n[Test 2] Testing CSV file discovery...")
    try:
        csv_filename, csv_path = find_csv_file(dataset_path)
        print(f"   ✅ CSV file found: {csv_filename}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ CSV discovery failed: {e}")
        tests_failed += 1
        return
    
    # Test 3: Dataset Loading
    print("\n[Test 3] Testing dataset loading...")
    try:
        df = load_dataset(csv_path)
        print(f"   ✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        tests_failed += 1
        return
    
    # Test 4: Verify Dataset Contents
    print("\n[Test 4] Verifying dataset contents...")
    try:
        expected_columns = ['Id', 'SepalLengthCm', 'SepalWidthCm', 
                          'PetalLengthCm', 'PetalWidthCm', 'Species']
        
        assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
        assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
        assert list(df.columns) == expected_columns, "Column names don't match"
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
        
        print("   ✅ Dataset structure verified")
        print(f"      - 150 samples ✓")
        print(f"      - 6 columns ✓")
        print(f"      - No missing values ✓")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Dataset verification failed: {e}")
        tests_failed += 1
    
    # Test 5: Feature and Target Identification
    print("\n[Test 5] Identifying features and target...")
    try:
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        target = 'Species'
        
        species = df[target].unique()
        assert len(species) == 3, f"Expected 3 species, found {len(species)}"
        
        print("   ✅ Features and target identified")
        print(f"      - Features: {', '.join(features)}")
        print(f"      - Target: {target}")
        print(f"      - Classes: {', '.join(species)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Feature/target identification failed: {e}")
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Passed: {tests_passed}/5")
    print(f"❌ Tests Failed: {tests_failed}/5")
    
    if tests_failed == 0:
        print("\n🎉 All Day 1 functionality verified successfully!")
        print("✅ Ready to proceed to Day 2")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    verify_day1_setup()
