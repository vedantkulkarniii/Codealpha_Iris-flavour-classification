"""
Day 2 Verification Script
Tests all Day 2 functionality including data validation and preprocessing.
"""

import sys
sys.path.insert(0, 'src')

from setup_dataset import setup_iris_dataset
from data_validation import validate_dataset
from preprocessing import preprocess_dataset


def verify_day2_functionality():
    """Verify all Day 2 components are working correctly."""
    
    print("="*70)
    print("DAY 2 VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Load Dataset
    print("\n[Test 1] Loading dataset...")
    try:
        df, csv_path = setup_iris_dataset()
        print(f"   ✅ Dataset loaded: {df.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: Data Validation
    print("\n[Test 2] Running data validation...")
    try:
        results = validate_dataset(df)
        print(f"   ✅ Validation complete")
        print(f"      - Missing values: {results['missing_values'].sum()}")
        print(f"      - Duplicates: {results['num_duplicates']}")
        print(f"      - Features: {len(results['features'])}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        tests_failed += 1
    
    # Test 3: Data Preprocessing
    print("\n[Test 3] Testing preprocessing pipeline...")
    try:
        X, y, df_clean = preprocess_dataset(df)
        print(f"   ✅ Preprocessing successful")
        print(f"      - Original: {df.shape}")
        print(f"      - Cleaned: {df_clean.shape}")
        print(f"      - Features (X): {X.shape}")
        print(f"      - Target (y): {y.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Preprocessing failed: {e}")
        tests_failed += 1
        return
    
    # Test 4: Verify Id Column Removed
    print("\n[Test 4] Verifying Id column removal...")
    try:
        assert 'Id' not in df_clean.columns, "Id column still present"
        print(f"   ✅ Id column removed successfully")
        print(f"      - Columns: {', '.join(df_clean.columns)}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Id removal failed: {e}")
        tests_failed += 1
    
    # Test 5: Verify Duplicates Removed
    print("\n[Test 5] Verifying duplicate removal...")
    try:
        assert len(df_clean) == 147, f"Expected 147 records, got {len(df_clean)}"
        assert df_clean.duplicated().sum() == 0, "Duplicates still present"
        print(f"   ✅ Duplicates removed successfully")
        print(f"      - Records: 150 → 147")
        print(f"      - Removed: 3 duplicates")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Duplicate removal verification failed: {e}")
        tests_failed += 1
    
    # Test 6: Verify Feature-Target Separation
    print("\n[Test 6] Verifying feature-target separation...")
    try:
        expected_features = ['SepalLengthCm', 'SepalWidthCm', 
                           'PetalLengthCm', 'PetalWidthCm']
        assert list(X.columns) == expected_features, "Feature columns don't match"
        assert X.shape == (147, 4), f"Expected X shape (147, 4), got {X.shape}"
        assert y.shape == (147,), f"Expected y shape (147,), got {y.shape}"
        assert y.name == 'Species', f"Expected target name 'Species', got {y.name}"
        
        print(f"   ✅ Feature-target separation verified")
        print(f"      - Features: {list(X.columns)}")
        print(f"      - Target: {y.name}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Separation verification failed: {e}")
        tests_failed += 1
    
    # Test 7: Verify Data Types
    print("\n[Test 7] Verifying data types...")
    try:
        for col in X.columns:
            assert X[col].dtype == 'float64', f"{col} is not float64"
        assert y.dtype == 'object', "Target is not object type"
        
        print(f"   ✅ Data types verified")
        print(f"      - All features: float64")
        print(f"      - Target: object")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Data type verification failed: {e}")
        tests_failed += 1
    
    # Test 8: Verify No Missing Values
    print("\n[Test 8] Verifying no missing values...")
    try:
        assert X.isnull().sum().sum() == 0, "Missing values in features"
        assert y.isnull().sum() == 0, "Missing values in target"
        
        print(f"   ✅ No missing values")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Missing value check failed: {e}")
        tests_failed += 1
    
    # Test 9: Verify Class Distribution
    print("\n[Test 9] Verifying class distribution...")
    try:
        class_counts = y.value_counts()
        assert len(class_counts) == 3, f"Expected 3 classes, got {len(class_counts)}"
        
        # Check each class has reasonable number of samples
        for species, count in class_counts.items():
            assert count >= 48 and count <= 50, f"{species} has unusual count: {count}"
        
        print(f"   ✅ Class distribution verified")
        for species, count in class_counts.sort_index().items():
            print(f"      - {species}: {count} samples")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Class distribution verification failed: {e}")
        tests_failed += 1
    
    # Test 10: Verify Feature Ranges
    print("\n[Test 10] Verifying feature value ranges...")
    try:
        # Check that features have reasonable ranges based on known Iris dataset
        assert X['SepalLengthCm'].min() >= 4.0, "SepalLengthCm min out of range"
        assert X['SepalLengthCm'].max() <= 8.0, "SepalLengthCm max out of range"
        assert X['SepalWidthCm'].min() >= 2.0, "SepalWidthCm min out of range"
        assert X['PetalLengthCm'].min() >= 1.0, "PetalLengthCm min out of range"
        assert X['PetalWidthCm'].min() >= 0.1, "PetalWidthCm min out of range"
        
        print(f"   ✅ Feature ranges verified")
        for col in X.columns:
            print(f"      - {col}: [{X[col].min():.1f}, {X[col].max():.1f}]")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Feature range verification failed: {e}")
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Passed: {tests_passed}/10")
    print(f"❌ Tests Failed: {tests_failed}/10")
    
    if tests_failed == 0:
        print("\n🎉 All Day 2 functionality verified successfully!")
        print("✅ Data validation pipeline working")
        print("✅ Preprocessing pipeline working")
        print("✅ Data quality assured")
        print("✅ Ready to proceed to Day 3 (EDA)")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    verify_day2_functionality()
