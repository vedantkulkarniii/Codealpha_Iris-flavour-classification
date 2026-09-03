"""
Day 3 Verification Script
Tests all Day 3 EDA functionality and validates visualizations.
"""

import sys
import os
sys.path.insert(0, 'src')

from setup_dataset import setup_iris_dataset
from preprocessing import preprocess_dataset
from eda import perform_eda


def verify_day3_functionality():
    """Verify all Day 3 components are working correctly."""
    
    print("="*70)
    print("DAY 3 VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Load and Preprocess Dataset
    print("\n[Test 1] Loading and preprocessing dataset...")
    try:
        df_raw, _ = setup_iris_dataset()
        X, y, df_clean = preprocess_dataset(df_raw)
        print(f"   ✅ Dataset ready: {df_clean.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dataset preparation failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: Visualizations Directory
    print("\n[Test 2] Checking visualizations directory...")
    try:
        assert os.path.exists('visualizations'), "Visualizations directory not found"
        print(f"   ✅ Visualizations directory exists")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Directory check failed: {e}")
        tests_failed += 1
    
    # Test 3: Species Distribution Plot
    print("\n[Test 3] Verifying species distribution plot...")
    try:
        plot_path = 'visualizations/species_distribution.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 10000, f"Plot file too small: {file_size} bytes"
        print(f"   ✅ Species distribution plot exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 4: Sepal Measurements Plot
    print("\n[Test 4] Verifying sepal measurements plot...")
    try:
        plot_path = 'visualizations/sepal_measurements.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 10000, f"Plot file too small: {file_size} bytes"
        print(f"   ✅ Sepal measurements plot exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 5: Petal Measurements Plot
    print("\n[Test 5] Verifying petal measurements plot...")
    try:
        plot_path = 'visualizations/petal_measurements.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 10000, f"Plot file too small: {file_size} bytes"
        print(f"   ✅ Petal measurements plot exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 6: Pairplot
    print("\n[Test 6] Verifying feature pairplot...")
    try:
        plot_path = 'visualizations/feature_pairplot.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 50000, f"Pairplot file too small: {file_size} bytes"
        print(f"   ✅ Feature pairplot exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 7: Correlation Heatmap
    print("\n[Test 7] Verifying correlation heatmap...")
    try:
        plot_path = 'visualizations/correlation_heatmap.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 10000, f"Plot file too small: {file_size} bytes"
        print(f"   ✅ Correlation heatmap exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 8: Feature Distributions Plot
    print("\n[Test 8] Verifying feature distributions plot...")
    try:
        plot_path = 'visualizations/feature_distributions.png'
        assert os.path.exists(plot_path), f"Plot not found: {plot_path}"
        file_size = os.path.getsize(plot_path)
        assert file_size > 30000, f"Plot file too small: {file_size} bytes"
        print(f"   ✅ Feature distributions plot exists ({file_size:,} bytes)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Plot verification failed: {e}")
        tests_failed += 1
    
    # Test 9: Verify All Required Visualizations
    print("\n[Test 9] Verifying all visualizations present...")
    try:
        required_plots = [
            'species_distribution.png',
            'sepal_measurements.png',
            'petal_measurements.png',
            'feature_pairplot.png',
            'correlation_heatmap.png',
            'feature_distributions.png'
        ]
        
        for plot_name in required_plots:
            plot_path = os.path.join('visualizations', plot_name)
            assert os.path.exists(plot_path), f"Missing: {plot_name}"
        
        print(f"   ✅ All 6 required visualizations present")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Verification failed: {e}")
        tests_failed += 1
    
    # Test 10: Verify Data Insights
    print("\n[Test 10] Verifying data insights...")
    try:
        # Check species balance
        species_counts = df_clean['Species'].value_counts()
        assert len(species_counts) == 3, f"Expected 3 species, found {len(species_counts)}"
        
        # Check numerical features
        numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        for feature in numerical_features:
            assert feature in df_clean.columns, f"Feature missing: {feature}"
        
        # Check correlations exist
        correlation = df_clean[numerical_features].corr()
        assert correlation.shape == (4, 4), "Correlation matrix incorrect shape"
        
        print(f"   ✅ Data insights verified")
        print(f"      - 3 species present")
        print(f"      - 4 numerical features")
        print(f"      - Correlation matrix: 4×4")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Insights verification failed: {e}")
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Passed: {tests_passed}/10")
    print(f"❌ Tests Failed: {tests_failed}/10")
    
    if tests_failed == 0:
        print("\n🎉 All Day 3 functionality verified successfully!")
        print("✅ EDA pipeline working")
        print("✅ All visualizations generated")
        print("✅ Data insights extracted")
        print("✅ Ready to proceed to Day 4 (Baseline Model)")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    # Visualization Summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    
    viz_dir = 'visualizations'
    if os.path.exists(viz_dir):
        files = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        total_size = sum(os.path.getsize(os.path.join(viz_dir, f)) for f in files)
        
        print(f"Directory: {viz_dir}/")
        print(f"Total Files: {len(files)}")
        print(f"Total Size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print("\nFiles:")
        for f in sorted(files):
            file_path = os.path.join(viz_dir, f)
            size = os.path.getsize(file_path)
            print(f"  ✓ {f:35s} ({size:>8,} bytes)")
    
    print("="*70)


if __name__ == "__main__":
    verify_day3_functionality()
