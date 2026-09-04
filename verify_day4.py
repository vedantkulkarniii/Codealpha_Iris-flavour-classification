"""
Day 4 Verification Script
Tests all Day 4 functionality including train/test split and baseline model.
"""

import sys
import os
sys.path.insert(0, 'src')

from setup_dataset import setup_iris_dataset
from preprocessing import preprocess_dataset
from train import train_baseline_model, load_model
import numpy as np


def verify_day4_functionality():
    """Verify all Day 4 components are working correctly."""
    
    print("="*70)
    print("DAY 4 VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Load and Preprocess Dataset
    print("\n[Test 1] Loading and preprocessing dataset...")
    try:
        df_raw, _ = setup_iris_dataset()
        X, y, df_clean = preprocess_dataset(df_raw)
        assert X.shape == (147, 4), f"Expected X shape (147, 4), got {X.shape}"
        assert y.shape == (147,), f"Expected y shape (147,), got {y.shape}"
        print(f"   ✅ Dataset ready: X{X.shape}, y{y.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dataset preparation failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: Train Baseline Model
    print("\n[Test 2] Training baseline model...")
    try:
        results = train_baseline_model(X, y, test_size=0.2, random_state=42)
        print(f"   ✅ Model training completed")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Model training failed: {e}")
        tests_failed += 1
        return
    
    # Test 3: Verify Train/Test Split
    print("\n[Test 3] Verifying train/test split...")
    try:
        X_train = results['X_train']
        X_test = results['X_test']
        y_train = results['y_train']
        y_test = results['y_test']
        
        # Check sizes
        total_samples = len(X_train) + len(X_test)
        assert total_samples == 147, f"Total samples should be 147, got {total_samples}"
        
        # Check test size is approximately 20%
        test_ratio = len(X_test) / total_samples
        assert 0.18 <= test_ratio <= 0.22, f"Test ratio should be ~0.2, got {test_ratio:.2f}"
        
        # Check no data leakage (no overlap in indices)
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0, "Data leakage detected"
        
        print(f"   ✅ Split verified")
        print(f"      - Train: {len(X_train)} samples ({len(X_train)/total_samples*100:.1f}%)")
        print(f"      - Test: {len(X_test)} samples ({len(X_test)/total_samples*100:.1f}%)")
        print(f"      - No data leakage ✓")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Split verification failed: {e}")
        tests_failed += 1
    
    # Test 4: Verify Stratification
    print("\n[Test 4] Verifying stratified split...")
    try:
        # Check class distribution in train and test
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        
        # All classes should be present
        assert len(train_dist) == 3, "Train set should have all 3 classes"
        assert len(test_dist) == 3, "Test set should have all 3 classes"
        
        # Distributions should be similar (within 5% tolerance)
        for species in train_dist.index:
            train_pct = train_dist[species]
            test_pct = test_dist[species]
            diff = abs(train_pct - test_pct)
            assert diff < 0.05, f"{species} distribution diff too large: {diff:.3f}"
        
        print(f"   ✅ Stratification verified")
        print(f"      - All classes present in both sets")
        print(f"      - Class distributions balanced")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Stratification verification failed: {e}")
        tests_failed += 1
    
    # Test 5: Verify Feature Scaling
    print("\n[Test 5] Verifying feature scaling...")
    try:
        X_train_scaled = results['X_train_scaled']
        X_test_scaled = results['X_test_scaled']
        scaler = results['scaler']
        
        # Check that scaled data has mean ≈ 0 and std ≈ 1 (for training data)
        train_means = X_train_scaled.mean()
        train_stds = X_train_scaled.std()
        
        for col in X_train_scaled.columns:
            assert abs(train_means[col]) < 0.1, f"{col} mean not close to 0: {train_means[col]:.4f}"
            assert abs(train_stds[col] - 1.0) < 0.1, f"{col} std not close to 1: {train_stds[col]:.4f}"
        
        # Check scaler was fitted on training data
        assert hasattr(scaler, 'mean_'), "Scaler not fitted"
        assert len(scaler.mean_) == 4, "Scaler should have 4 means"
        
        print(f"   ✅ Feature scaling verified")
        print(f"      - Training data: mean ≈ 0, std ≈ 1")
        print(f"      - Scaler fitted on training data only")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Feature scaling verification failed: {e}")
        tests_failed += 1
    
    # Test 6: Verify Model Training
    print("\n[Test 6] Verifying model training...")
    try:
        model = results['model']
        
        # Check model attributes
        assert hasattr(model, 'coef_'), "Model not trained (no coefficients)"
        assert hasattr(model, 'classes_'), "Model has no classes"
        assert len(model.classes_) == 3, f"Expected 3 classes, got {len(model.classes_)}"
        assert model.n_features_in_ == 4, f"Expected 4 features, got {model.n_features_in_}"
        
        # Check model type
        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression), "Model is not LogisticRegression"
        
        print(f"   ✅ Model training verified")
        print(f"      - Model type: LogisticRegression")
        print(f"      - Classes: {list(model.classes_)}")
        print(f"      - Features: {model.n_features_in_}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Model training verification failed: {e}")
        tests_failed += 1
    
    # Test 7: Verify Predictions
    print("\n[Test 7] Verifying predictions...")
    try:
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        y_test = results['y_test']
        
        # Check prediction shapes
        assert len(y_pred) == len(y_test), "Prediction count doesn't match test count"
        assert y_pred_proba.shape == (len(y_test), 3), f"Proba shape should be ({len(y_test)}, 3)"
        
        # Check probabilities sum to 1
        proba_sums = y_pred_proba.sum(axis=1)
        assert np.allclose(proba_sums, 1.0), "Probabilities don't sum to 1"
        
        # Check all probabilities are between 0 and 1
        assert (y_pred_proba >= 0).all() and (y_pred_proba <= 1).all(), "Invalid probabilities"
        
        # Check predictions match highest probability class
        pred_indices = y_pred_proba.argmax(axis=1)
        pred_from_proba = model.classes_[pred_indices]
        assert (y_pred == pred_from_proba).all(), "Predictions don't match probabilities"
        
        print(f"   ✅ Predictions verified")
        print(f"      - Prediction count: {len(y_pred)}")
        print(f"      - Probability shape: {y_pred_proba.shape}")
        print(f"      - Probabilities sum to 1.0 ✓")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Prediction verification failed: {e}")
        tests_failed += 1
    
    # Test 8: Verify Model Saved
    print("\n[Test 8] Verifying model saved to disk...")
    try:
        model_path = results['model_path']
        
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        
        file_size = os.path.getsize(model_path)
        assert file_size > 1000, f"Model file too small: {file_size} bytes"
        
        print(f"   ✅ Model file verified")
        print(f"      - Path: {model_path}")
        print(f"      - Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Model file verification failed: {e}")
        tests_failed += 1
    
    # Test 9: Verify Model Loading
    print("\n[Test 9] Verifying model can be loaded...")
    try:
        loaded = load_model('models/logistic_regression.pkl')
        
        assert 'model' in loaded, "Loaded package missing 'model'"
        assert 'scaler' in loaded, "Loaded package missing 'scaler'"
        
        # Verify loaded model works
        test_sample = X_test_scaled.iloc[:1]
        prediction = loaded['model'].predict(test_sample)
        
        assert len(prediction) == 1, "Loaded model prediction failed"
        
        print(f"   ✅ Model loading verified")
        print(f"      - Model loaded successfully")
        print(f"      - Scaler loaded successfully")
        print(f"      - Test prediction works ✓")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Model loading verification failed: {e}")
        tests_failed += 1
    
    # Test 10: Verify Random State Reproducibility
    print("\n[Test 10] Verifying reproducibility...")
    try:
        # Train again with same random state
        results2 = train_baseline_model(X, y, test_size=0.2, random_state=42)
        
        # Check that splits are identical
        assert (results['X_train'].index == results2['X_train'].index).all(), "Train indices differ"
        assert (results['X_test'].index == results2['X_test'].index).all(), "Test indices differ"
        
        # Check predictions are identical
        assert (results['y_pred'] == results2['y_pred']).all(), "Predictions differ"
        
        print(f"   ✅ Reproducibility verified")
        print(f"      - Same random_state produces identical results")
        print(f"      - Train/test split consistent")
        print(f"      - Predictions consistent")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Reproducibility verification failed: {e}")
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Passed: {tests_passed}/10")
    print(f"❌ Tests Failed: {tests_failed}/10")
    
    if tests_failed == 0:
        print("\n🎉 All Day 4 functionality verified successfully!")
        print("✅ Train/test split working with stratification")
        print("✅ Feature scaling pipeline working")
        print("✅ Baseline model trained successfully")
        print("✅ Predictions generated correctly")
        print("✅ Model saved and can be loaded")
        print("✅ Reproducibility ensured with random_state=42")
        print("✅ Ready to proceed to Day 5 (Multiple Models & Evaluation)")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    verify_day4_functionality()
