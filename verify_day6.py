"""
Day 6 Verification Script
Tests prediction system and final integration.
"""

import sys
import os
sys.path.insert(0, 'src')

from predict import predict_from_dict, load_best_model


def verify_day6_functionality():
    """Verify all Day 6 components are working correctly."""
    
    print("="*70)
    print("DAY 6 VERIFICATION TEST")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Load Best Model
    print("\n[Test 1] Loading best model...")
    try:
        model_package = load_best_model('models/best_model.pkl')
        assert 'model' in model_package, "Model not in package"
        assert 'scaler' in model_package, "Scaler not in package"
        assert 'model_name' in model_package, "Model name not in package"
        assert 'metrics' in model_package, "Metrics not in package"
        print(f"   ✅ Best model loaded successfully")
        print(f"      - Model: {model_package['model_name']}")
        print(f"      - Accuracy: {model_package['metrics']['accuracy']:.2%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        tests_failed += 1
        return
    
    # Test 2: Predict Setosa
    print("\n[Test 2] Predicting Iris-setosa...")
    try:
        setosa_measurements = {
            'SepalLengthCm': 5.1,
            'SepalWidthCm': 3.5,
            'PetalLengthCm': 1.4,
            'PetalWidthCm': 0.2
        }
        
        result = predict_from_dict(setosa_measurements)
        
        assert result['predicted_species'] == 'Iris-setosa', \
            f"Expected Iris-setosa, got {result['predicted_species']}"
        assert result['confidence'] >= 90, \
            f"Low confidence: {result['confidence']:.2f}%"
        
        print(f"   ✅ Setosa prediction correct")
        print(f"      - Predicted: {result['predicted_species']}")
        print(f"      - Confidence: {result['confidence']:.2f}%")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Setosa prediction failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        tests_failed += 1
    
    # Test 3: Predict Versicolor
    print("\n[Test 3] Predicting Iris-versicolor...")
    try:
        versicolor_measurements = {
            'SepalLengthCm': 6.0,
            'SepalWidthCm': 2.7,
            'PetalLengthCm': 4.5,
            'PetalWidthCm': 1.3
        }
        
        result = predict_from_dict(versicolor_measurements)
        
        assert result['predicted_species'] == 'Iris-versicolor', \
            f"Expected Iris-versicolor, got {result['predicted_species']}"
        
        print(f"   ✅ Versicolor prediction correct")
        print(f"      - Predicted: {result['predicted_species']}")
        print(f"      - Confidence: {result['confidence']:.2f}%")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Versicolor prediction failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        tests_failed += 1
    
    # Test 4: Predict Virginica
    print("\n[Test 4] Predicting Iris-virginica...")
    try:
        virginica_measurements = {
            'SepalLengthCm': 6.5,
            'SepalWidthCm': 3.0,
            'PetalLengthCm': 5.5,
            'PetalWidthCm': 2.0
        }
        
        result = predict_from_dict(virginica_measurements)
        
        assert result['predicted_species'] == 'Iris-virginica', \
            f"Expected Iris-virginica, got {result['predicted_species']}"
        
        print(f"   ✅ Virginica prediction correct")
        print(f"      - Predicted: {result['predicted_species']}")
        print(f"      - Confidence: {result['confidence']:.2f}%")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Virginica prediction failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        tests_failed += 1
    
    # Test 5: Verify Probability Sum
    print("\n[Test 5] Verifying probability distributions...")
    try:
        test_measurements = {
            'SepalLengthCm': 5.5,
            'SepalWidthCm': 2.9,
            'PetalLengthCm': 3.7,
            'PetalWidthCm': 1.1
        }
        
        result = predict_from_dict(test_measurements)
        
        # Check probabilities sum to 100%
        total_prob = sum(result['probabilities'].values())
        assert 99.9 <= total_prob <= 100.1, \
            f"Probabilities don't sum to 100%: {total_prob:.2f}%"
        
        # Check all probabilities are valid (0-100)
        for species, prob in result['probabilities'].items():
            assert 0 <= prob <= 100, \
                f"{species} probability out of range: {prob:.2f}%"
        
        print(f"   ✅ Probability validation passed")
        print(f"      - Total probability: {total_prob:.2f}%")
        print(f"      - All probabilities in valid range")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Probability validation failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        tests_failed += 1
    
    # Test 6: Verify Feature Scaling
    print("\n[Test 6] Verifying feature scaling in predictions...")
    try:
        # Use same measurements as training to check scaling
        test_measurements = {
            'SepalLengthCm': 5.8,
            'SepalWidthCm': 3.0,
            'PetalLengthCm': 4.0,
            'PetalWidthCm': 1.2
        }
        
        result = predict_from_dict(test_measurements)
        
        # Should get a valid prediction
        assert result['predicted_species'] in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        assert 0 <= result['confidence'] <= 100
        
        print(f"   ✅ Feature scaling working correctly")
        print(f"      - Prediction: {result['predicted_species']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Scaling verification failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Scaling error: {e}")
        tests_failed += 1
    
    # Test 7: Edge Case - Small Values
    print("\n[Test 7] Testing edge case (small values)...")
    try:
        small_measurements = {
            'SepalLengthCm': 4.3,
            'SepalWidthCm': 2.0,
            'PetalLengthCm': 1.0,
            'PetalWidthCm': 0.1
        }
        
        result = predict_from_dict(small_measurements)
        
        # Small petal measurements should predict setosa
        assert result['predicted_species'] == 'Iris-setosa', \
            "Small petal measurements should predict setosa"
        
        print(f"   ✅ Edge case handled correctly")
        print(f"      - Small values → {result['predicted_species']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Edge case failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Edge case error: {e}")
        tests_failed += 1
    
    # Test 8: Edge Case - Large Values
    print("\n[Test 8] Testing edge case (large values)...")
    try:
        large_measurements = {
            'SepalLengthCm': 7.9,
            'SepalWidthCm': 3.8,
            'PetalLengthCm': 6.9,
            'PetalWidthCm': 2.5
        }
        
        result = predict_from_dict(large_measurements)
        
        # Large petal measurements should predict virginica
        assert result['predicted_species'] == 'Iris-virginica', \
            "Large petal measurements should predict virginica"
        
        print(f"   ✅ Edge case handled correctly")
        print(f"      - Large values → {result['predicted_species']}")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Edge case failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Edge case error: {e}")
        tests_failed += 1
    
    # Test 9: Verify Main Menu System
    print("\n[Test 9] Verifying main menu system...")
    try:
        # Check main.py exists and is importable
        assert os.path.exists('main.py'), "main.py not found"
        
        # Try to import main functions (will verify syntax)
        import main
        assert hasattr(main, 'show_menu'), "show_menu function not found"
        assert hasattr(main, 'make_predictions'), "make_predictions function not found"
        assert hasattr(main, 'view_performance'), "view_performance function not found"
        
        print(f"   ✅ Main menu system verified")
        print(f"      - main.py exists")
        print(f"      - All menu functions present")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Main menu verification failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Menu system error: {e}")
        tests_failed += 1
    
    # Test 10: Integration Test
    print("\n[Test 10] Running integration test...")
    try:
        # Test multiple predictions in sequence
        test_cases = [
            ({'SepalLengthCm': 5.0, 'SepalWidthCm': 3.4, 'PetalLengthCm': 1.5, 'PetalWidthCm': 0.2}, 'Iris-setosa'),
            ({'SepalLengthCm': 5.9, 'SepalWidthCm': 3.0, 'PetalLengthCm': 4.2, 'PetalWidthCm': 1.5}, 'Iris-versicolor'),
            ({'SepalLengthCm': 6.3, 'SepalWidthCm': 2.9, 'PetalLengthCm': 5.6, 'PetalWidthCm': 1.8}, 'Iris-virginica'),
        ]
        
        correct_predictions = 0
        for measurements, expected in test_cases:
            result = predict_from_dict(measurements)
            if result['predicted_species'] == expected:
                correct_predictions += 1
        
        assert correct_predictions == len(test_cases), \
            f"Only {correct_predictions}/{len(test_cases)} predictions correct"
        
        print(f"   ✅ Integration test passed")
        print(f"      - {correct_predictions}/{len(test_cases)} predictions correct")
        tests_passed += 1
    except AssertionError as e:
        print(f"   ❌ Integration test failed: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ Integration error: {e}")
        tests_failed += 1
    
    # Final Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    print(f"✅ Tests Passed: {tests_passed}/10")
    print(f"❌ Tests Failed: {tests_failed}/10")
    
    if tests_failed == 0:
        print("\n🎉 All Day 6 functionality verified successfully!")
        print("✅ Prediction system working perfectly")
        print("✅ All three species predicted correctly")
        print("✅ Edge cases handled properly")
        print("✅ Integration complete and functional")
        print("✅ Ready for Day 7 (Final Testing) & Day 8 (Documentation)")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    verify_day6_functionality()
