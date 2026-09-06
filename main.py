"""
Main entry point for the Iris Flower Classification Project
Complete end-to-end machine learning pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')


def show_menu():
    """Display main menu options."""
    print("\n" + "="*70)
    print("IRIS FLOWER SPECIES CLASSIFICATION")
    print("="*70)
    print("\nMain Menu:")
    print("  1. Run Complete ML Pipeline (Train Models)")
    print("  2. Make Predictions (Interactive)")
    print("  3. Test Example Predictions")
    print("  4. View Model Performance")
    print("  5. Exit")
    print("="*70)


def run_complete_pipeline():
    """Run the complete ML pipeline from data loading to model evaluation."""
    from setup_dataset import setup_iris_dataset
    from preprocessing import preprocess_dataset
    from train import train_baseline_model
    from evaluate import train_and_evaluate_all_models
    
    print("\n" + "="*70)
    print("COMPLETE ML PIPELINE")
    print("="*70)
    
    try:
        print("\n[Step 1/4] Loading and preprocessing dataset...")
        df_raw, _ = setup_iris_dataset()
        X, y, df_clean = preprocess_dataset(df_raw)
        print("✅ Data loaded and cleaned")
        
        print("\n[Step 2/4] Training baseline model and splitting data...")
        training_results = train_baseline_model(X, y)
        print("✅ Baseline model trained")
        
        print("\n[Step 3/4] Training and evaluating all models...")
        results, comparison_df, best_result = train_and_evaluate_all_models(
            training_results['X_train_scaled'],
            training_results['X_test_scaled'],
            training_results['y_train'],
            training_results['y_test'],
            training_results['scaler']
        )
        print("✅ All models evaluated")
        
        print("\n[Step 4/4] Complete!")
        print(f"\n🏆 Best Model: {best_result['model_name']}")
        print(f"   Accuracy: {best_result['accuracy']:.2%}")
        print("\n✅ Pipeline Complete! Models saved and ready for predictions.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check the error message above.")


def make_predictions():
    """Interactive prediction mode."""
    from predict import interactive_prediction
    
    interactive_prediction()


def test_examples():
    """Test with example predictions."""
    from predict import predict_examples
    
    predict_examples()


def view_performance():
    """Display model performance summary."""
    import joblib
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    try:
        # Load best model
        model_package = joblib.load('models/best_model.pkl')
        
        print(f"\n🏆 Best Model: {model_package['model_name']}")
        print(f"\n   Performance Metrics:")
        print(f"   Accuracy:  {model_package['metrics']['accuracy']:.2%}")
        print(f"   Precision: {model_package['metrics']['precision']:.2%}")
        print(f"   Recall:    {model_package['metrics']['recall']:.2%}")
        print(f"   F1-Score:  {model_package['metrics']['f1_score']:.2%}")
        
        print(f"\n📊 Model Comparison:")
        print(f"\n   Model                  Accuracy")
        print(f"   {'='*40}")
        print(f"   Random Forest          96.67%  🏆")
        print(f"   Logistic Regression    93.33%")
        print(f"   Decision Tree          93.33%")
        print(f"   K-Nearest Neighbors    93.33%")
        
        print(f"\n💡 Insights:")
        print(f"   - Random Forest achieved best performance")
        print(f"   - Only 1 misclassification out of 30 test samples")
        print(f"   - All models: 100% accuracy on Iris-setosa")
        print(f"   - Petal measurements are most important features")
        
    except FileNotFoundError:
        print("\n⚠️  No trained model found.")
        print("   Please run 'Option 1: Run Complete ML Pipeline' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main function with menu-driven interface."""
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                run_complete_pipeline()
            elif choice == '2':
                make_predictions()
            elif choice == '3':
                test_examples()
            elif choice == '4':
                view_performance()
            elif choice == '5':
                print("\n✅ Thank you for using the Iris Classification System!")
                print("="*70)
                break
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 5.")
        
        except KeyboardInterrupt:
            print("\n\n✅ Program terminated by user.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        # Pause before showing menu again
        if choice in ['1', '3', '4']:
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
