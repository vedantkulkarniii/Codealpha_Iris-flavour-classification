"""
Prediction Module
Interactive prediction system for new iris flower measurements.
"""

import pandas as pd
import numpy as np
import joblib
import os


def load_best_model(model_path='models/best_model.pkl'):
    """
    Load the best trained model with scaler.
    
    Args:
        model_path (str): Path to saved model
    
    Returns:
        dict: Model package with model, scaler, and metrics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model_package = joblib.load(model_path)
    
    print(f"✓ Model loaded successfully")
    print(f"  Model: {model_package['model_name']}")
    print(f"  Accuracy: {model_package['metrics']['accuracy']:.2%}")
    
    return model_package


def get_user_input():
    """
    Get flower measurements from user input.
    
    Returns:
        dict: Dictionary with feature measurements
    """
    print("\n" + "="*70)
    print("IRIS FLOWER PREDICTION")
    print("="*70)
    print("\nPlease enter the flower measurements in centimeters:")
    print("(Typical ranges: Sepal Length 4-8, Sepal Width 2-5,")
    print("                 Petal Length 1-7, Petal Width 0-3)")
    print()
    
    measurements = {}
    
    while True:
        try:
            measurements['SepalLengthCm'] = float(input("Enter Sepal Length (cm): "))
            if measurements['SepalLengthCm'] <= 0:
                raise ValueError("Sepal Length must be positive")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive number.")
    
    while True:
        try:
            measurements['SepalWidthCm'] = float(input("Enter Sepal Width (cm): "))
            if measurements['SepalWidthCm'] <= 0:
                raise ValueError("Sepal Width must be positive")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive number.")
    
    while True:
        try:
            measurements['PetalLengthCm'] = float(input("Enter Petal Length (cm): "))
            if measurements['PetalLengthCm'] <= 0:
                raise ValueError("Petal Length must be positive")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive number.")
    
    while True:
        try:
            measurements['PetalWidthCm'] = float(input("Enter Petal Width (cm): "))
            if measurements['PetalWidthCm'] <= 0:
                raise ValueError("Petal Width must be positive")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a positive number.")
    
    return measurements


def create_feature_dataframe(measurements):
    """
    Create a properly formatted DataFrame from measurements.
    
    Args:
        measurements (dict): Dictionary with feature measurements
    
    Returns:
        pandas.DataFrame: DataFrame with correct column order
    """
    # Ensure correct column order
    feature_order = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    df = pd.DataFrame([measurements], columns=feature_order)
    
    return df


def make_prediction(model_package, measurements):
    """
    Make a prediction for given measurements.
    
    Args:
        model_package (dict): Loaded model package
        measurements (dict): Flower measurements
    
    Returns:
        tuple: (predicted_species, prediction_probabilities)
    """
    # Create DataFrame
    X = create_feature_dataframe(measurements)
    
    print("\n" + "="*70)
    print("PROCESSING PREDICTION")
    print("="*70)
    
    print("\n📊 Input Features:")
    for feature, value in measurements.items():
        print(f"   {feature:20s}: {value:.2f} cm")
    
    # Scale features
    scaler = model_package['scaler']
    X_scaled = scaler.transform(X)
    
    print("\n🔄 Scaling features...")
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("   Scaled values:")
    for feature, value in zip(X_scaled_df.columns, X_scaled_df.iloc[0]):
        print(f"   {feature:20s}: {value:7.4f}")
    
    # Make prediction
    model = model_package['model']
    
    print(f"\n🤖 Making prediction using {model_package['model_name']}...")
    predicted_species = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    return predicted_species, prediction_proba


def display_prediction(predicted_species, prediction_proba, model_name):
    """
    Display prediction results in a user-friendly format.
    
    Args:
        predicted_species (str): Predicted species
        prediction_proba (array): Prediction probabilities
        model_name (str): Name of the model used
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    # Get class names (assuming order: setosa, versicolor, virginica)
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    # Find the predicted class
    max_prob_idx = prediction_proba.argmax()
    confidence = prediction_proba[max_prob_idx] * 100
    
    print(f"\n🌸 Predicted Species: {predicted_species}")
    print(f"✨ Confidence: {confidence:.2f}%")
    
    print(f"\n📊 Prediction Probabilities:")
    for species, prob in zip(class_names, prediction_proba):
        prob_pct = prob * 100
        bar_length = int(prob_pct / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"   {species:20s}: {prob_pct:5.2f}% {bar}")
    
    print(f"\n🎯 Model Used: {model_name}")
    
    # Interpretation
    print("\n💡 Interpretation:")
    if confidence >= 90:
        print("   Very high confidence prediction - the model is very certain.")
    elif confidence >= 75:
        print("   High confidence prediction - the model is fairly certain.")
    elif confidence >= 60:
        print("   Moderate confidence - there may be some similarity to other species.")
    else:
        print("   Low confidence - measurements may be ambiguous or unusual.")
    
    print("\n" + "="*70)


def predict_from_dict(measurements_dict, model_path='models/best_model.pkl'):
    """
    Make prediction from a dictionary of measurements (for programmatic use).
    
    Args:
        measurements_dict (dict): Dictionary with measurements
        model_path (str): Path to model file
    
    Returns:
        dict: Prediction results
    """
    model_package = load_best_model(model_path)
    predicted_species, prediction_proba = make_prediction(model_package, measurements_dict)
    
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    return {
        'predicted_species': predicted_species,
        'confidence': float(prediction_proba.max() * 100),
        'probabilities': {
            species: float(prob * 100)
            for species, prob in zip(class_names, prediction_proba)
        }
    }


def interactive_prediction():
    """
    Interactive prediction mode - get user input and make predictions.
    """
    try:
        # Load model
        model_package = load_best_model()
        
        while True:
            # Get user input
            measurements = get_user_input()
            
            # Make prediction
            predicted_species, prediction_proba = make_prediction(model_package, measurements)
            
            # Display results
            display_prediction(predicted_species, prediction_proba, model_package['model_name'])
            
            # Ask if user wants to predict another
            print("\n" + "="*70)
            another = input("\nWould you like to predict another flower? (yes/no): ").strip().lower()
            if another not in ['yes', 'y']:
                print("\n✅ Thank you for using the Iris Flower Prediction System!")
                print("="*70)
                break
            print()  # Add spacing
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the model has been trained and saved.")
    except KeyboardInterrupt:
        print("\n\n✅ Prediction cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


def predict_examples():
    """
    Test predictions with example measurements.
    """
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    examples = [
        {
            'name': 'Example 1 (Typical Setosa)',
            'measurements': {
                'SepalLengthCm': 5.1,
                'SepalWidthCm': 3.5,
                'PetalLengthCm': 1.4,
                'PetalWidthCm': 0.2
            }
        },
        {
            'name': 'Example 2 (Typical Versicolor)',
            'measurements': {
                'SepalLengthCm': 6.0,
                'SepalWidthCm': 2.7,
                'PetalLengthCm': 4.5,
                'PetalWidthCm': 1.3
            }
        },
        {
            'name': 'Example 3 (Typical Virginica)',
            'measurements': {
                'SepalLengthCm': 6.5,
                'SepalWidthCm': 3.0,
                'PetalLengthCm': 5.5,
                'PetalWidthCm': 2.0
            }
        }
    ]
    
    try:
        model_package = load_best_model()
        
        for example in examples:
            print(f"\n{'='*70}")
            print(f"{example['name']}")
            print('='*70)
            
            predicted_species, prediction_proba = make_prediction(
                model_package, 
                example['measurements']
            )
            
            display_prediction(
                predicted_species, 
                prediction_proba, 
                model_package['model_name']
            )
        
        print("\n✅ Example predictions completed!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        # Run example predictions
        predict_examples()
    else:
        # Run interactive mode
        interactive_prediction()
