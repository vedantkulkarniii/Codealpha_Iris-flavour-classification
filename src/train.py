"""
Model Training Module
Handles train/test split, feature scaling, and model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    
    print(f"\nOriginal dataset size: {len(X)} samples")
    print(f"Test size: {test_size * 100:.0f}%")
    print(f"Random state: {random_state}")
    print(f"Stratification: Enabled (maintains class distribution)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n✓ Data split completed")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Show class distribution
    print(f"\n📊 Training set class distribution:")
    for species, count in y_train.value_counts().sort_index().items():
        percentage = (count / len(y_train)) * 100
        print(f"   {species:20s}: {count:3d} samples ({percentage:5.2f}%)")
    
    print(f"\n📊 Test set class distribution:")
    for species, count in y_test.value_counts().sort_index().items():
        percentage = (count / len(y_test)) * 100
        print(f"   {species:20s}: {count:3d} samples ({percentage:5.2f}%)")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler (z-score normalization).
    Fit on training data, transform both train and test.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Test features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)
    
    print("\nScaling method: StandardScaler (z-score normalization)")
    print("Formula: z = (x - mean) / std")
    
    scaler = StandardScaler()
    
    # Fit on training data only
    scaler.fit(X_train)
    
    print(f"\n✓ Scaler fitted on training data")
    print(f"\nFeature means (from training data):")
    for feature, mean in zip(X_train.columns, scaler.mean_):
        print(f"   {feature:20s}: {mean:7.4f}")
    
    print(f"\nFeature standard deviations (from training data):")
    for feature, std in zip(X_train.columns, scaler.scale_):
        print(f"   {feature:20s}: {std:7.4f}")
    
    # Transform both train and test
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"\n✓ Features scaled")
    print(f"  Training set scaled: {X_train_scaled.shape}")
    print(f"  Test set scaled: {X_test_scaled.shape}")
    
    # Show example of scaled values
    print(f"\n📊 Example: First training sample (before and after scaling)")
    print(f"   Original values:")
    for feature, value in X_train.iloc[0].items():
        print(f"      {feature:20s}: {value:7.4f}")
    
    print(f"   Scaled values:")
    for feature, value in X_train_scaled.iloc[0].items():
        print(f"      {feature:20s}: {value:7.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model (baseline).
    
    Args:
        X_train (pandas.DataFrame): Scaled training features
        y_train (pandas.Series): Training target
        random_state (int): Random seed
    
    Returns:
        LogisticRegression: Trained model
    """
    print("\n" + "="*70)
    print("BASELINE MODEL TRAINING: LOGISTIC REGRESSION")
    print("="*70)
    
    print("\nModel: Logistic Regression")
    print("Type: Linear classifier with probabilistic output")
    print("Solver: lbfgs (Limited-memory BFGS)")
    print("Multi-class: One-vs-Rest (OvR)")
    print(f"Max iterations: 200")
    print(f"Random state: {random_state}")
    
    # Initialize model
    model = LogisticRegression(
        random_state=random_state,
        max_iter=200,
        solver='lbfgs'
    )
    
    print(f"\n🔄 Training model...")
    
    # Train model
    model.fit(X_train, y_train)
    
    print(f"✓ Model training completed")
    
    # Show model details
    print(f"\n📊 Model Details:")
    print(f"   Classes: {list(model.classes_)}")
    print(f"   Number of features: {model.n_features_in_}")
    print(f"   Feature names: {list(X_train.columns)}")
    
    # Show coefficients
    print(f"\n📊 Model Coefficients (feature importance indicators):")
    for class_idx, class_name in enumerate(model.classes_):
        print(f"\n   {class_name}:")
        coefficients = model.coef_[class_idx]
        for feature, coef in zip(X_train.columns, coefficients):
            print(f"      {feature:20s}: {coef:7.4f}")
    
    print(f"\n📊 Model Intercepts:")
    for class_name, intercept in zip(model.classes_, model.intercept_):
        print(f"   {class_name:20s}: {intercept:7.4f}")
    
    return model


def make_predictions(model, X_test):
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained model
        X_test (pandas.DataFrame): Test features
    
    Returns:
        tuple: (predictions, prediction_probabilities)
    """
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    print(f"\nGenerating predictions for {len(X_test)} test samples...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"✓ Predictions generated")
    
    # Show prediction distribution
    print(f"\n📊 Prediction Distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for species, count in zip(unique, counts):
        percentage = (count / len(y_pred)) * 100
        print(f"   {species:20s}: {count:3d} samples ({percentage:5.2f}%)")
    
    # Show example predictions with probabilities
    print(f"\n📊 Example Predictions (first 5 test samples):")
    print(f"   {'Predicted':20s}  {'Confidence':>10s}  {'Probabilities'}")
    print(f"   {'-'*20}  {'-'*10}  {'-'*50}")
    
    for i in range(min(5, len(y_pred))):
        pred_class = y_pred[i]
        max_proba = y_pred_proba[i].max()
        proba_str = " / ".join([f"{p:.3f}" for p in y_pred_proba[i]])
        print(f"   {pred_class:20s}  {max_proba:9.3f}  [{proba_str}]")
    
    return y_pred, y_pred_proba


def save_model(model, scaler, filepath='models/logistic_regression.pkl'):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath (str): Path to save the model
    """
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save both model and scaler together
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None
    }
    
    joblib.dump(model_package, filepath)
    
    file_size = os.path.getsize(filepath)
    
    print(f"\n✓ Model and scaler saved")
    print(f"  Filepath: {filepath}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print(f"  Contents: Model + Scaler + Feature names")
    
    return filepath


def load_model(filepath='models/logistic_regression.pkl'):
    """
    Load saved model and scaler from disk.
    
    Args:
        filepath (str): Path to saved model
    
    Returns:
        dict: Dictionary with 'model', 'scaler', and 'feature_names'
    """
    print(f"\nLoading model from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_package = joblib.load(filepath)
    
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model_package['model']).__name__}")
    print(f"  Scaler type: {type(model_package['scaler']).__name__}")
    if model_package['feature_names']:
        print(f"  Features: {', '.join(model_package['feature_names'])}")
    
    return model_package


def train_baseline_model(X, y, test_size=0.2, random_state=42):
    """
    Complete pipeline to train baseline Logistic Regression model.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Test set proportion
        random_state (int): Random seed
    
    Returns:
        dict: Dictionary with all training artifacts
    """
    print("\n" + "="*70)
    print("BASELINE MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Step 2: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 3: Train model
    model = train_logistic_regression(X_train_scaled, y_train, random_state)
    
    # Step 4: Make predictions
    y_pred, y_pred_proba = make_predictions(model, X_test_scaled)
    
    # Step 5: Save model
    model_path = save_model(model, scaler)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"✅ Data split: {len(X_train)} train / {len(X_test)} test")
    print(f"✅ Features scaled using StandardScaler")
    print(f"✅ Model trained: Logistic Regression")
    print(f"✅ Predictions generated: {len(y_pred)} samples")
    print(f"✅ Model saved: {model_path}")
    print("="*70)
    
    # Return all artifacts
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model_path': model_path
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from setup_dataset import setup_iris_dataset
    from preprocessing import preprocess_dataset
    
    print("Loading and preprocessing dataset...")
    df_raw, _ = setup_iris_dataset()
    X, y, df_clean = preprocess_dataset(df_raw)
    
    print("\n\nStarting baseline model training...")
    results = train_baseline_model(X, y)
    
    print("\n✅ Baseline model training complete!")
    print(f"   Ready for evaluation on Day 5")
