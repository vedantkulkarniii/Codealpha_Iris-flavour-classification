"""
Model Evaluation Module
Train multiple classification models and compare their performance.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def train_knn(X_train, y_train, n_neighbors=5, random_state=42):
    """
    Train K-Nearest Neighbors classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_neighbors (int): Number of neighbors
        random_state (int): Random seed
    
    Returns:
        KNeighborsClassifier: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING: K-NEAREST NEIGHBORS (KNN)")
    print("="*70)
    
    print(f"\nModel: K-Nearest Neighbors")
    print(f"Algorithm: Distance-based classification")
    print(f"Number of neighbors (k): {n_neighbors}")
    print(f"Distance metric: Euclidean")
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    print(f"\n🔄 Training KNN model...")
    model.fit(X_train, y_train)
    
    print(f"✓ KNN training completed")
    print(f"  Classes: {list(model.classes_)}")
    print(f"  Training samples: {len(X_train)}")
    
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed
    
    Returns:
        DecisionTreeClassifier: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING: DECISION TREE")
    print("="*70)
    
    print(f"\nModel: Decision Tree")
    print(f"Algorithm: Tree-based classification")
    print(f"Criterion: Gini impurity")
    print(f"Random state: {random_state}")
    
    model = DecisionTreeClassifier(random_state=random_state)
    
    print(f"\n🔄 Training Decision Tree...")
    model.fit(X_train, y_train)
    
    print(f"✓ Decision Tree training completed")
    print(f"  Classes: {list(model.classes_)}")
    print(f"  Tree depth: {model.get_depth()}")
    print(f"  Number of leaves: {model.get_n_leaves()}")
    
    # Feature importance
    print(f"\n📊 Feature Importances:")
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        print(f"   {feature:20s}: {importance:.4f}")
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators (int): Number of trees
        random_state (int): Random seed
    
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING: RANDOM FOREST")
    print("="*70)
    
    print(f"\nModel: Random Forest")
    print(f"Algorithm: Ensemble of decision trees")
    print(f"Number of trees: {n_estimators}")
    print(f"Criterion: Gini impurity")
    print(f"Random state: {random_state}")
    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    print(f"\n🔄 Training Random Forest...")
    model.fit(X_train, y_train)
    
    print(f"✓ Random Forest training completed")
    print(f"  Classes: {list(model.classes_)}")
    print(f"  Number of trees: {model.n_estimators}")
    
    # Feature importance
    print(f"\n📊 Feature Importances:")
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        print(f"   {feature:20s}: {importance:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    return {
        'model_name': model_name,
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def generate_classification_report(y_test, y_pred, model_name, target_names=None):
    """
    Generate and display classification report.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        target_names (list): List of class names
    """
    print(f"\n" + "="*70)
    print(f"CLASSIFICATION REPORT: {model_name}")
    print("="*70)
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    return report


def plot_confusion_matrix(y_test, y_pred, model_name, class_names, save_path=None):
    """
    Plot confusion matrix for a model.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {save_path}")
    
    plt.close()


def create_comparison_table(results):
    """
    Create a comparison table for all models.
    
    Args:
        results (list): List of result dictionaries
    
    Returns:
        pandas.DataFrame: Comparison table
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1-Score': r['f1_score']
        }
        for r in results
    ])
    
    # Sort by F1-Score descending
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def identify_best_model(results):
    """
    Identify the best performing model based on F1-Score.
    
    Args:
        results (list): List of result dictionaries
    
    Returns:
        dict: Best model result dictionary
    """
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print("\n" + "="*70)
    print("BEST MODEL IDENTIFIED")
    print("="*70)
    
    print(f"\n🏆 Best Model: {best_result['model_name']}")
    print(f"\n   Performance Metrics:")
    print(f"   Accuracy:  {best_result['accuracy']:.4f}")
    print(f"   Precision: {best_result['precision']:.4f}")
    print(f"   Recall:    {best_result['recall']:.4f}")
    print(f"   F1-Score:  {best_result['f1_score']:.4f}")
    
    return best_result


def save_best_model(best_result, scaler, filepath='models/best_model.pkl'):
    """
    Save the best model with scaler.
    
    Args:
        best_result (dict): Best model result
        scaler: Fitted scaler
        filepath (str): Path to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_package = {
        'model': best_result['model'],
        'model_name': best_result['model_name'],
        'scaler': scaler,
        'metrics': {
            'accuracy': best_result['accuracy'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'f1_score': best_result['f1_score']
        }
    }
    
    joblib.dump(model_package, filepath)
    
    file_size = os.path.getsize(filepath)
    
    print(f"\n✓ Best model saved")
    print(f"  Filepath: {filepath}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    print(f"  Contents: Model + Scaler + Metrics")


def train_and_evaluate_all_models(X_train, X_test, y_train, y_test, scaler):
    """
    Train and evaluate all classification models.
    
    Args:
        X_train: Training features (scaled)
        X_test: Test features (scaled)
        y_train: Training target
        y_test: Test target
        scaler: Fitted scaler
    
    Returns:
        tuple: (results_list, comparison_df, best_result)
    """
    print("\n" + "="*70)
    print("MULTIPLE MODEL TRAINING & EVALUATION PIPELINE")
    print("="*70)
    
    results = []
    
    # Get class names
    class_names = sorted(y_train.unique())
    
    # Model 1: Logistic Regression (already trained, load it)
    print("\n[Model 1/4] Loading Logistic Regression (baseline)...")
    try:
        from train import load_model
        lr_package = load_model('models/logistic_regression.pkl')
        lr_model = lr_package['model']
        lr_result = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        results.append(lr_result)
        
        # Generate classification report
        generate_classification_report(y_test, lr_result['y_pred'], 
                                      "Logistic Regression", class_names)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, lr_result['y_pred'], 
                            "Logistic Regression", class_names,
                            'visualizations/confusion_matrix_lr.png')
    except Exception as e:
        print(f"Warning: Could not load Logistic Regression: {e}")
    
    # Model 2: K-Nearest Neighbors
    print("\n[Model 2/4] Training K-Nearest Neighbors...")
    knn_model = train_knn(X_train, y_train)
    knn_result = evaluate_model(knn_model, X_test, y_test, "K-Nearest Neighbors")
    results.append(knn_result)
    
    generate_classification_report(y_test, knn_result['y_pred'], 
                                  "K-Nearest Neighbors", class_names)
    
    plot_confusion_matrix(y_test, knn_result['y_pred'], 
                        "K-Nearest Neighbors", class_names,
                        'visualizations/confusion_matrix_knn.png')
    
    # Model 3: Decision Tree
    print("\n[Model 3/4] Training Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train)
    dt_result = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    results.append(dt_result)
    
    generate_classification_report(y_test, dt_result['y_pred'], 
                                  "Decision Tree", class_names)
    
    plot_confusion_matrix(y_test, dt_result['y_pred'], 
                        "Decision Tree", class_names,
                        'visualizations/confusion_matrix_dt.png')
    
    # Model 4: Random Forest
    print("\n[Model 4/4] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_result)
    
    generate_classification_report(y_test, rf_result['y_pred'], 
                                  "Random Forest", class_names)
    
    plot_confusion_matrix(y_test, rf_result['y_pred'], 
                        "Random Forest", class_names,
                        'visualizations/confusion_matrix_rf.png')
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Identify best model
    best_result = identify_best_model(results)
    
    # Save best model
    save_best_model(best_result, scaler)
    
    print("\n" + "="*70)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*70)
    print(f"✅ 4 models trained and evaluated")
    print(f"✅ Classification reports generated")
    print(f"✅ 4 confusion matrices saved")
    print(f"✅ Model comparison table created")
    print(f"✅ Best model identified and saved")
    print("="*70)
    
    return results, comparison_df, best_result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from setup_dataset import setup_iris_dataset
    from preprocessing import preprocess_dataset
    from train import train_baseline_model
    
    print("Loading and preprocessing dataset...")
    df_raw, _ = setup_iris_dataset()
    X, y, df_clean = preprocess_dataset(df_raw)
    
    print("\n\nTraining baseline model and splitting data...")
    training_results = train_baseline_model(X, y)
    
    print("\n\nTraining and evaluating all models...")
    results, comparison_df, best_result = train_and_evaluate_all_models(
        training_results['X_train_scaled'],
        training_results['X_test_scaled'],
        training_results['y_train'],
        training_results['y_test'],
        training_results['scaler']
    )
    
    print("\n✅ All models trained and evaluated!")
