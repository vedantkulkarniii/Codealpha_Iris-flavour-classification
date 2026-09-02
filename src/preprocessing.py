"""
Data Preprocessing Module
Handles cleaning and preprocessing of the Iris dataset.
"""

import pandas as pd
import numpy as np


def remove_id_column(df):
    """
    Remove the Id column as it's not needed for modeling.
    
    Args:
        df (pandas.DataFrame): Dataset with Id column
    
    Returns:
        pandas.DataFrame: Dataset without Id column
    """
    if 'Id' in df.columns:
        df_clean = df.drop('Id', axis=1)
        print(f"✓ Removed 'Id' column")
        print(f"  Columns before: {len(df.columns)}")
        print(f"  Columns after: {len(df_clean.columns)}")
        return df_clean
    else:
        print("⚠️  'Id' column not found, skipping removal")
        return df.copy()


def handle_duplicates(df, keep='first'):
    """
    Handle duplicate records in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to clean
        keep (str): Which duplicates to keep ('first', 'last', or False)
    
    Returns:
        pandas.DataFrame: Dataset without duplicates
    """
    initial_count = len(df)
    duplicates_count = df.duplicated().sum()
    
    if duplicates_count > 0:
        print(f"\n⚠️  Found {duplicates_count} duplicate record(s)")
        print(f"  Removing duplicates (keeping {keep} occurrence)...")
        df_clean = df.drop_duplicates(keep=keep).reset_index(drop=True)
        removed = initial_count - len(df_clean)
        print(f"✓ Removed {removed} duplicate record(s)")
        print(f"  Records before: {initial_count}")
        print(f"  Records after: {len(df_clean)}")
        return df_clean
    else:
        print("✓ No duplicate records found")
        return df.copy()


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to clean
        strategy (str): Strategy to handle missing values ('drop', 'mean', 'median', 'mode')
    
    Returns:
        pandas.DataFrame: Dataset with missing values handled
    """
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        print("✓ No missing values found")
        return df.copy()
    
    print(f"\n⚠️  Found {missing_count} missing value(s)")
    print(f"  Strategy: {strategy}")
    
    df_clean = df.copy()
    
    if strategy == 'drop':
        initial_count = len(df_clean)
        df_clean = df_clean.dropna().reset_index(drop=True)
        removed = initial_count - len(df_clean)
        print(f"✓ Dropped {removed} row(s) with missing values")
    
    elif strategy in ['mean', 'median', 'mode']:
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_clean[col].mean()
                elif strategy == 'median':
                    fill_value = df_clean[col].median()
                else:  # mode
                    fill_value = df_clean[col].mode()[0]
                
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"✓ Filled missing values in '{col}' with {strategy}")
    
    return df_clean


def standardize_column_names(df):
    """
    Standardize column names (already in good format, but ensuring consistency).
    
    Args:
        df (pandas.DataFrame): Dataset with columns to standardize
    
    Returns:
        pandas.DataFrame: Dataset with standardized column names
    """
    df_clean = df.copy()
    original_columns = df_clean.columns.tolist()
    
    # Column names are already good, but we can strip whitespace
    df_clean.columns = df_clean.columns.str.strip()
    
    print("✓ Column names validated and standardized")
    print(f"  Columns: {', '.join(df_clean.columns)}")
    
    return df_clean


def verify_data_types(df):
    """
    Verify and ensure correct data types for all columns.
    
    Args:
        df (pandas.DataFrame): Dataset to verify
    
    Returns:
        pandas.DataFrame: Dataset with verified data types
    """
    df_clean = df.copy()
    
    print("\n🔍 Verifying data types...")
    
    # Ensure numerical columns are float
    numerical_expected = ['SepalLengthCm', 'SepalWidthCm', 
                         'PetalLengthCm', 'PetalWidthCm']
    
    for col in numerical_expected:
        if col in df_clean.columns:
            if df_clean[col].dtype != 'float64':
                df_clean[col] = df_clean[col].astype('float64')
                print(f"  Converted '{col}' to float64")
            else:
                print(f"  '{col}': float64 ✓")
    
    # Ensure Species is object/string
    if 'Species' in df_clean.columns:
        if df_clean['Species'].dtype == 'object':
            print(f"  'Species': object ✓")
        else:
            df_clean['Species'] = df_clean['Species'].astype('object')
            print(f"  Converted 'Species' to object")
    
    return df_clean


def get_feature_names():
    """
    Get the list of feature column names.
    
    Returns:
        list: Feature column names
    """
    return ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def get_target_name():
    """
    Get the target column name.
    
    Returns:
        str: Target column name
    """
    return 'Species'


def separate_features_target(df):
    """
    Separate features and target variable.
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    feature_columns = get_feature_names()
    target_column = get_target_name()
    
    # Verify columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print("\n✓ Separated features and target")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Feature columns: {', '.join(feature_columns)}")
    print(f"  Target column: {target_column}")
    
    return X, y


def preprocess_dataset(df, remove_duplicates=True):
    """
    Complete preprocessing pipeline for the Iris dataset.
    
    Args:
        df (pandas.DataFrame): Raw dataset
        remove_duplicates (bool): Whether to remove duplicate records
    
    Returns:
        tuple: (X, y, df_clean) - Features, target, and cleaned dataframe
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    print(f"\nInitial dataset shape: {df.shape}")
    
    # Step 1: Remove Id column
    print("\n[Step 1/5] Removing unnecessary columns...")
    df_clean = remove_id_column(df)
    
    # Step 2: Handle duplicates
    print("\n[Step 2/5] Handling duplicate records...")
    if remove_duplicates:
        df_clean = handle_duplicates(df_clean, keep='first')
    else:
        print("  Skipping duplicate removal (as requested)")
    
    # Step 3: Handle missing values
    print("\n[Step 3/5] Handling missing values...")
    df_clean = handle_missing_values(df_clean, strategy='drop')
    
    # Step 4: Standardize column names
    print("\n[Step 4/5] Standardizing column names...")
    df_clean = standardize_column_names(df_clean)
    
    # Step 5: Verify data types
    print("\n[Step 5/5] Verifying data types...")
    df_clean = verify_data_types(df_clean)
    
    # Separate features and target
    print("\n" + "-"*70)
    X, y = separate_features_target(df_clean)
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Records removed: {len(df) - len(df_clean)}")
    print(f"Final features (X): {X.shape}")
    print(f"Final target (y): {y.shape}")
    print(f"\nTarget distribution:")
    for species, count in y.value_counts().sort_index().items():
        print(f"  {species}: {count} samples")
    print("="*70)
    
    return X, y, df_clean


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from setup_dataset import setup_iris_dataset
    
    print("Loading dataset...")
    df, _ = setup_iris_dataset()
    
    print("\n\nStarting preprocessing...")
    X, y, df_clean = preprocess_dataset(df)
    
    print("\n✅ Preprocessing complete!")
    print(f"\nCleaned dataset ready for modeling:")
    print(f"  Features: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Target: {y.shape[0]} samples")
