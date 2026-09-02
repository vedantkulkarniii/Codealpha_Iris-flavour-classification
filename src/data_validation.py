"""
Data Validation Module
Comprehensive analysis of the Iris dataset including missing values, duplicates, 
data types, and class distribution.
"""

import pandas as pd
import numpy as np


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to check
    
    Returns:
        pandas.Series: Missing value counts per column
    """
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Percentage': missing_pct.values
    })
    
    print(missing_df.to_string(index=False))
    
    total_missing = missing.sum()
    if total_missing == 0:
        print("\n✅ No missing values found in the dataset!")
    else:
        print(f"\n⚠️  Total missing values: {total_missing}")
    
    return missing


def check_duplicates(df):
    """
    Check for duplicate records in the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to check
    
    Returns:
        int: Number of duplicate rows
    """
    print("\n" + "="*60)
    print("DUPLICATE RECORDS ANALYSIS")
    print("="*60)
    
    # Check duplicates excluding Id column
    cols_to_check = [col for col in df.columns if col != 'Id']
    duplicates = df[cols_to_check].duplicated()
    num_duplicates = duplicates.sum()
    
    print(f"Total records: {len(df)}")
    print(f"Duplicate records (excluding Id): {num_duplicates}")
    
    if num_duplicates > 0:
        print(f"\n⚠️  Found {num_duplicates} duplicate record(s)")
        print("\nDuplicate rows:")
        print(df[duplicates].to_string())
    else:
        print("\n✅ No duplicate records found!")
    
    return num_duplicates


def analyze_data_types(df):
    """
    Analyze and display data types of all columns.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
    """
    print("\n" + "="*60)
    print("DATA TYPES ANALYSIS")
    print("="*60)
    
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    print(dtype_df.to_string(index=False))
    
    # Identify feature types
    print("\n📊 Column Categories:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n  Numerical columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"    - {col}")
    
    print(f"\n  Categorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"    - {col}")


def analyze_target_distribution(df, target_column='Species'):
    """
    Analyze the distribution of the target variable (Species).
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        target_column (str): Name of the target column
    """
    print("\n" + "="*60)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*60)
    
    if target_column not in df.columns:
        print(f"❌ Column '{target_column}' not found in dataset")
        return
    
    # Count by species
    species_counts = df[target_column].value_counts().sort_index()
    species_pct = (df[target_column].value_counts(normalize=True) * 100).sort_index()
    
    print(f"\nTarget Column: {target_column}")
    print(f"Number of Classes: {df[target_column].nunique()}")
    print(f"\nClass Distribution:")
    
    for species in species_counts.index:
        count = species_counts[species]
        pct = species_pct[species]
        print(f"  {species:20s}: {count:3d} samples ({pct:5.2f}%)")
    
    # Check for balance
    max_count = species_counts.max()
    min_count = species_counts.min()
    
    if max_count == min_count:
        print("\n✅ Dataset is perfectly balanced across all classes!")
    else:
        imbalance_ratio = max_count / min_count
        print(f"\n⚠️  Class imbalance ratio: {imbalance_ratio:.2f}:1")


def analyze_numerical_features(df):
    """
    Analyze numerical features with detailed statistics.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
    """
    print("\n" + "="*60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*60)
    
    # Get numerical columns (exclude Id)
    numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if col != 'Id']
    
    if not numerical_cols:
        print("No numerical features found")
        return
    
    print(f"\nAnalyzing {len(numerical_cols)} numerical features:")
    print(", ".join(numerical_cols))
    
    # Detailed statistics
    stats_df = df[numerical_cols].describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    
    print("\n📊 Statistical Summary:")
    print(stats_df.to_string())
    
    # Check for outliers using IQR method
    print("\n📉 Outlier Detection (IQR Method):")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"  {col:20s}: {len(outliers)} outliers detected")
        else:
            print(f"  {col:20s}: No outliers detected ✓")


def identify_features_and_target(df):
    """
    Identify and display feature columns and target column.
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
    
    Returns:
        tuple: (feature_columns, target_column)
    """
    print("\n" + "="*60)
    print("FEATURE AND TARGET IDENTIFICATION")
    print("="*60)
    
    # Define features (exclude Id and Species)
    feature_columns = ['SepalLengthCm', 'SepalWidthCm', 
                      'PetalLengthCm', 'PetalWidthCm']
    target_column = 'Species'
    
    print(f"\n📊 Input Features ({len(feature_columns)}):")
    for i, feature in enumerate(feature_columns, 1):
        if feature in df.columns:
            print(f"  {i}. {feature} ✓")
        else:
            print(f"  {i}. {feature} ❌ (not found)")
    
    print(f"\n🎯 Target Variable:")
    if target_column in df.columns:
        print(f"  - {target_column} ✓")
        print(f"  - Classes: {df[target_column].nunique()}")
    else:
        print(f"  - {target_column} ❌ (not found)")
    
    print(f"\n🔍 Other Columns:")
    other_cols = [col for col in df.columns 
                 if col not in feature_columns and col != target_column]
    if other_cols:
        for col in other_cols:
            print(f"  - {col} (to be removed in preprocessing)")
    else:
        print("  None")
    
    return feature_columns, target_column


def validate_dataset(df):
    """
    Comprehensive dataset validation combining all checks.
    
    Args:
        df (pandas.DataFrame): Dataset to validate
    
    Returns:
        dict: Validation results
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("="*70)
    
    results = {}
    
    # Run all validation checks
    results['missing_values'] = check_missing_values(df)
    results['num_duplicates'] = check_duplicates(df)
    analyze_data_types(df)
    analyze_target_distribution(df)
    analyze_numerical_features(df)
    results['features'], results['target'] = identify_features_and_target(df)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"✅ Total Records: {len(df)}")
    print(f"✅ Total Columns: {len(df.columns)}")
    print(f"✅ Missing Values: {results['missing_values'].sum()}")
    print(f"✅ Duplicate Records: {results['num_duplicates']}")
    print(f"✅ Feature Columns: {len(results['features'])}")
    print(f"✅ Target Column: {results['target']}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from setup_dataset import setup_iris_dataset
    
    print("Loading dataset for validation...")
    df, _ = setup_iris_dataset()
    
    print("\n\nStarting validation...")
    validate_dataset(df)
