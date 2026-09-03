"""
Exploratory Data Analysis (EDA) Module
Comprehensive visualization and analysis of the Iris dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def create_output_directory(directory='visualizations'):
    """
    Create output directory for visualizations if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")
    else:
        print(f"✓ Directory exists: {directory}")


def plot_species_distribution(df, save_path='visualizations/species_distribution.png'):
    """
    Create a count plot showing the distribution of species.
    
    Args:
        df (pandas.DataFrame): Dataset with Species column
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create count plot
    ax = sns.countplot(data=df, x='Species', palette='Set2', hue='Species', legend=False)
    
    # Customize plot
    plt.title('Distribution of Iris Species', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Species', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    
    # Add count labels on bars
    for container in ax.containers:
        ax.bar_label(container, fontsize=11, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Print insights
    print("\n📊 Species Distribution Insights:")
    counts = df['Species'].value_counts()
    for species, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"   - {species}: {count} samples ({percentage:.2f}%)")


def plot_sepal_measurements(df, save_path='visualizations/sepal_measurements.png'):
    """
    Create a scatter plot of Sepal Length vs Sepal Width.
    
    Args:
        df (pandas.DataFrame): Dataset with sepal measurements
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with species differentiation
    species_list = df['Species'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, species in enumerate(sorted(species_list)):
        species_data = df[df['Species'] == species]
        plt.scatter(species_data['SepalLengthCm'], 
                   species_data['SepalWidthCm'],
                   label=species, 
                   alpha=0.7, 
                   s=80,
                   color=colors[i],
                   edgecolors='black',
                   linewidth=0.5)
    
    plt.title('Sepal Length vs Sepal Width', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
    plt.legend(title='Species', title_fontsize=11, fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Print insights
    print("\n📊 Sepal Measurements Insights:")
    print(f"   - Sepal Length range: [{df['SepalLengthCm'].min():.2f}, {df['SepalLengthCm'].max():.2f}] cm")
    print(f"   - Sepal Width range: [{df['SepalWidthCm'].min():.2f}, {df['SepalWidthCm'].max():.2f}] cm")
    print(f"   - Setosa tends to have shorter sepals but wider")
    print(f"   - Virginica typically has the longest sepals")


def plot_petal_measurements(df, save_path='visualizations/petal_measurements.png'):
    """
    Create a scatter plot of Petal Length vs Petal Width.
    
    Args:
        df (pandas.DataFrame): Dataset with petal measurements
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with species differentiation
    species_list = df['Species'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, species in enumerate(sorted(species_list)):
        species_data = df[df['Species'] == species]
        plt.scatter(species_data['PetalLengthCm'], 
                   species_data['PetalWidthCm'],
                   label=species, 
                   alpha=0.7, 
                   s=80,
                   color=colors[i],
                   edgecolors='black',
                   linewidth=0.5)
    
    plt.title('Petal Length vs Petal Width', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
    plt.ylabel('Petal Width (cm)', fontsize=12, fontweight='bold')
    plt.legend(title='Species', title_fontsize=11, fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Print insights
    print("\n📊 Petal Measurements Insights:")
    print(f"   - Petal Length range: [{df['PetalLengthCm'].min():.2f}, {df['PetalLengthCm'].max():.2f}] cm")
    print(f"   - Petal Width range: [{df['PetalWidthCm'].min():.2f}, {df['PetalWidthCm'].max():.2f}] cm")
    print(f"   - Clear separation between species based on petal measurements")
    print(f"   - Setosa has distinctly smaller petals")
    print(f"   - Strong positive correlation between petal length and width")


def plot_pairplot(df, save_path='visualizations/feature_pairplot.png'):
    """
    Create a comprehensive pairplot showing relationships between all features.
    
    Args:
        df (pandas.DataFrame): Dataset with all features
        save_path (str): Path to save the plot
    """
    # Select only feature columns and target
    plot_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    plot_df = df[plot_cols].copy()
    
    # Create pairplot
    pairplot = sns.pairplot(plot_df, 
                           hue='Species', 
                           palette='Set2',
                           diag_kind='hist',
                           plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'black', 'linewidth': 0.5},
                           height=2.5)
    
    pairplot.fig.suptitle('Feature Relationships Across Species', 
                         fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Print insights
    print("\n📊 Pairplot Insights:")
    print(f"   - Petal features show strong separation between species")
    print(f"   - Sepal features have more overlap between species")
    print(f"   - Strong correlation between petal length and width")
    print(f"   - Setosa is clearly distinguishable from other species")


def plot_correlation_heatmap(df, save_path='visualizations/correlation_heatmap.png'):
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        df (pandas.DataFrame): Dataset with numerical features
        save_path (str): Path to save the plot
    """
    # Select numerical features
    numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle (optional, for cleaner look)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1,
                mask=mask)
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    # Print insights
    print("\n📊 Correlation Insights:")
    print(f"   Strong positive correlations:")
    
    # Find strong correlations (> 0.8, excluding diagonal)
    strong_corrs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                feat1 = correlation_matrix.columns[i]
                feat2 = correlation_matrix.columns[j]
                strong_corrs.append((feat1, feat2, corr_value))
    
    for feat1, feat2, corr in strong_corrs:
        print(f"   - {feat1} ↔ {feat2}: {corr:.3f}")
    
    if not strong_corrs:
        print("   - No very strong correlations (>0.8) found")
    
    # Show all correlations
    print(f"\n   All pairwise correlations:")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            feat1 = correlation_matrix.columns[i]
            feat2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            print(f"   - {feat1} ↔ {feat2}: {corr_value:.3f}")


def plot_feature_distributions(df, save_path='visualizations/feature_distributions.png'):
    """
    Create distribution plots for all numerical features by species.
    
    Args:
        df (pandas.DataFrame): Dataset with features
        save_path (str): Path to save the plot
    """
    numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        # Create violin plot with box plot overlay
        sns.violinplot(data=df, x='Species', y=feature, hue='Species', ax=axes[idx], 
                      palette='Set2', alpha=0.6, legend=False)
        sns.boxplot(data=df, x='Species', y=feature, hue='Species', ax=axes[idx],
                   width=0.3, palette='Set2', showcaps=True, 
                   boxprops={'alpha': 0.8}, showfliers=False, legend=False)
        
        axes[idx].set_title(f'{feature} Distribution by Species', 
                          fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Species', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(f'{feature} (cm)', fontsize=10, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distributions Across Species', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    print("\n📊 Distribution Insights:")
    print(f"   - Each feature shows distinct patterns across species")
    print(f"   - Setosa shows clear separation in petal measurements")
    print(f"   - Some overlap exists in sepal measurements")


def perform_eda(df):
    """
    Perform complete exploratory data analysis with all visualizations.
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Create output directory
    create_output_directory('visualizations')
    
    print("\n[Visualization 1/6] Species Distribution...")
    plot_species_distribution(df)
    
    print("\n[Visualization 2/6] Sepal Measurements...")
    plot_sepal_measurements(df)
    
    print("\n[Visualization 3/6] Petal Measurements...")
    plot_petal_measurements(df)
    
    print("\n[Visualization 4/6] Feature Pairplot...")
    plot_pairplot(df)
    
    print("\n[Visualization 5/6] Correlation Heatmap...")
    plot_correlation_heatmap(df)
    
    print("\n[Visualization 6/6] Feature Distributions...")
    plot_feature_distributions(df)
    
    print("\n" + "="*70)
    print("EDA COMPLETE")
    print("="*70)
    print("✅ All visualizations saved to 'visualizations/' directory")
    print("✅ 6 comprehensive plots generated")
    print("\n💡 Key Findings:")
    print("   1. Dataset is well-balanced across three species")
    print("   2. Petal measurements are excellent discriminators")
    print("   3. Strong correlation between petal length and width")
    print("   4. Setosa is clearly separable from other species")
    print("   5. Versicolor and Virginica have some overlap in features")
    print("="*70)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from setup_dataset import setup_iris_dataset
    from preprocessing import preprocess_dataset
    
    print("Loading and preprocessing dataset...")
    df_raw, _ = setup_iris_dataset()
    X, y, df_clean = preprocess_dataset(df_raw)
    
    print("\n\nStarting EDA...")
    perform_eda(df_clean)
