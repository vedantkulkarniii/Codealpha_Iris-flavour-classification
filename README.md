# Iris Flower Species Classification

## Project Overview
A machine learning project to classify iris flower species based on sepal and petal measurements using the classic Iris dataset.

## Objective
Build a classification model that can accurately predict the species of an iris flower (Setosa, Versicolor, or Virginica) based on four physical measurements.

## Dataset Source
- **Dataset**: Iris CSV Dataset
- **Source**: [Kaggle - Iris CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- **Download Method**: KaggleHub Python API

## Technologies
- Python 3.x
- KaggleHub - Dataset downloading
- Pandas - Data manipulation
- NumPy - Numerical operations
- Matplotlib - Visualization
- Seaborn - Statistical visualization
- Scikit-learn - Machine learning

## Dataset Features
The dataset contains 150 samples with 6 columns:
- **Id**: Sample identifier (1-150)
- **SepalLengthCm**: Sepal length in centimeters
- **SepalWidthCm**: Sepal width in centimeters
- **PetalLengthCm**: Petal length in centimeters
- **PetalWidthCm**: Petal width in centimeters
- **Species**: Target variable (Iris-setosa, Iris-versicolor, Iris-virginica)

## Installation

### Prerequisites
- Python 3.x
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/vedantkulkarniii/Iris-flavour-classification.git
cd Iris-flavour-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download and setup the dataset:
```bash
cd src
python setup_dataset.py
```

The dataset will be automatically downloaded from Kaggle using KaggleHub and stored in the local cache.

## Project Structure
```
iris-classification/
│
├── data/                    # Data directory
├── src/                     # Source code
│   ├── dataset_downloader.py   # KaggleHub dataset downloader
│   ├── data_loader.py          # Dataset loading utilities
│   ├── data_validation.py      # Data validation and analysis
│   ├── preprocessing.py        # Data cleaning pipeline
│   ├── eda.py                  # Exploratory data analysis
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Model evaluation & comparison
│   ├── predict.py              # Prediction system
│   └── setup_dataset.py        # Integrated setup pipeline
│
├── models/                  # Trained models (to be added)
├── visualizations/          # Generated plots (to be added)
├── notebooks/              # Jupyter notebooks (to be added)
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Data Preprocessing
The preprocessing pipeline includes:
- **Missing Value Analysis**: Check for and handle missing data
- **Duplicate Detection**: Identify and remove duplicate records (3 found)
- **Data Type Validation**: Ensure correct data types for all columns
- **Column Cleaning**: Remove unnecessary columns (Id column)
- **Feature-Target Separation**: Split data into features (X) and target (y)
- **Class Distribution**: Analyze target variable balance

After preprocessing:
- Dataset reduced from 150 to 147 samples (3 duplicates removed)
- 4 numerical features retained
- Perfectly balanced target classes

## Exploratory Data Analysis (EDA)
Comprehensive visual analysis to understand feature relationships and patterns:

### Visualizations Generated
1. **Species Distribution** - Count plot showing balanced classes
2. **Sepal Measurements** - Scatter plot of sepal length vs width by species
3. **Petal Measurements** - Scatter plot of petal length vs width by species
4. **Feature Pairplot** - Complete pairwise feature relationships
5. **Correlation Heatmap** - Feature correlation matrix
6. **Feature Distributions** - Violin plots showing distributions by species

### Key Findings
- **Petal measurements** are excellent discriminators between species
- **Strong correlation** (0.962) between petal length and width
- **Setosa** is clearly separable from other species
- **Versicolor and Virginica** have some overlap in feature space
- All visualizations saved in `visualizations/` directory

## Machine Learning Pipeline

### Train/Test Split
- **Split ratio**: 80/20 (117 train / 30 test)
- **Stratification**: Enabled to maintain class balance
- **Random state**: 42 for reproducibility
- **No data leakage**: Strict separation between train and test

### Feature Scaling
- **Method**: StandardScaler (z-score normalization)
- **Formula**: z = (x - μ) / σ
- **Fit on training data only**, transform both train and test
- Prevents data leakage from test set

### Baseline Model: Logistic Regression
- **Algorithm**: Logistic Regression with One-vs-Rest strategy
- **Solver**: lbfgs (Limited-memory BFGS)
- **Features**: 4 scaled numerical features
- **Classes**: 3 iris species
- **Model saved**: `models/logistic_regression.pkl` (includes scaler)

## Model Evaluation & Comparison

### Models Trained
1. **Logistic Regression** (Baseline) - Linear classifier
2. **K-Nearest Neighbors (KNN)** - Distance-based classifier (k=5)
3. **Decision Tree** - Tree-based classifier
4. **Random Forest** - Ensemble of 100 trees

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **96.67%** | **96.97%** | **96.67%** | **96.66%** |
| Logistic Regression | 93.33% | 93.33% | 93.33% | 93.33% |
| Decision Tree | 93.33% | 93.33% | 93.33% | 93.33% |
| K-Nearest Neighbors | 93.33% | 94.44% | 93.33% | 93.27% |

### Best Model: Random Forest 🏆
- **Accuracy**: 96.67% (29/30 correct predictions)
- **Only 1 misclassification** on test set
- Confusion matrices saved for all models
- Best model saved to `models/best_model.pkl`

## Prediction System

### Interactive Prediction
Run predictions on new flower measurements:
```bash
python src/predict.py
```

### Main Application
Menu-driven interface with options:
```bash
python main.py
```

**Features:**
1. Run Complete ML Pipeline
2. Make Predictions (Interactive)
3. Test Example Predictions
4. View Model Performance
5. Exit

### Example Usage
```python
from src.predict import predict_from_dict

measurements = {
    'SepalLengthCm': 5.1,
    'SepalWidthCm': 3.5,
    'PetalLengthCm': 1.4,
    'PetalWidthCm': 0.2
}

result = predict_from_dict(measurements)
# Result: {'predicted_species': 'Iris-setosa', 'confidence': 100.0, ...}
```

## Project Status
✅ Day 1: Project Setup & Dataset - Completed
✅ Day 2: Data Understanding & Cleaning - Completed
✅ Day 3: Exploratory Data Analysis - Completed
✅ Day 4: Train/Test Split & Baseline Model - Completed
✅ Day 5: Multiple Models & Evaluation - Completed
✅ Day 6: Prediction System & Integration - Completed
🚧 Day 7-8: Final Testing & Documentation - In Progress
