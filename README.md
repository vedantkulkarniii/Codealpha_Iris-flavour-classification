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
│   └── setup_dataset.py        # Integrated setup pipeline
│
├── models/                  # Trained models (to be added)
├── visualizations/          # Generated plots (to be added)
├── notebooks/              # Jupyter notebooks (to be added)
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Project Status
✅ Day 1: Project Setup & Dataset - Completed
🚧 Day 2: Data Understanding & Cleaning - Not Started
