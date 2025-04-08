# House Prices EDA Project

## Overview
This project demonstrates Exploratory Data Analysis (EDA) on the Kaggle House Prices dataset.  
The goal is to showcase proficiency in data cleaning, visualization, and feature engineering, producing a reproducible analysis that can be featured in my ML resume portfolio.

## Project Structure
```
EDA_House_Prices/
├── data/
│   ├── raw/                        # Raw dataset as downloaded from Kaggle
│   └── processed/                  # Cleaned and preprocessed data files
├── notebooks/
│   └── EDA_Project.ipynb           # Jupyter Notebook containing the complete analysis
├── src/
│   ├── data_loading.py             # Script for loading and initial exploration
│   ├── data_cleaning.py            # Data cleaning and preprocessing functions
│   ├── visualization.py            # Visualization functions for EDA
│   └── feature_engineering.py      # Feature engineering functions and analysis
├── environment.yml                 # Conda environment configuration file
├── requirements.txt                # Pip dependencies
├── README.md                       # Project overview, setup instructions, and documentation
└── .gitignore                      # Git ignore file
```

## Dataset
- **Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Description:** The dataset contains numerical and categorical features, missing values, and outliers, making it ideal for EDA and feature engineering.

## Tools and Libraries
- **Python:** 3.11
- **Pandas:** 1.5.3
- **NumPy:** 1.23.5
- **Matplotlib:** 3.6.3
- **Seaborn:** 0.12.2
- **Scikit-Learn**
- **(Optional) Plotly:** For interactive visualizations

## Environment Setup
- **Using Conda:**  
  Create the environment with:
  ```bash
  conda env create -f environment.yml
  conda activate eda-env
  ```
- **Using Pip:**  
  Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

## Analysis Summary
- **Data Loading and Exploration:**  
  Initial analysis included data structure, missing value checks, and duplicate detection.
- **Data Cleaning:**  
  Missing values were imputed, outliers were handled, and data types were correctly converted.
- **EDA:**  
  Various visualizations (histograms, scatter plots, heatmaps) were used to gain insights on data distribution and relationships between features.
- **Feature Engineering:**  
  New features were created, categorical variables encoded, and feature importance evaluated using linear regression.

## How to Run
- Open the Jupyter Notebook in the `notebooks/` directory for an interactive exploration.
- Run the scripts available in the `src/` directory for step-by-step tasks.

## Future Work
- Extend feature engineering and modeling.
- Implement predictive models and hyperparameter tuning.
- Deploy interactive dashboards for real-time insights.

