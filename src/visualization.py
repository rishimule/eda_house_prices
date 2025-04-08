#!/usr/bin/env python3
"""
File: visualization.py
Description: This script performs Exploratory Data Analysis (EDA) on the House Prices dataset.
It includes univariate analysis, bivariate/multivariate analysis, and missing data visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data():
    data_path = 'data/raw/house_prices_data/train.csv'
    return pd.read_csv(data_path)


def univariate_analysis(df):
    # List of numerical features to analyze
    num_features = ['SalePrice', 'GrLivArea']

    for col in num_features:
        # Histogram with density plot (KDE)
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram and Density Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

        # Boxplot for outlier detection
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()


def bivariate_multivariate_analysis(df):
    # Filter the DataFrame to only include numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    # Correlation heatmap for numerical features
    plt.figure(figsize=(12, 10))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Pair plot for selected features
    sns.pairplot(df[['SalePrice', 'GrLivArea', 'OverallQual']])
    plt.suptitle('Pair Plot of Selected Features', y=1.02)
    plt.show()

    # Scatter plot: GrLivArea vs SalePrice
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
    plt.title('Scatter Plot: GrLivArea vs SalePrice')
    plt.xlabel('GrLivArea')
    plt.ylabel('SalePrice')
    plt.show()

    # Bar plot: Average SalePrice by OverallQual
    plt.figure(figsize=(8, 6))
    avg_saleprice = df.groupby('OverallQual')['SalePrice'].mean().reset_index()
    sns.barplot(x='OverallQual', y='SalePrice', data=avg_saleprice)
    plt.title('Average SalePrice by OverallQual')
    plt.xlabel('OverallQual')
    plt.ylabel('Average SalePrice')
    plt.show()


def missing_data_visualization(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Data Points')
    plt.show()


def main():
    sns.set(style="whitegrid")  # Set the visualization style
    df = load_data()

    # Univariate Analysis
    univariate_analysis(df)

    # Bivariate/Multivariate Analysis
    bivariate_multivariate_analysis(df)

    # Missing Data Visualization
    missing_data_visualization(df)


if __name__ == '__main__':
    main()
