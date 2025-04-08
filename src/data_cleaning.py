#!/usr/bin/env python3
"""
File: data_cleaning.py
Description: This script performs data cleaning and preprocessing on the House Prices dataset.
It handles missing values, detects and removes outliers, and converts data types as appropriate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    train_data_path = 'data/raw/house_prices_data/train.csv'
    return pd.read_csv(train_data_path)


def handle_missing_values(df):
    # Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Impute missing values for numerical columns with median
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Impute missing values for categorical columns with mode
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Drop columns with more than 50% missing values
    threshold = 0.5
    cols_to_drop = df.columns[df.isnull().mean() > threshold]
    if len(cols_to_drop) > 0:
        print("Dropping columns with more than 50% missing values:", list(cols_to_drop))
        df = df.drop(columns=cols_to_drop)
    return df


def detect_and_handle_outliers(df):
    # Visualize histograms for key numerical columns (e.g., 'SalePrice' and 'GrLivArea')
    for col in ['SalePrice', 'GrLivArea']:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    # Boxplot for 'SalePrice'
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df['SalePrice'])
    plt.title('Boxplot of SalePrice')
    plt.xlabel('SalePrice')
    plt.show()

    # Remove outliers in 'SalePrice' using the IQR method
    Q1 = df['SalePrice'].quantile(0.25)
    Q3 = df['SalePrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    initial_count = df.shape[0]
    df = df[(df['SalePrice'] >= lower_bound) & (df['SalePrice'] <= upper_bound)]
    final_count = df.shape[0]
    print(f"Removed {initial_count - final_count} outliers from SalePrice.")
    return df


def convert_data_types(df):
    # Example: Convert 'MSZoning' to categorical if it exists
    if 'MSZoning' in df.columns:
        df['MSZoning'] = df['MSZoning'].astype('category')
    return df


def main():
    # Load the dataset
    df = load_data()

    # Handle missing values
    df = handle_missing_values(df)

    # Detect and handle outliers
    df = detect_and_handle_outliers(df)

    # Convert data types
    df = convert_data_types(df)

    # Final Data Check: Display DataFrame information
    print("Final Data Information:")
    print(df.info())


if __name__ == '__main__':
    main()
