#!/usr/bin/env python3
"""
File: data_loading.py
Description: Load the dataset and perform initial exploration including checking the data structure,
missing values, and duplicates.
"""

import pandas as pd


def main():
    # Define the path to the training data file
    train_data_path = 'data/raw/house_prices_data/train.csv'

    # Load the dataset
    df_train = pd.read_csv(train_data_path)

    # Display the first few rows of the dataset
    print("Head of the dataset:")
    print(df_train.head())

    # Get information about the dataset (data types, non-null counts)
    print("\nData Info:")
    print(df_train.info())

    # Display summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(df_train.describe())

    # Identify missing values in each column
    print("\nMissing Values per Column:")
    print(df_train.isnull().sum())

    # Check for duplicate rows in the dataset
    print("\nNumber of Duplicate Rows:")
    print(df_train.duplicated().sum())


if __name__ == '__main__':
    main()
