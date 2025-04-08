#!/usr/bin/env python3
"""
File: feature_engineering.py
Description: This script performs feature engineering on the House Prices dataset, including:
             - Creating new features based on domain insights.
             - Encoding categorical variables using one-hot encoding.
             - Scaling numerical features using standardization.
             - Evaluating feature importance using a simple linear regression model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def create_new_features(df):
    # Create new feature 'TotalSF' if the required columns exist
    required_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
    if set(required_cols).issubset(df.columns):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Create an interaction feature 'OverallGrade' if 'OverallQual' and 'OverallCond' exist
    if {'OverallQual', 'OverallCond'}.issubset(df.columns):
        df['OverallGrade'] = df['OverallQual'] * df['OverallCond']
    return df


def encode_categorical(df):
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Apply one-hot encoding; drop the first level to prevent multicollinearity
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def scale_features(df):
    # Identify numeric columns (for scaling, exclude the target 'SalePrice')
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
    return df_scaled


def evaluate_feature_importance(df):
    # Separate predictors (X) and target variable (y)
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    # Check for missing values in X and y, and fill them if necessary.
    print("Missing values in X before fill:", X.isnull().sum().sum())
    print("Missing values in y before fill:", y.isnull().sum().sum())

    # Fill missing values in predictors with 0 (alternative: use proper imputation techniques)
    X = X.fillna(0)

    # Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Create a DataFrame to display feature importance based on model coefficients
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    feature_importance['AbsoluteCoefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='AbsoluteCoefficient', ascending=False)

    print("Top 10 Feature Importances:")
    print(feature_importance.head(10))

    # Plot the top 10 features by absolute coefficient value
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='AbsoluteCoefficient', y='Feature')
    plt.title('Top 10 Feature Importances (Absolute Coefficient Values)')
    plt.xlabel('Absolute Coefficient')
    plt.ylabel('Feature')
    plt.show()


def main():
    sns.set(style="whitegrid")

    # Load the dataset
    df = pd.read_csv('data/raw/house_prices_data/train.csv')

    # Create new features based on domain insights
    df = create_new_features(df)

    # Encode categorical variables using one-hot encoding
    df_encoded = encode_categorical(df)

    # Scale numerical features using StandardScaler
    df_encoded_scaled = scale_features(df_encoded)

    # Evaluate feature importance using a simple linear regression model
    evaluate_feature_importance(df_encoded_scaled)


if __name__ == '__main__':
    main()
