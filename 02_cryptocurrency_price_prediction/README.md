# Cryptocurrency Price Prediction

**Course:** Predictive Analytics and Machine Learning Using Python
**Institution:** Berlin School of Business and Innovation
**Year:** 2025
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Google Colab

-----

## Project Overview

This project applies machine learning techniques to predict the future closing prices of Bitcoin and Ethereum using historical price data. Three models are compared and evaluated to determine the most accurate approach for cryptocurrency price forecasting.

-----

## Dataset

Two datasets were used:

**Bitcoin Dataset**

- 1,151 records covering June 2019 to August 2022
- Features: Date, Open, High, Low, Close, Volume, Currency

**Ethereum Dataset**

- 2,358 records covering March 2016 to August 2022
- Features: Date, Open, High, Low, Close, Price, Currency

-----

## Methodology

**Phase 1: Data Exploration**

- Descriptive statistics for both datasets
- Time-series visualisation of closing prices
- Boxplot analysis to identify outliers
- Confirmed no missing values in either dataset

**Phase 2: Feature Engineering and Preprocessing**

- Created Price Change (Close minus Open)
- Created Price Spread (High minus Low)
- Defined target variable Future Close using shift(-1)
- Applied MinMax Scaling to normalise features to range [0, 1]
- Removed rows with NaN values after shifting

**Phase 3: Feature Selection**
Three methods applied to both datasets:

|Method              |Bitcoin Result                 |Ethereum Result           |
|--------------------|-------------------------------|--------------------------|
|Filter (SelectKBest)|Close: 156907, Volume: 471     |Close: 511308             |
|Wrapper (RFE)       |Close selected, Volume excluded|Only one feature available|
|Embedded (LASSO)    |Close: 0.996, Volume: ~0       |Close: 0.997              |

All three methods confirmed Close as the dominant predictive feature.

**Phase 4: Model Training and Evaluation**
Models evaluated using Root Mean Squared Error (RMSE):

|Model              |Bitcoin RMSE  |Ethereum RMSE|
|-------------------|--------------|-------------|
|Linear Regression  |1269.50 (Best)|73.65 (Best) |
|Random Forest      |1583.34       |93.04        |
|K-Nearest Neighbors|1491.50       |85.87        |

-----

## Key Findings

- Linear Regression outperformed both Random Forest and KNN for both cryptocurrencies
- The Close price is the strongest predictor of future closing price across all feature selection methods
- Bitcoin shows higher volatility and larger RMSE values compared to Ethereum
- Volume had minimal predictive value and was excluded from the final model

-----

## Files

|File                          |Description                                                                            |
|------------------------------|---------------------------------------------------------------------------------------|
|`cryptocurrency_prediction.py`|Full Python script covering EDA, preprocessing, feature selection, and model evaluation|

-----

## Skills Demonstrated

Predictive Modelling, Feature Engineering, Feature Selection, Linear Regression, Random Forest, K-Nearest Neighbors, Model Evaluation, RMSE, MinMax Scaling, Time-Series Analysis, Python, Scikit-learn, Matplotlib
