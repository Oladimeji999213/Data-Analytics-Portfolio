============================================================
# Cryptocurrency Price Prediction Using Machine Learning
# Course: Predictive Analytics and Machine Learning Using Python
# Author: Oyewole Oladimeji Abdulqudus | Q1045935
# Berlin School of Business and Innovation | 2025
# ============================================================

# ---- PHASE 1: DATA LOADING AND EXPLORATION ----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
bitcoin_data = pd.read_csv('/content/Bitcoin.csv')
ethereum_data = pd.read_csv('/content/ethereum.csv')

# Preview datasets
print(bitcoin_data)
print(ethereum_data)

# Generate summary statistics
print("Bitcoin statistics:\n", bitcoin_data.describe())
print("Ethereum statistics:\n", ethereum_data.describe())

# Check for missing values
print("Missing values in Bitcoin data:\n", bitcoin_data.isnull().sum())
print("Missing values in Ethereum data:\n", ethereum_data.isnull().sum())

# Plot Bitcoin and Ethereum closing prices over time
plt.figure(figsize=(10, 4))
plt.plot(bitcoin_data['Date'], bitcoin_data['Close'], label='Bitcoin Closing Price', color='blue')
plt.plot(ethereum_data['date'], ethereum_data['Close'], label='Ethereum Closing Price', color='orange')
plt.title('Bitcoin and Ethereum Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Boxplot to detect trends and outliers
plt.figure(figsize=(8, 4))
sns.boxplot(data=[bitcoin_data['Close'], ethereum_data['Close']], notch=True)
plt.xticks([0, 1], ['Bitcoin', 'Ethereum'])
plt.title("Boxplot of Closing Prices")
plt.show()

# ---- PHASE 2: DATA PREPROCESSING ----

# Add new features: Price Change and Price Spread
bitcoin_data['Price Change'] = bitcoin_data['Close'] - bitcoin_data['Open']
bitcoin_data['Price Spread'] = bitcoin_data['High'] - bitcoin_data['Low']

ethereum_data['Price Change'] = ethereum_data['Close'] - ethereum_data['Open']
ethereum_data['Price Spread'] = ethereum_data['High'] - ethereum_data['Low']

# Define target variable: Future closing price
bitcoin_data['Future Close'] = bitcoin_data['Close'].shift(-1)
ethereum_data['Future Close'] = ethereum_data['Close'].shift(-1)

# Drop rows with NaN values after the shift
bitcoin_data = bitcoin_data.dropna()
ethereum_data = ethereum_data.dropna()

# Scale features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

bitcoin_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price Change', 'Price Spread']
ethereum_features = ['Open', 'High', 'Low', 'Close', 'price', 'Price Change', 'Price Spread']

scaler = MinMaxScaler()

# Scale Bitcoin features
bitcoin_scaled_data = scaler.fit_transform(bitcoin_data[bitcoin_features])

# Scale Ethereum features
ethereum_scaled_data = scaler.fit_transform(ethereum_data[ethereum_features])

# Re-add target variable
bitcoin_data['Future Close'] = bitcoin_data['Future Close'].values
ethereum_data['Future Close'] = ethereum_data['Future Close'].values

print(ethereum_data)
print(bitcoin_data)

# ---- PHASE 3: FEATURE SELECTION ----

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Handle missing values and infinities
for data in [bitcoin_data, ethereum_data]:
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

# Re-engineer target variable
bitcoin_data['Future_Close'] = bitcoin_data['Close'].shift(-1)
ethereum_data['Future_Close'] = ethereum_data['Close'].shift(-1)

# Drop rows with missing Future_Close values
bitcoin_data.dropna(subset=['Future_Close'], inplace=True)
ethereum_data.dropna(subset=['Future_Close'], inplace=True)

# Define features and target for both datasets
bitcoin_features = ['Close', 'Volume']
ethereum_features = ['Close']

X_bitcoin = bitcoin_data[bitcoin_features]
y_bitcoin = bitcoin_data['Future_Close']

X_ethereum = ethereum_data[ethereum_features]
y_ethereum = ethereum_data['Future_Close']

# Train-test split
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_bitcoin, y_bitcoin, test_size=0.2, random_state=42)
X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(
    X_ethereum, y_ethereum, test_size=0.2, random_state=42)

# 1. Filter Method (SelectKBest)
print("\nFilter Method: SelectKBest (Bitcoin)")
kbest_bitcoin = SelectKBest(score_func=f_regression, k='all')
kbest_bitcoin.fit(X_b_train, y_b_train)
for i, col in enumerate(bitcoin_features):
    print(f"Feature: {col}, Score: {kbest_bitcoin.scores_[i]}")

print("\nFilter Method: SelectKBest (Ethereum)")
kbest_ethereum = SelectKBest(score_func=f_regression, k='all')
kbest_ethereum.fit(X_e_train, y_e_train)
for i, col in enumerate(ethereum_features):
    print(f"Feature: {col}, Score: {kbest_ethereum.scores_[i]}")

# 2. Wrapper Method (RFE)
print("\nWrapper Method: RFE (Bitcoin)")
rfe_bitcoin = RFE(estimator=Lasso(alpha=0.01), n_features_to_select=1)
rfe_bitcoin.fit(X_b_train, y_b_train)
for i, col in enumerate(bitcoin_features):
    print(f"Feature: {col}, Selected: {rfe_bitcoin.support_[i]}")

if len(ethereum_features) > 1:
    print("\nWrapper Method: RFE (Ethereum)")
    rfe_ethereum = RFE(estimator=Lasso(alpha=0.01), n_features_to_select=1)
    rfe_ethereum.fit(X_e_train, y_e_train)
    for i, col in enumerate(ethereum_features):
        print(f"Feature: {col}, Selected: {rfe_ethereum.support_[i]}")
else:
    print("\nWrapper Method: RFE skipped for Ethereum (only one feature available).")

# 3. Embedded Method (LASSO)
print("\nEmbedded Method: LASSO (Bitcoin)")
lasso_bitcoin = Lasso(alpha=0.01)
lasso_bitcoin.fit(X_b_train, y_b_train)
for i, col in enumerate(bitcoin_features):
    print(f"Feature: {col}, Coefficient: {lasso_bitcoin.coef_[i]}")

print("\nEmbedded Method: LASSO (Ethereum)")
lasso_ethereum = Lasso(alpha=0.01)
lasso_ethereum.fit(X_e_train, y_e_train)
for i, col in enumerate(ethereum_features):
    print(f"Feature: {col}, Coefficient: {lasso_ethereum.coef_[i]}")

# ---- PHASE 4: MODEL TRAINING AND EVALUATION ----

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Use Close as best feature based on Phase 3 results
X_b_train = X_b_train[['Close']]
X_b_test = X_b_test[['Close']]

X_e_train = X_e_train[['Close']]
X_e_test = X_e_test[['Close']]

# Initialise models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate models for Bitcoin
print("\nBitcoin Model Evaluation:")
bitcoin_results = {}
for model_name, model in models.items():
    model.fit(X_b_train, y_b_train)
    predictions = model.predict(X_b_test)
    rmse = np.sqrt(mean_squared_error(y_b_test, predictions))
    bitcoin_results[model_name] = rmse
    print(f"{model_name}: RMSE = {rmse}")

# Train and evaluate models for Ethereum
print("\nEthereum Model Evaluation:")
ethereum_results = {}
for model_name, model in models.items():
    model.fit(X_e_train, y_e_train)
    predictions = model.predict(X_e_test)
    rmse = np.sqrt(mean_squared_error(y_e_test, predictions))
    ethereum_results[model_name] = rmse
    print(f"{model_name}: RMSE = {rmse}")

# ---- PHASE 5: VISUALISATION OF RESULTS ----

# Model RMSE Comparison for Bitcoin
models_list = ["Linear Regression", "Random Forest", "KNN"]
rmse_values = [1269.50, 1583.34, 1491.50]

plt.figure(figsize=(6, 3))
plt.bar(models_list, rmse_values, color=['blue', 'green', 'orange'])
plt.title("Model RMSE Comparison (Bitcoin Dataset)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.ylim(1000, 1700)
plt.show()
