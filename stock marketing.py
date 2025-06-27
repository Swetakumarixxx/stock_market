#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install yfinance')


# In[10]:


import yfinance as yf

data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
print(data.head())
data.to_csv('real_stock_data.csv')


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# # 1️⃣ Load data (Example: using a CSV file)
# data = pd.read_csv("C:/Users/SWETA KUMARI/Downloads/stock_data.csv")  # Replace with your CSV file
# print(data.head())
# Install yfinance if not installed
# pip install yfinance

import yfinance as yf

# Example: Get Apple (AAPL) data from Jan 2020 to today
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-12-31')

print(data.head())

# Save to CSV
data.to_csv('real_stock_data.csv')


# 2️⃣ Create lag feature
data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# 3️⃣ Split data
X = data[['Prev_Close']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4️⃣ Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Closing Price')
plt.plot(y_pred, label='Predicted Closing Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression: Actual vs Predicted Closing Prices')
plt.show()

# 6️⃣ Simple Neural Network
from keras.models import Sequential
from keras.layers import Dense

nn_model = Sequential()
nn_model.add(Dense(32, input_dim=1, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(optimizer='adam', loss='mse')

# Train neural network
nn_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Predict
nn_pred = nn_model.predict(X_test)

# Evaluate NN
nn_mse = mean_squared_error(y_test, nn_pred)
print(f'NN MSE: {nn_mse}')

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual Closing Price')
plt.plot(nn_pred, label='NN Predicted Closing Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.title('Neural Network: Actual vs Predicted Closing Prices')
plt.show()


# In[12]:


from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression MSE: {mse:.4f}')
print(f'Linear Regression R² Score: {r2:.4f}')

# Neural Network
nn_mse = mean_squared_error(y_test, nn_pred)
r2_nn = r2_score(y_test, nn_pred)
print(f'Neural Network MSE: {nn_mse:.4f}')
print(f'Neural Network R² Score: {r2_nn:.4f}')


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load real stock data (assumes you've saved it as 'real_stock_data.csv')
data = pd.read_csv('real_stock_data.csv')
data.reset_index(inplace=True)
data.rename(columns={'Date': 'Date'}, inplace=True)

# Create lag features
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Close_lag3'] = data['Close'].shift(3)

# Moving Average
data['MA5'] = data['Close'].rolling(window=5).mean()

data.dropna(inplace=True)

# Features and target
X = data[['Close_lag1', 'Close_lag2', 'Close_lag3', 'MA5']]
y = data['Close']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# Define deeper MLP
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Advanced MLP MSE: {mse:.4f}')
print(f'Advanced MLP R² Score: {r2:.4f}')

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Closing Prices')
plt.show()


# In[14]:


# Load your real stock data
data = pd.read_csv('real_stock_data.csv')

# Reset index if needed
data.reset_index(inplace=True)

# Rename Date column if needed
data.rename(columns={'Date': 'Date'}, inplace=True)

# ✅ Ensure 'Close' is numeric
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop rows with NaNs that came from bad parsing
data.dropna(subset=['Close'], inplace=True)

# Create lag features
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Close_lag3'] = data['Close'].shift(3)

# Moving Average
data['MA5'] = data['Close'].rolling(window=5).mean()

# Drop any NaNs from shifting and rolling
data.dropna(inplace=True)


# In[16]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load real stock data (assumes you've saved it as 'real_stock_data.csv')
data = pd.read_csv('real_stock_data.csv')
data.reset_index(inplace=True)
data.rename(columns={'Date': 'Date'}, inplace=True)

# Create lag features
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Close_lag3'] = data['Close'].shift(3)
# Make sure 'Close' is numeric
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Moving Average
data['MA5'] = data['Close'].rolling(window=5).mean()

data.dropna(inplace=True)

# Features and target
X = data[['Close_lag1', 'Close_lag2', 'Close_lag3', 'MA5']]
y = data['Close']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# Define deeper MLP
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Advanced MLP MSE: {mse:.4f}')
print(f'Advanced MLP R² Score: {r2:.4f}')

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Closing Prices')
plt.show()


# In[ ]:




