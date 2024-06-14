import os
# Disable TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from keras.models import load_model
import lightgbm as lgb
from xgboost import XGBClassifier
import MetaTrader5 as mt5
from datetime import datetime
# import pytz module for working with time zone
import pytz


import warnings
import tensorflow as tf
import keras.backend as K

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress only specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress all warnings
warnings.filterwarnings('ignore')

working_dir = "C:/Users/david/OneDrive/Documents/DeafAbraxas"  # Update this as necessary

# historical data function
def update_historical_data(df_historical, new_data):
    # Concatenate the new data with the historical dataset
    updated_data = pd.concat([df_historical, new_data], ignore_index=True)
    
    # Remove the oldest row if the dataset exceeds the desired size
    # Specify the maximum size of the historical dataset
    max_historical_size = 10000  # Adjust this value based on your requirements
    if len(updated_data) > max_historical_size:
        updated_data = updated_data.iloc[1:]
    
    return updated_data

# Define a function to preprocess the data
def features_engineering(data):
    # Check if the input is a single row or a DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data).T

    # Moving Averages
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['MA_50'] = data['close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    data['BB_std'] = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])

    # Momentum
    data['Momentum'] = data['close'] - data['close'].shift(10)

    # Stochastic Oscillator
    data['Stochastic_K'] = ((data['close'] - data['low'].rolling(window=14).min()) /
                            (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()

    # Lagged Features
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        data[f'close_lag_{lag}'] = data['close'].shift(lag)
        data[f'volume_lag_{lag}'] = data['tick_volume'].shift(lag)
        data[f'MA_10_lag_{lag}'] = data['MA_10'].shift(lag)
        data[f'RSI_lag_{lag}'] = data['RSI'].shift(lag)
        data[f'BB_middle_lag_{lag}'] = data['BB_middle'].shift(lag)

    # Price Change Features
    data['price_change_1'] = data['close'].pct_change(periods=1)
    data['price_change_5'] = data['close'].pct_change(periods=5)
    data['price_change_10'] = data['close'].pct_change(periods=10)

    # Volatility Features
    data['volatility_10'] = data['close'].rolling(window=10).std()
    data['volatility_20'] = data['close'].rolling(window=20).std()
    data['volatility_50'] = data['close'].rolling(window=50).std()

    # Average True Range (ATR)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    data['ATR'] = ranges.max(axis=1).rolling(window=14).mean()

    # Drop rows with missing values
    data.dropna(inplace=True)

    return data

# Define the f1_m function if it's a custom metric
def f1_m(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

selected_features_names = [
    'open', 'high', 'MA_50', 'BB_std', 'BB_upper', 'BB_lower', 'Momentum',
       'close_lag_1', 'close_lag_2', 'close_lag_3', 'BB_middle_lag_5',
       'MA_10_lag_10', 'BB_middle_lag_10', 'price_change_1', 'price_change_5',
       'price_change_10', 'volatility_10', 'volatility_20', 'volatility_50',
       'ATR'
]

# Load the trained models
#model1
with open(os.path.join(working_dir, "models/30m/deaf_abrax_30m_lg_model.sav"), "rb") as file:
    model1 = pickle.load(file)

#model2
with open(os.path.join(working_dir, "models/30m/deaf_abrax_30m_xg_model.sav"), "rb") as file:
    model2 = pickle.load(file)

#model3
    model3 = load_model('C:/Users/david/OneDrive/Documents/DeafAbraxas/models/30m/deaf_abrax_30m_lstm_model.hdf5', custom_objects={'f1_m': f1_m})

# Load the meta model
with open(os.path.join(working_dir, "models/30m/deaf_abrax_30m_meta_model.sav"), "rb") as file:
    best_meta_model = pickle.load(file)

# Load the conformal threshold
with open(os.path.join(working_dir, "models/30m/deaf_abrax_30m_conformal_thr.sav"), "rb") as file:
    threshold = pickle.load(file)


# read data

# display data on the MetaTrader 5 package
#print("MetaTrader5 package author: ", mt5.__author__)
#print("MetaTrader5 package version: ", mt5.__version__)
# 
## establish MetaTrader 5 connection to a specified trading account
#if not mt5.initialize(path="C:/Program Files/RoboForex - MetaTrader 5/terminal64.exe"):
#    print("initialize() failed, error code =", mt5.last_error())
#    quit()
#
#if not mt5.login(login=67112819, password="J1w7zhV1$ARcYU3", server="RoboForex-ECN"):
#    print("login failed, error code: {}".format(mt5.last_error()))
#    quit()
# 
## display data on connection status, server name and trading account
#print(mt5.terminal_info())
## display data on MetaTrader 5 version
#print(mt5.version())
# 
## set time zone to UTC
#timezone = pytz.timezone("Etc/UTC")
## create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
#utc_from = datetime(2019, 1, 1, tzinfo=timezone)
#utc_to = datetime(2024, 5, 31, tzinfo=timezone) # datetime.now(tz=timezone)
#
#symbol0 = "EURUSD"
#rates = mt5.copy_rates_range(symbol0, mt5.TIMEFRAME_H4, utc_from, utc_to)
#
## shut down connection to the MetaTrader 5 terminal
#mt5.shutdown()
# 
## display each element of obtained data in a new line
#print("Display obtained data 'as is'")
#counter=0
#for rate in rates:
#    counter+=1
#    if counter<=10:
#        print(rate)
# 
## create DataFrame out of the obtained data
#rates_frame = pd.DataFrame(rates)
## convert time in seconds into the 'datetime' format
#rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
# 
## display data
#print("\nDisplay dataframe with data")
#print(rates_frame.shape)
#print(rates_frame.head(2))
#print(rates_frame.tail(2))
#
##import to csv
#
#data_to_predict = rates_frame.copy()
#print(data_to_predict.head(2))

#read data 
df_historical = pd.read_csv("C:/Users/david/OneDrive/Documents/DeafAbraxas/data/30M/EURUSD_30M.csv", nrows=10000)
print(df_historical.shape, 'rows in dataset with historical data')

new_data_to_predict = df_historical.tail(1) #here i need to adapt metatrader 

# Update the historical dataset with the new data
df_historical = update_historical_data(df_historical, new_data_to_predict)
print(df_historical.shape, 'rows in dataset with historical data and new data')

new_data = df_historical.drop(columns=['real_volume'], axis=1)
#convert datatime to_datetime
new_data.rename(columns={"time": "datetime"}, inplace=True)
new_data['datetime'] = pd.to_datetime(new_data['datetime'])
print(new_data.shape)

features_df = features_engineering(new_data)
new_data_features = features_df[selected_features_names]
print(new_data_features.shape)

# Predict probabilities for each base model
lg_preds = model1.predict_proba(new_data_features)
xg_preds = model2.predict_proba(new_data_features)
lstm_preds = model3.predict(new_data_features)

# Combine these probabilities into a single input for the meta-model
combined_preds = np.hstack([lg_preds, xg_preds, lstm_preds])

# Predict class probabilities with the meta-model
final_proba = best_meta_model.predict_proba(combined_preds)
#print(final_proba)
last_proba = final_proba[-1]
print(last_proba)


# Apply Conformal Prediction
def apply_conformal_prediction(probs, threshold):
    conforms = np.max(probs) >= threshold
    return conforms

conformal_intervals = apply_conformal_prediction(last_proba, threshold)

print("Final Probabilities:")
print(last_proba)
print("Conformal Intervals:")
print(str(last_proba[0]) + "|" + str(last_proba[1]) + "|" + str(last_proba[2]) + "|" + str(conformal_intervals))




