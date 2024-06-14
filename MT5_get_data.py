from datetime import datetime
import MetaTrader5 as mt5
# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone
import pytz

working_path = "C:/Users/david/OneDrive/Documents/DeafAbraxas/"

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)
 
# establish MetaTrader 5 connection to a specified trading account
if not mt5.initialize(path="C:/Program Files/RoboForex - MetaTrader 5/terminal64.exe"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

if not mt5.login(login=67112819, password="J1w7zhV1$ARcYU3", server="RoboForex-ECN"):
    print("login failed, error code: {}".format(mt5.last_error()))
    quit()
 
# display data on connection status, server name and trading account
print(mt5.terminal_info())
# display data on MetaTrader 5 version
print(mt5.version())
 
# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")
# create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
utc_from = datetime(2019, 1, 1, tzinfo=timezone)
utc_to = datetime(2024, 5, 31, tzinfo=timezone)
# get bars from USDJPY M5 within the interval of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone

symbol0 = "EURUSD"
symbol1 = "GBPUSD"
symbol2 = "AUDUSD"
symbol3 = "NZDUSD"
symbol4 = "USDCHF"
symbol5 = "USDJPY"

#rates = mt5.copy_rates_range(symbol0, mt5.TIMEFRAME_H4, utc_from, utc_to)
#rates = mt5.copy_rates_range(symbol1, mt5.TIMEFRAME_H4, utc_from, utc_to)
#rates = mt5.copy_rates_range(symbol2, mt5.TIMEFRAME_H4, utc_from, utc_to)
#rates = mt5.copy_rates_range(symbol3, mt5.TIMEFRAME_H4, utc_from, utc_to)
#rates = mt5.copy_rates_range(symbol4, mt5.TIMEFRAME_H4, utc_from, utc_to)
rates = mt5.copy_rates_range(symbol5, mt5.TIMEFRAME_H4, utc_from, utc_to)
 
# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
 
# display each element of obtained data in a new line
print("Display obtained data 'as is'")
counter=0
for rate in rates:
    counter+=1
    if counter<=10:
        print(rate)
 
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the 'datetime' format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
 
# display data
print("\nDisplay dataframe with data")
print(rates_frame.shape)
print(rates_frame.head(2))
print(rates_frame.tail(2))

#import to csv
#data_to_save_name = working_path + "data/" + symbol0 + "_H4.csv"
#data_to_save_name = working_path + "data/" + symbol1 + "_H4.csv"
#data_to_save_name = working_path + "data/" + symbol2 + "_H4.csv"
#data_to_save_name = working_path + "data/" + symbol3 + "_H4.csv"
#data_to_save_name = working_path + "data/" + symbol4 + "_H4.csv"
data_to_save_name = working_path + "data/" + symbol5 + "_H4.csv"

#save to csv
rates_frame.to_csv(data_to_save_name, index=False)

