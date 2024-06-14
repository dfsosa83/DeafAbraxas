from datetime import datetime
import MetaTrader5 as mt5
# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone
import pytz

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
#timezone = pytz.timezone("Europe/Kiev") # Eastern European Summer Time (EEST) -> MetaTrader market watch (Forex timezone)
timezone = pytz.timezone("GMT") # Eastern European Summer Time (EEST)
# create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
#date_from = datetime(2024, 5, 22, tzinfo=timezone)
date_from = datetime.now(timezone)
date_from = date_from.replace(second=0)
date_to = date_from.replace(second=20)
print("Date from:", date_from, ", Date to:", date_to)
# request 50 EURUSD ticks starting from second 0
#ticks = mt5.copy_ticks_from("EURUSD", date_from, 50, mt5.COPY_TICKS_ALL)
ticks = mt5.copy_ticks_range("EURUSD", date_from, date_to, mt5.COPY_TICKS_ALL)
print("Ticks received:", len(ticks))

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()

# display data on each tick on a new line
""" print("Display obtained ticks 'as is'")
count = 0
for tick in ticks:
    count += 1
    print(tick)
    if count >= 50:
        break """

# create DataFrame out of the obtained data
ticks_frame = pd.DataFrame(ticks)
# convert time in seconds into the datetime format
ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
 
# display data
print("\nDisplay dataframe with ticks")
#print(ticks_frame.head(10))
print(ticks_frame)
