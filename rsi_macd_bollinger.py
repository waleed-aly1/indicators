import pandas as pd

def rsi(prices, window_length=14):
    delta = prices.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    #delta = delta[1:]
    delta.iloc[0] = 0

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    RSI1.iloc[:2] = 50

    return RSI1

def mac_d(dataset,column=['C']):

    dataset['26ema'] = dataset[column].ewm(span=26).mean()
    close = column
    dataset['12ema'] = dataset[close].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    dataset['MACD_signal'] = dataset['MACD'].rolling(window=8).mean()

    dataset['MACD_histogram'] = dataset['MACD'] - dataset['MACD_signal']

    dataset['stoch_fast_K'] = 100 * ((dataset[close] - dataset[close].rolling(window=14).min()) /
                                     (dataset[close].rolling(window=14).max() - dataset[close].rolling(
                                         window=14).min()))

    dataset['stoch_slow_K'] = dataset['stoch_fast_K'].rolling(window=3).mean()

    dataset['stoch_D'] = dataset['stoch_slow_K'].rolling(window=3).mean()

    return dataset

def bollinger_bands(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['close'].rolling(window=21).mean()
    dataset['20sd'] = dataset['close'].rolling_std(20)

    # Create Bollinger Bands
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    return dataset

"""
Testing area below

data = pd.read_csv('Sample_data.csv', parse_dates=[['Date', 'Time']], index_col=['Date_Time'])
print(mac_d(data))
"""
