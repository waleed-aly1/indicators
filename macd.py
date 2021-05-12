import pandas as pd

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

"""
Testing area below

data = pd.read_csv('./SampleData/Sample_data.csv', parse_dates=[['Date', 'Time']], index_col=['Date_Time'])
print(mac_d(data))
"""
