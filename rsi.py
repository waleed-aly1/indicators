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

"""
Test sample below
"""
data = pd.read_csv('./SampleData/Sample_data.csv', parse_dates=[['Date', 'Time']], index_col=['Date_Time'])
data['rsi'] = rsi(data['C'])
