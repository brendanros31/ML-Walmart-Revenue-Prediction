import pandas as pd


# Feature Extraction for Date-Time
def featExtract(df, Date=True):
    if Date == True:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df