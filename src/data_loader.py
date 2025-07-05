import pandas as pd


# Loading data
def load_data(path):
    df = pd.read_csv(path)
    return df


# Fill or drop missing, convert types, etc.
def clean_data(df):
    return df.dropna()
