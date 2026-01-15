import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df
