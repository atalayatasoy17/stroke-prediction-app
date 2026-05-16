import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/atalayatasoy17/stroke-prediction-app/main/data/healthcare-dataset-stroke-data.csv"

def load_data():
    df = pd.read_csv(DATA_URL)
    return df