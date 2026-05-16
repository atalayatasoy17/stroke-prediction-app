import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/atalayatasoy17/stroke-prediction-app/refs/heads/main/data/healthcare-dataset-stroke-data.csv"

def fetch_data():
    print("Veri indiriliyor...")
    df = pd.read_csv(DATA_URL)
    print(f"Veri yüklendi. Satır: {df.shape[0]}, Sütun: {df.shape[1]}")
    return df

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
