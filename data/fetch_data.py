import os
import pandas as pd


def fetch_data(output_dir: str = "data") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "healthcare-dataset-stroke-data.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Veri zaten mevcut: {csv_path}")
        return pd.read_csv(csv_path)

    print("[INFO] Kaggle'dan veri indiriliyor...")
    os.system(
        f"kaggle datasets download -d fedesoriano/stroke-prediction-dataset "
        f"--path {output_dir} --unzip"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Veri indirilemedi. Kaggle API key kontrol edin.")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
    print(f"\nEksik değerler:\n{df.isnull().sum()}")
    print(f"\nHedef dağılımı:\n{df['stroke'].value_counts()}")