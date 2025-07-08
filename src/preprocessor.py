"""Preprocess CMAPSS data and cache features."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import requests


def download_cmapss_if_missing(data_dir: str):
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    target = os.path.join(raw_dir, "train_FD001.txt")
    if os.path.exists(target):
        print("[Preprocessor] CMAPSS data already exists.")
        return

    # List of candidate URLs (primary + fallbacks)
    urls = [
        # Reliable mirror: GitHub raw from mapr-demos repository
        "https://raw.githubusercontent.com/mapr-demos/predictive-maintenance/master/notebooks/jupyter/Dataset/CMAPSSData/train_FD001.txt",
        # Kaggle mirror via raw GitHub
        "https://raw.githubusercontent.com/center-for-ml/CMAPSS/master/data/train_FD001.txt",
        # Analytics Vidhya reference via ti.arc.nasa.gov
        "https://ti.arc.nasa.gov/c/6/",
    ]

    for url in urls:
        print(f"[Preprocessor] Attempting download from: {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # Quick size check
            if len(resp.content) < 1000:
                print(f"[Preprocessor] Warning: Downloaded content too small ({len(resp.content)} bytes), skipping")
                continue

            with open(target, "wb") as f:
                f.write(resp.content)
            print(f"[Preprocessor] Download successful from: {url}")
            return

        except Exception as e:
            print(f"[Preprocessor] Error downloading from {url}: {e}")

    raise RuntimeError(
        "🎯 Failed to download train_FD001.txt from all sources. "
        "Please download manually and place it in data/raw/"
    )


def load_cmapss_raw(data_dir):
    download_cmapss_if_missing(data_dir)
    print("[Preprocessor] Loading CMAPSS raw data...")
    df = pd.read_csv(os.path.join(data_dir, "raw", "train_FD001.txt"), sep=' ', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["unit", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in
                                                                         range(1, df.shape[1] - 4)]
    return df


def windowed_features(df, window_size=30):
    print("[Preprocessor] Creating windowed features...")
    features = []
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit].reset_index(drop=True)
        for i in range(len(unit_df) - window_size):
            window = unit_df.iloc[i:i + window_size, 2:].values  # skip 'unit' and 'cycle'
            features.append(window)
    return np.array(features)


def preprocess_and_save(config):
    raw_path = config['data']['raw_path']
    processed_path = config['data']['processed_path']
    os.makedirs(processed_path, exist_ok=True)

    df = load_cmapss_raw(raw_path)
    data = windowed_features(df)
    original_shape = data.shape
    data_reshaped = data.reshape(-1, data.shape[-1])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped).reshape(original_shape)

    np.save(config['data']['features_cache'], data_scaled)

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/preprocessor.pkl')

    print(f"[Preprocessor] Saved features to {config['data']['features_cache']}, shape: {data_scaled.shape}")
    print("[Preprocessor] Preprocessor saved to models/preprocessor.pkl")