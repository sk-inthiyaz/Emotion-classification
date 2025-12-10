import joblib
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")

def check_scaler():
    try:
        print("Loading scaler...")
        scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
        print(f"Scaler type: {type(scaler)}")
        
        if hasattr(scaler, "mean_"):
            print(f"Scaler mean_ shape: {scaler.mean_.shape}")
        elif hasattr(scaler, "center_"):
             print(f"Scaler center_ shape: {scaler.center_.shape}")
        else:
            print("Scaler has no mean_ or center_")

        print("Loading PCA...")
        pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
        print(f"PCA n_components_: {pca.n_components_}")
        print(f"PCA n_features_: {pca.n_features_}") # Input features to PCA

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_scaler()
