import joblib
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
OUTPUT_FILE = Path("scaler_info.txt")

def check_scaler():
    with open(OUTPUT_FILE, "w") as f:
        try:
            f.write("Loading scaler...\n")
            scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
            f.write(f"Scaler type: {type(scaler)}\n")
            
            if hasattr(scaler, "mean_"):
                f.write(f"Scaler mean_ shape: {scaler.mean_.shape}\n")
            elif hasattr(scaler, "center_"):
                 f.write(f"Scaler center_ shape: {scaler.center_.shape}\n")
            else:
                f.write("Scaler has no mean_ or center_\n")

            f.write("Loading PCA...\n")
            pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
            f.write(f"PCA n_components_: {pca.n_components_}\n")
            f.write(f"PCA n_features_: {pca.n_features_}\n")

        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    check_scaler()
