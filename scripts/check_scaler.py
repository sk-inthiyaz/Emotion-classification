import joblib
from pathlib import Path

MODELS_DIR = Path("models")

def check_scaler():
    try:
        scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
        print(f"Scaler type: {type(scaler)}")
        
        if hasattr(scaler, "n_features_in_"):
            print(f"n_features_in_: {scaler.n_features_in_}")
        elif hasattr(scaler, "mean_"):
            print(f"mean_ shape: {scaler.mean_.shape}")
        elif hasattr(scaler, "scale_"):
            print(f"scale_ shape: {scaler.scale_.shape}")
        else:
            print("Could not determine input shape.")
            
        pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
        print(f"PCA n_components: {pca.n_components_}")
        print(f"PCA n_features_: {pca.n_features_}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_scaler()
