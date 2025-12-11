import joblib
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")

def inspect_models():
    with open("inspect_results.txt", "w") as f:
        f.write("--- Inspecting Models ---\n")
        try:
            pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
            scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
            classifier = joblib.load(MODELS_DIR / "xlsr_classifier.pkl")

            f.write(f"\nPCA:\n")
            f.write(f"  n_components: {pca.n_components_}\n")
            if hasattr(pca, "n_features_in_"):
                f.write(f"  n_features_in_: {pca.n_features_in_}\n")
            f.write(f"  Components stats: Min={pca.components_.min():.4f}, Max={pca.components_.max():.4f}, Mean={pca.components_.mean():.4f}\n")
            
            f.write(f"\nScaler:\n")
            f.write(f"  Mean stats: Min={scaler.mean_.min():.4f}, Max={scaler.mean_.max():.4f}\n")
            f.write(f"  Scale stats: Min={scaler.scale_.min():.4f}, Max={scaler.scale_.max():.4f}\n")
            
            f.write(f"\nClassifier ({type(classifier).__name__}):\n")
            if hasattr(classifier, "coef_"):
                f.write(f"  Coef shape: {classifier.coef_.shape}\n")
                f.write(f"  Coef stats: Min={classifier.coef_.min():.4f}, Max={classifier.coef_.max():.4f}\n")
            if hasattr(classifier, "intercept_"):
                f.write(f"  Intercept stats: Min={classifier.intercept_.min():.4f}, Max={classifier.intercept_.max():.4f}\n")

        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    inspect_models()
