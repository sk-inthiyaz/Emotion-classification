import joblib
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models")

def check_zero_scale():
    with open("zero_scale_results.txt", "w") as f:
        try:
            scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
            scale = scaler.scale_
            
            zeros = np.where(scale == 0)[0]
            near_zeros = np.where(np.abs(scale) < 1e-6)[0]
            
            f.write(f"Total components: {len(scale)}\n")
            f.write(f"Exact zeros: {len(zeros)}\n")
            f.write(f"Near zeros (< 1e-6): {len(near_zeros)}\n")
            
            if len(near_zeros) > 0:
                f.write(f"Indices of near zeros: {near_zeros}\n")
                f.write(f"Values at these indices: {scale[near_zeros]}\n")

        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    check_zero_scale()
