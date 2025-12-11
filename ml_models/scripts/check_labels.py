import joblib
from pathlib import Path

# Path to models
MODELS_DIR = Path("models")
encoder_path = MODELS_DIR / "xlsr_label_encoder.pkl"

try:
    if encoder_path.exists():
        encoder = joblib.load(encoder_path)
        print("Speaker Labels found in model:")
        print(encoder.classes_)
    else:
        print(f"Label encoder not found at {encoder_path}")
except Exception as e:
    print(f"Error loading encoder: {e}")
