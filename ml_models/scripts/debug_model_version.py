import joblib
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from importlib.machinery import SourceFileLoader
import types

# Load feature extractor class dynamically
loader = SourceFileLoader("wavlm_feature_extraction", str(SRC_DIR / "2_wavlm_feature_extraction.py"))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)
XLSRFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")

MODELS_DIR = ROOT_DIR / "models"

def debug_model_version():
    print("Loading artifacts...")
    try:
        classifier = joblib.load(MODELS_DIR / "xlsr_classifier.pkl")
        pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
        encoder = joblib.load(MODELS_DIR / "xlsr_label_encoder.pkl")
        scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
        
        # Patch scaler
        if hasattr(scaler, "scale_"):
            scaler.scale_[scaler.scale_ < 1e-6] = 1.0

        # Try original XLSR
        model_name = "facebook/wav2vec2-large-xlsr-53"
        print(f"Testing model: {model_name}")
        
        extractor = XLSRFeatureExtractor(model_name=model_name)
        
        # Use the existing debug file or download it
        filename = "1221_debug.flac"
        file_path = Path(filename)
        if not file_path.exists():
            print(f"File {filename} not found. Downloading...")
            from datasets import load_dataset, Audio
            ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            target_speaker = 1221
            for sample in ds:
                if sample['speaker_id'] == target_speaker:
                    with open(filename, "wb") as f:
                        f.write(sample['audio']['bytes'])
                    print(f"Downloaded {filename}")
                    break

        print(f"\n--- Testing {model_name} for {filename} (True Label: 1221) ---")
        
        # Extract features using the new custom pooling
        emb = extractor.extract_from_file(str(file_path), pooling="xlsr_custom")
        
        # PCA
        emb_reshaped = emb.reshape(1, -1)
        emb_pca = pca.transform(emb_reshaped)
        
        # Scaler
        emb_pca = scaler.transform(emb_pca)
        
        # Predict
        probs = classifier.predict_proba(emb_pca)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        print("Top 3 Probabilities:")
        for idx in top3_idx:
            label = encoder.inverse_transform([idx])[0] if idx < len(encoder.classes_) else idx
            print(f"  Label {label}: {probs[idx]:.4f}")
            
        pred_raw = int(classifier.predict(emb_pca)[0])
        if pred_raw < len(encoder.classes_):
             pred_label = str(encoder.inverse_transform([pred_raw])[0])
        else:
             pred_label = str(pred_raw)
        print(f"Prediction: {pred_label}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_version()
