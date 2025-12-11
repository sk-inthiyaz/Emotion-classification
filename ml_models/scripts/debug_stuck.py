import joblib
import torch
import numpy as np
import sys
from pathlib import Path
import glob

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

def debug_stuck_prediction():
    print("Loading artifacts...")
    try:
        classifier = joblib.load(MODELS_DIR / "xlsr_classifier.pkl")
        pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
        encoder = joblib.load(MODELS_DIR / "xlsr_label_encoder.pkl")
        scaler = joblib.load(MODELS_DIR / "xlsr_scaler.pkl")
        # Patch scaler to prevent division by zero/near-zero
        # Set scale to a huge number to force the output to be 0 (matching training distribution of 0 variance)
        if hasattr(scaler, "scale_"):
            scaler.scale_[scaler.scale_ < 1e-6] = 1e10
            
        extractor = XLSRFeatureExtractor(model_name="facebook/wav2vec2-large-xlsr-53")
        
        # Download a sample locally for testing
        print("Downloading a sample for testing...")
        from datasets import load_dataset, Audio
        ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        
        sample_files = []
        target_speaker = 1221 # Try a different speaker
        
        for sample in ds:
            if sample['speaker_id'] == target_speaker:
                filename = f"{target_speaker}_debug.flac"
                with open(filename, "wb") as f:
                    f.write(sample['audio']['bytes'])
                sample_files.append(Path(filename))
                print(f"Downloaded {filename}")
                break
                
        if not sample_files:
            print("Could not download sample.")
            return

        print(f"\nTesting with {len(sample_files)} files:")
        
        for file_path in sample_files:
            print(f"\n--- Processing {file_path.name} ---")
            
            # 1. Feature Extraction
            emb = extractor.extract_from_file(str(file_path), pooling="mean_max")
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding stats: Mean={emb.mean():.4f}, Std={emb.std():.4f}, Min={emb.min():.4f}, Max={emb.max():.4f}")
            
            # 2. PCA
            emb_reshaped = emb.reshape(1, -1)
            emb_pca = pca.transform(emb_reshaped)
            
            # 2.5 Scaler
            emb_pca = scaler.transform(emb_pca)
            
            print(f"PCA shape: {emb_pca.shape}")
            print(f"PCA stats: Mean={emb_pca.mean():.4f}, Std={emb_pca.std():.4f}")
            
            # 3. Prediction
            pred_raw = int(classifier.predict(emb_pca)[0])
            
            if hasattr(classifier, "predict_proba"):
                probs = classifier.predict_proba(emb_pca)[0]
                top3_idx = np.argsort(probs)[-3:][::-1]
                print("Top 3 Probabilities:")
                for idx in top3_idx:
                    label = encoder.inverse_transform([idx])[0] if idx < len(encoder.classes_) else idx
                    print(f"  Label {label}: {probs[idx]:.4f}")
            
            print(f"Raw Prediction: {pred_raw}")
            
            # Decode label
            if pred_raw < len(encoder.classes_):
                 pred_label = str(encoder.inverse_transform([pred_raw])[0])
            else:
                 pred_label = str(pred_raw)
            
            print(f"Final Label: {pred_label}")
            
            # Clean up
            try:
                file_path.unlink()
            except:
                pass

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_stuck_prediction()
