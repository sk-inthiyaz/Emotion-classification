"""
Standalone test script for emotion inference
Tests HuBERT-large + SVM pipeline on a single audio file
"""

import sys
from pathlib import Path
import torch
import joblib
import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
sys.path.insert(0, str(SRC_DIR))

print("=" * 60)
print("Emotion Inference Test")
print("=" * 60)

# 1. Load feature extractor
print("\n1. Loading HuBERT-large feature extractor...")
try:
    from importlib.machinery import SourceFileLoader
    import types
    
    module_path = SRC_DIR / "2_wavlm_feature_extraction.py"
    loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    
    WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
    
    # Use HuBERT-large for best accuracy
    extractor = WavLMFeatureExtractor(model_name="facebook/hubert-large-ll60k")
    print("✓ Feature extractor loaded successfully")
except Exception as e:
    print(f"✗ Error loading feature extractor: {e}")
    sys.exit(1)

# 2. Load model artifacts
print("\n2. Loading SVM model and preprocessors...")
try:
    model_path = MODELS_DIR / "emotion_model_svm.pkl"
    scaler_path = MODELS_DIR / "emotion_scaler.pkl"
    encoder_path = MODELS_DIR / "emotion_label_encoder.pkl"
    
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        sys.exit(1)
    
    classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    print(f"✓ SVM classifier loaded")
    print(f"✓ Scaler loaded")
    print(f"✓ Label encoder loaded")
    print(f"  Classes: {list(encoder.classes_)}")
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    sys.exit(1)

# 3. Get audio file path
print("\n3. Getting audio file path...")
if len(sys.argv) > 1:
    audio_path = Path(sys.argv[1])
else:
    # Prompt user for file
    audio_file = input("Enter path to audio file (or press Enter to skip): ").strip()
    # Remove quotes if present
    audio_file = audio_file.strip('"').strip("'")
    if not audio_file:
        print("No file provided. Exiting.")
        sys.exit(0)
    audio_path = Path(audio_file)

if not audio_path.exists():
    print(f"✗ Audio file not found: {audio_path}")
    sys.exit(1)

print(f"✓ Audio file: {audio_path}")

# 4. Extract embeddings
print("\n4. Extracting HuBERT-large embeddings (this may take 5-10 seconds)...")
try:
    embedding = extractor.extract_from_file(str(audio_path))
    embedding = embedding.reshape(1, -1)
    print(f"✓ Embedding extracted, shape: {embedding.shape}")
except Exception as e:
    print(f"✗ Error extracting embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Scale embeddings
print("\n5. Scaling embeddings...")
try:
    embedding_scaled = scaler.transform(embedding)
    print(f"✓ Embeddings scaled")
except Exception as e:
    print(f"✗ Error scaling embeddings: {e}")
    sys.exit(1)

# 6. Predict emotion
print("\n6. Predicting emotion with SVM...")
try:
    pred_idx = int(classifier.predict(embedding_scaled)[0])
    pred_label = str(encoder.inverse_transform([pred_idx])[0])
    
    # Get probabilities
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(embedding_scaled)[0]
    elif hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(embedding_scaled)[0]
        exp_decision = np.exp(decision - np.max(decision))
        probs = exp_decision / exp_decision.sum()
    else:
        probs = np.ones(len(encoder.classes_)) / len(encoder.classes_)
    
    print(f"✓ Prediction complete")
except Exception as e:
    print(f"✗ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Display results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nPredicted Emotion: {pred_label.upper()}")
print(f"\nClass Probabilities:")
for i, label in enumerate(encoder.classes_):
    bar_length = int(probs[i] * 40)
    bar = "█" * bar_length + "░" * (40 - bar_length)
    print(f"  {label:12s} [{bar}] {probs[i]*100:5.1f}%")

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
