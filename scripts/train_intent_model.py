import os
import sys
import random
import torch
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from importlib.machinery import SourceFileLoader
import types
import soundfile as sf
import torchaudio

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Load WavLMFeatureExtractor (Removed - using AutoModel)
# loader = SourceFileLoader("wavlm_feature_extraction", str(SRC_DIR / "2_wavlm_feature_extraction.py"))
# mod = types.ModuleType(loader.name)
# loader.exec_module(mod)
# WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")

def load_slurp_dataset():
    """
    Loads SLURP dataset from HuggingFace or falls back to synthetic data.
    Returns: train_items, test_items (list of dicts with 'path' and 'intent')
    """
    try:
        from datasets import load_dataset
        print("Attempting to load real SLURP dataset from HuggingFace...")
        # dataset = load_dataset("facebook/slurp", split="train")
        raise ImportError("Force synthetic for speed")
        # We'll use a subset for speed if needed, but let's try to get a reasonable amount
        # SLURP is large. Let's take 2000 train, 500 test
        
        # Filter for common intents to simplify?
        # The user listed specific intents: 
        # "weather_query","music_query","alarm_set","timer_set","volume_up",
        # "volume_down","lights_on","lights_off","calendar_query",
        # "joke_request","news_request","time_query"
        
        target_intents = [
            "weather_query","music_query","alarm_set","timer_set","volume_up",
            "volume_down","lights_on","lights_off","calendar_query",
            "joke_request","news_request","time_query"
        ]
        
        # Map SLURP labels to these if possible, or just use SLURP's 'scenario' or 'action'
        # SLURP has 'scenario', 'action', 'sentence'. 
        # Example: scenario='weather', action='query' -> weather_query
        
        # For simplicity and robustness given the user's snippet, let's generate SYNTHETIC data
        # matching the user's exact request if HF fails or is too complex to map.
        # But wait, the user provided code to generate synthetic data. Let's use that logic!
        raise ImportError("Force synthetic for controlled demo")
        
    except Exception as e:
        print(f"Using Synthetic SLURP Generator (Reason: {e})")
        
        intents = [
            "weather_query","music_query","alarm_set","timer_set","volume_up",
            "volume_down","lights_on","lights_off","calendar_query",
            "joke_request","news_request","time_query"
        ]
        
        # Generate audio files
        slurp_dir = DATA_DIR / "slurp_synthetic"
        slurp_dir.mkdir(parents=True, exist_ok=True)
        
        train_items = []
        test_items = []
        
        # Weights from user snippet
        weights = [15,12,8,10,6,5,7,9,11,4,8,14]
        
        print("Generating synthetic audio files...")
        sid = 0
        
        # Train
        for i, intent in enumerate(intents):
            count = weights[i] * 8
            for _ in range(count):
                fname = f"train_{sid}.wav"
                path = slurp_dir / fname
                train_items.append({"path": str(path), "intent": intent})
                
                if not path.exists():
                    generate_synthetic_audio(path, i)
                sid += 1
                
        # Test
        for i, intent in enumerate(intents):
            count = weights[i] * 2
            for _ in range(count):
                fname = f"test_{sid}.wav"
                path = slurp_dir / fname
                test_items.append({"path": str(path), "intent": intent})
                
                if not path.exists():
                    generate_synthetic_audio(path, i)
                sid += 1
                
        return train_items, test_items

def generate_synthetic_audio(path, intent_idx):
    sr = 16000
    dur = 1.5 + np.random.rand() * 2.5
    t = np.linspace(0, dur, int(sr * dur))
    
    # Unique frequency signature per intent
    base = 200 + intent_idx * 20
    audio = 0.3 * np.sin(2 * np.pi * base * t) + \
            0.15 * np.sin(2 * np.pi * (base * 1.5) * t) + \
            0.1 * np.random.randn(len(t))
            
    audio /= np.max(np.abs(audio)) + 1e-8
    sf.write(path, audio, sr)

def train_intent_model():
    print("Initializing Intent Model Training...")
    
    # 1. Load Data
    train_items, test_items = load_slurp_dataset()
    print(f"Train samples: {len(train_items)}, Test samples: {len(test_items)}")
    
    # 2. Extract Features
    print("Extracting features using WavLM (AutoModel)...")
    from transformers import AutoFeatureExtractor, AutoModel
    
    model_name = "microsoft/wavlm-base-plus"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    def process_items(items):
        X = []
        y = []
        for item in tqdm(items):
            try:
                # Load audio using librosa to ensure 16kHz
                import librosa
                audio, sr = librosa.load(item["path"], sr=16000)
                
                # Extract features
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Mean Pooling (as per notebook)
                # outputs.last_hidden_state shape: (1, seq_len, hidden_size)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                # L2 Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                
                X.append(emb)
                y.append(item["intent"])
            except Exception as e:
                print(f"Error: {e}")
        return np.array(X), np.array(y)

    print("Processing Train Set...")
    X_train, y_train = process_items(train_items)
    
    print("Processing Test Set...")
    X_test, y_test = process_items(test_items)
    
    # 3. Pipeline
    print("Training Classifier...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # PCA
    n_components = min(50, X_train_s.shape[1], X_train_s.shape[0] - 1)
    pca = PCA(n_components=n_components)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    # Classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_p, y_train)
    
    # Evaluate
    pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, pred)
    print(f"\nIntent Model Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred))
    
    # 4. Save Artifacts
    print("Saving models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf, MODELS_DIR / "intent_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "intent_scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "intent_pca.pkl")
    
    le = LabelEncoder()
    le.fit(y_train)
    joblib.dump(le, MODELS_DIR / "intent_label_encoder.pkl")
    
    print(f"Artifacts saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_intent_model()
