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
import torchaudio # Still needed for resampling if we want, or we can use scipy/librosa

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Load WavLMFeatureExtractor
loader = SourceFileLoader("wavlm_feature_extraction", str(SRC_DIR / "2_wavlm_feature_extraction.py"))
mod = types.ModuleType(loader.name)
loader.exec_module(mod)
WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")

def train_gender_model():
    print("Initializing Gender Model Training...")
    
    # 1. Load Dataset (LibriSpeech dev-clean)
    # We assume it's already downloaded or we try to download it using torchaudio just to get the files
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dataset_path = DATA_DIR / "LibriSpeech/dev-clean"
    if not dataset_path.exists():
        print("Dataset not found, attempting download...")
        try:
            torchaudio.datasets.LIBRISPEECH(str(DATA_DIR), url="dev-clean", download=True)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            # If download failed but folder exists, we might proceed?
            # But likely it failed.
            pass

    # 2. Map Gender
    gender_map = {}
    spk_file = DATA_DIR / "LibriSpeech/SPEAKERS.TXT"
    
    if not spk_file.exists():
        # Try to find it
        possible_paths = list(DATA_DIR.glob("**/SPEAKERS.TXT"))
        if possible_paths:
            spk_file = possible_paths[0]
            print(f"Found speaker file at {spk_file}")
        else:
            print("Could not locate SPEAKERS.TXT")
            return

    with open(spk_file) as f:
        for line in f:
            if line.startswith(";"):
                continue
            p = line.split()
            if len(p) >= 3:
                gender_map[int(p[0])] = 0 if p[2].strip() == "M" else 1 # 0=Male, 1=Female

    print(f"Speakers with gender: {len(gender_map)}")

    # 3. Build Dataset (Manual Walk)
    print("Scanning dataset (manual walk)...")
    all_samples = []
    
    if not dataset_path.exists():
         print(f"Dataset path still not found: {dataset_path}")
         return

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                # Filename format: SPEAKER-CHAPTER-ID.flac
                parts = file.split("-")
                if len(parts) >= 3:
                    try:
                        spk = int(parts[0])
                        if spk in gender_map:
                            all_samples.append({
                                "path": str(Path(root) / file),
                                "spk": spk,
                                "gender": gender_map[spk]
                            })
                    except:
                        pass
    
    print(f"Found {len(all_samples)} valid samples.")
    
    if len(all_samples) == 0:
        print("No samples found! Exiting.")
        return

    male = [x for x in all_samples if x["gender"] == 0]
    female = [x for x in all_samples if x["gender"] == 1]
    
    # Use up to 500 samples per class (1000 total) for reasonable training time on CPU
    N = min(500, len(male), len(female))
    print(f"Balancing dataset: {N} Male + {N} Female (Total {2*N})")
    
    dataset_indices = random.sample(male, N) + random.sample(female, N)
    random.shuffle(dataset_indices)
    
    # 4. Extract features
    print("Extracting features...")
    extractor = WavLMFeatureExtractor(model_name="microsoft/wavlm-base")
    
    X = []
    y = []
    spk_ids = []
    
    for item_meta in tqdm(dataset_indices, desc="Extracting"):
        path = item_meta["path"]
        try:
            # Load audio using soundfile
            wav, sr = sf.read(path)
            wav = torch.from_numpy(wav).float()
            
            # Resample if needed (WavLM expects 16k)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                
            # Ensure 1D
            if wav.ndim > 1:
                wav = wav.mean(dim=0)
                
            # Extract embedding
            emb = extractor.extract_embedding(wav, pooling="mean")
            
            # L2 Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            X.append(emb)
            y.append(item_meta["gender"])
            spk_ids.append(item_meta["spk"])
        except Exception as e:
            print(f"Error extracting features for {path}: {e}")

    X = np.array(X)
    y = np.array(y)
    spk_ids = np.array(spk_ids)
    
    print(f"Features shape: {X.shape}")
    
    if len(X) == 0:
        print("No features extracted. Exiting.")
        return

    # 5. Split Train/Test by Speaker
    unique_spk = list(set(spk_ids))
    random.shuffle(unique_spk)
    test_spk_set = set(unique_spk[: int(len(unique_spk) * 0.25)])
    
    train_idx = [i for i, s in enumerate(spk_ids) if s not in test_spk_set]
    test_idx = [i for i, s in enumerate(spk_ids) if s in test_spk_set]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 6. Pipeline: Scaler -> PCA -> LogisticRegression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # PCA
    n_components = min(50, X_train_s.shape[1], X_train_s.shape[0] - 1)
    print(f"PCA Components: {n_components}")
    pca = PCA(n_components=n_components)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    # Classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_p, y_train)
    
    # Evaluate
    pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, pred)
    print(f"\nGender Model Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred, target_names=["Male", "Female"]))
    
    # 7. Save Artifacts
    print("Saving models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(clf, MODELS_DIR / "gender_classifier.pkl")
    joblib.dump(scaler, MODELS_DIR / "gender_scaler.pkl")
    joblib.dump(pca, MODELS_DIR / "gender_pca.pkl")
    
    # Label Encoder
    le = LabelEncoder()
    le.classes_ = np.array(["Male", "Female"]) # 0=Male, 1=Female
    joblib.dump(le, MODELS_DIR / "gender_label_encoder.pkl")
    
    print(f"Artifacts saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_gender_model()
