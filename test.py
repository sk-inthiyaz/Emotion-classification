import joblib
from pathlib import Path

MODELS_DIR = Path("models")

# Paths
clf_path = MODELS_DIR / "intent_model_svm.pkl"
scaler_path = MODELS_DIR / "intent_scaler.pkl"
encoder_path = MODELS_DIR / "intent_label_encoder.pkl"

try:
    clf = joblib.load(clf_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    print("ALL FILES LOADED SUCCESSFULLY!")
    print("Encoder:", type(encoder), encoder)
except Exception as e:
    print("ERROR:", e)


import numpy as np

# Fake embedding vector same size as your real embeddings
fake_emb = np.random.randn(1, 512)

# Scale
fake_scaled = scaler.transform(fake_emb)

# Predict
pred = clf.predict(fake_scaled)
print("Pred Index:", pred)

# Check decoding (dict-based)
inv = {v:k for k,v in encoder.items()}
print("Pred Label:", inv[pred[0]])


