"""
Configuration Module - Backend
Centralized settings for the Emotion Classification Flask application.
Contains paths (pointing to new clean folder structure), model configurations, 
and application constants.

FOLDER STRUCTURE:
├── backend/
│   ├── config.py (this file)
│   ├── app/
│   │   ├── app.py
│   │   ├── templates/
│   │   ├── static/
│   │   └── uploads/
│   └── services/
│       ├── emotion.py
│       ├── gender.py
│       ├── intent.py
│       ├── speaker.py
│       └── utils/
│
├── ml_models/
│   ├── models/
│   ├── data/
│   ├── src/
│   └── scripts/
│
└── docs/
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS (Updated for new folder structure)
# ============================================================================

# Root directory of the project (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Backend directory
BACKEND_DIR = PROJECT_ROOT / "backend"

# ML Models and training directory
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"

# ML subdirectories
SRC_DIR = ML_MODELS_DIR / "src"
MODELS_DIR = ML_MODELS_DIR / "models"
DATA_DIR = ML_MODELS_DIR / "data"
RESULTS_DIR = ML_MODELS_DIR / "results"
SCRIPTS_DIR = ML_MODELS_DIR / "scripts"
EMBEDDINGS_DIR = ML_MODELS_DIR / "embeddings"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

# Backend directories
APP_DIR = BACKEND_DIR / "app"
SERVICES_DIR = BACKEND_DIR / "services"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
UPLOADS_DIR = APP_DIR / "uploads"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Emotion Model Configuration
EMOTION_MODEL = {
    "feature_extractor": "facebook/hubert-large-ll60k",
    "classifier": str(MODELS_DIR / "emotion_model_svm.pkl"),
    "scaler": str(MODELS_DIR / "emotion_scaler.pkl"),
    "label_encoder": str(MODELS_DIR / "emotion_label_encoder.pkl"),
    "accuracy": 0.7914,  # 79.14% on CREMA-D
    "description": "HuBERT-large + SVM classifier"
}

# Gender Model Configuration
GENDER_MODEL = {
    "feature_extractor": "microsoft/wavlm-base-plus",
    "classifier": str(MODELS_DIR / "gender_classifier.pkl"),
    "scaler": str(MODELS_DIR / "gender_scaler.pkl"),
    "pca": str(MODELS_DIR / "gender_pca.pkl"),
    "label_encoder": str(MODELS_DIR / "gender_label_encoder.pkl"),
    "classes": ["Male", "Female"],  # 0=Male, 1=Female (CRITICAL: DO NOT CHANGE)
    "description": "WavLM-base-plus + Logistic Regression"
}

# Intent Model Configuration (by Sahasra)
INTENT_MODEL = {
    "feature_extractor": "microsoft/wavlm-base-plus",
    "classifier": str(MODELS_DIR / "intent_classifier.pkl"),
    "scaler": str(MODELS_DIR / "intent_scaler.pkl"),
    "pca": str(MODELS_DIR / "intent_pca.pkl"),
    "label_encoder": str(MODELS_DIR / "intent_label_encoder.pkl"),
    "description": "WavLM-base-plus + Intent Classification (SLURP dataset)"
}

# Speaker Model Configuration
SPEAKER_MODEL = {
    "feature_extractor": "facebook/wav2vec2-large-xlsr-53",
    "classifier": str(MODELS_DIR / "xlsr_classifier.pkl"),
    "pca": str(MODELS_DIR / "xlsr_pca.pkl"),
    "scaler": str(MODELS_DIR / "xlsr_scaler.pkl"),
    "label_encoder": str(MODELS_DIR / "xlsr_label_encoder.pkl"),
    "description": "XLSR-53 + Speaker Identification"
}

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

# Standard audio settings for all models
AUDIO_SETTINGS = {
    "sample_rate": 16000,           # 16kHz for all models
    "n_channels": 1,                # Mono
    "bit_depth": 16,                # 16-bit
}

# Supported audio formats
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "webm"}

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

FLASK_CONFIG = {
    "SECRET_KEY": os.environ.get("FLASK_SECRET", "dev-secret-key-change-in-production"),
    "DEBUG": os.environ.get("FLASK_DEBUG", True),
    "TEMPLATES_AUTO_RELOAD": True,
    "MAX_CONTENT_LENGTH": 50 * 1024 * 1024,  # 50MB max file upload
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# PyTorch device (CPU or GPU)
DEVICE = "cpu"  # Use "cuda" if GPU is available and desired

# ============================================================================
# ENSURE REQUIRED DIRECTORIES EXIST
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        MODELS_DIR,
        DATA_DIR,
        RESULTS_DIR,
        EMBEDDINGS_DIR,
        UPLOADS_DIR,
        TEMPLATES_DIR,
        STATIC_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()
