"""
Speaker Identification Service
Identifies speakers from audio using XLSR embeddings + Classifier.

Model Details:
- Feature Extractor: XLSR-53 (Wav2Vec2 Large)
- Classifier: Support Vector Machine / Classifier
- Dimensionality Reduction: PCA + Scaler

This service identifies unique speakers from audio input.
"""

import os
from pathlib import Path
from typing import Dict
import joblib
import torch
import numpy as np
import sys

# Add parent directory to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPEAKER_MODEL, SRC_DIR, MODELS_DIR, DEVICE
from services.utils.audio import load_feature_extractor, get_probabilities


class SpeakerInferenceService:
    """
    Speaker identification inference service.
    
    Identifies speakers from audio using XLSR-53 embeddings with
    classifier for speaker recognition.
    """
    
    def __init__(self):
        """Initialize the speaker inference service."""
        self.device = torch.device(DEVICE)
        self._extractor = None
        self._classifier = None
        self._pca = None
        self._scaler = None
        self._encoder = None
    
    def _load_artifacts(self):
        """Load pre-trained model artifacts from disk."""
        if self._classifier is not None:
            return  # Already loaded
        
        # Set artifact paths
        model_path = MODELS_DIR / SPEAKER_MODEL["classifier"].split("/")[-1]
        pca_path = MODELS_DIR / SPEAKER_MODEL["pca"].split("/")[-1]
        encoder_path = MODELS_DIR / SPEAKER_MODEL["label_encoder"].split("/")[-1]
        scaler_path = MODELS_DIR / SPEAKER_MODEL["scaler"].split("/")[-1]
        
        # Verify required files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Speaker model not found: {model_path}")
        if not pca_path.exists():
            raise FileNotFoundError(f"Speaker PCA not found: {pca_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Speaker encoder not found: {encoder_path}")
        
        # Load feature extractor
        if self._extractor is None:
            model_name = SPEAKER_MODEL["feature_extractor"]
            self._extractor = load_feature_extractor(SRC_DIR, model_name)
        
        # Load model artifacts
        self._classifier = joblib.load(model_path)
        self._pca = joblib.load(pca_path)
        self._encoder = joblib.load(encoder_path)
        
        # Load and fix scaler if it exists
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            # Fix potential division-by-zero issues in scaler
            # Some components may have very small scale values
            if hasattr(self._scaler, "scale_"):
                self._scaler.scale_[self._scaler.scale_ < 1e-6] = 1.0
        else:
            self._scaler = None
    
    def predict_speaker(self, audio_path: Path) -> Dict[str, object]:
        """
        Identify speaker from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, ogg, m4a, webm)
            
        Returns:
            Dictionary with:
            - 'label': Predicted speaker ID
            - 'probabilities': Dict of {speaker: confidence}
        """
        self._load_artifacts()
        
        # Extract embeddings using XLSR-53 with custom pooling strategy
        # Uses layers 6-13 with mean pooling across layers + mean/std pooling
        embedding = self._extractor.extract_from_file(str(audio_path), pooling="xlsr_custom")
        embedding = embedding.reshape(1, -1)
        
        # Apply PCA for dimensionality reduction
        embedding_pca = self._pca.transform(embedding)
        
        # Apply scaler if available
        if self._scaler:
            embedding_pca = self._scaler.transform(embedding_pca)
        
        # Predict speaker
        prediction_raw = int(self._classifier.predict(embedding_pca)[0])
        
        # Map prediction to speaker label
        if prediction_raw < len(self._encoder.classes_):
            # Prediction is an index
            prediction_label = str(self._encoder.inverse_transform([prediction_raw])[0])
        else:
            # Prediction is already a label
            prediction_label = str(prediction_raw)
        
        # Get probability estimates
        labels = list(self._encoder.classes_)
        probabilities = get_probabilities(self._classifier, embedding_pca, labels)
        
        return {
            "label": prediction_label,
            "probabilities": probabilities
        }
