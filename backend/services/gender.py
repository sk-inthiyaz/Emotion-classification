"""
Gender Classification Service
Classifies speaker gender from audio using WavLM embeddings + Logistic Regression.

Model Details:
- Dataset: LibriSpeech dev-clean
- Feature Extractor: WavLM-base-plus
- Classifier: Logistic Regression
- Dimensionality Reduction: PCA

Classes: Male (0), Female (1)
CRITICAL: Do NOT change the class mapping - it must remain 0=Male, 1=Female
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

from config import GENDER_MODEL, SRC_DIR, MODELS_DIR, DEVICE
from services.utils.audio import (
    load_feature_extractor,
    normalize_audio_to_wav,
    get_probabilities
)


class GenderInferenceService:
    """
    Gender classification inference service.
    
    Identifies speaker gender using WavLM-base-plus embeddings with
    Logistic Regression classifier after PCA dimensionality reduction.
    """
    
    def __init__(self):
        """Initialize the gender inference service."""
        self.device = torch.device(DEVICE)
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._pca = None
        self._encoder = None
        
        # Set artifact paths
        self._model_path = MODELS_DIR / GENDER_MODEL["classifier"].split("/")[-1]
        self._scaler_path = MODELS_DIR / GENDER_MODEL["scaler"].split("/")[-1]
        self._pca_path = MODELS_DIR / GENDER_MODEL["pca"].split("/")[-1]
        self._encoder_path = MODELS_DIR / GENDER_MODEL["label_encoder"].split("/")[-1]
    
    def _load_artifacts(self):
        """Load pre-trained model artifacts from disk."""
        if self._classifier is not None:
            return  # Already loaded
        
        # Verify all required files exist
        if not self._model_path.exists():
            raise FileNotFoundError(f"Missing gender model: {self._model_path}")
        
        # Load feature extractor
        if self._extractor is None:
            model_name = GENDER_MODEL["feature_extractor"]
            self._extractor = load_feature_extractor(SRC_DIR, model_name)
        
        # Load model artifacts
        self._classifier = joblib.load(self._model_path)
        
        if self._scaler_path.exists():
            self._scaler = joblib.load(self._scaler_path)
        
        if self._pca_path.exists():
            self._pca = joblib.load(self._pca_path)
        
        if self._encoder_path.exists():
            self._encoder = joblib.load(self._encoder_path)
    
    def predict_gender(self, audio_path: Path, is_recording: bool = False) -> Dict[str, object]:
        """
        Classify speaker gender from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, ogg, m4a, webm)
            is_recording: Whether input is from microphone recording (for logging)
            
        Returns:
            Dictionary with:
            - 'label': Predicted gender ('Male' or 'Female')
            - 'probabilities': Dict of {gender: confidence}
            
        Note:
            CRITICAL: Class mapping is 0=Male, 1=Female. This must NOT be changed
            to ensure consistency with the trained model.
        """
        self._load_artifacts()
        
        # Normalize audio to 16kHz mono WAV
        temp_wav = audio_path.with_suffix(".converted.wav")
        try:
            clean_path = normalize_audio_to_wav(audio_path, temp_wav)
        except Exception as e:
            print(f"Audio normalization failed: {e}")
            raise
        
        try:
            # Extract embeddings
            embedding = self._extractor.extract_from_file(str(clean_path), pooling="mean")
            embedding = embedding.reshape(1, -1)
            
            # Apply scaler
            if self._scaler:
                embedding = self._scaler.transform(embedding)
            
            # Apply PCA for dimensionality reduction
            if self._pca:
                embedding = self._pca.transform(embedding)
            
            # Predict gender
            prediction_idx = self._classifier.predict(embedding)[0]
            
            # Map prediction to gender class
            # CRITICAL: 0=Male, 1=Female (DO NOT CHANGE)
            GENDER_CLASSES = GENDER_MODEL["classes"]  # ["Male", "Female"]
            prediction_label = GENDER_CLASSES[prediction_idx]
            
            # Get probability estimates
            probabilities = get_probabilities(self._classifier, embedding, GENDER_CLASSES)
            
            return {
                "label": prediction_label,
                "probabilities": probabilities
            }
        
        except Exception as e:
            print(f"Gender prediction error: {e}")
            raise
        
        finally:
            # Clean up temporary file
            if str(temp_wav) != str(audio_path) and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
