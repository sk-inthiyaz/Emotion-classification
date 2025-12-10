"""
Emotion Classification Service
Classifies audio emotions using HuBERT-large embeddings + SVM classifier.

Model Performance:
- Accuracy: 79.14%
- Dataset: CREMA-D
- Architecture: HuBERT-large + SVM

Classes: Neutral, Sad, Happy, Angry
"""

from pathlib import Path
from typing import Dict
import joblib
import torch
import numpy as np
import sys
from pathlib import Path as PathlibPath

# Add parent directory to path for backend imports
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))

from config import EMOTION_MODEL, SRC_DIR, MODELS_DIR, DEVICE
from services.utils.audio import load_feature_extractor, get_probabilities


class EmotionInferenceService:
    """
    Emotion classification inference service.
    
    Uses HuBERT-large embeddings with SVM classifier for high-accuracy
    emotion classification (79.14% on CREMA-D).
    """
    
    def __init__(self):
        """Initialize the emotion inference service."""
        self.device = torch.device(DEVICE)
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._encoder = None
    
    @property
    def extractor(self):
        """
        Lazy-load feature extractor (HuBERT-large).
        Reduces startup time by only loading when first needed.
        """
        if self._extractor is None:
            model_name = EMOTION_MODEL["feature_extractor"]
            self._extractor = load_feature_extractor(SRC_DIR, model_name)
        return self._extractor
    
    def _load_artifacts(self):
        """Load pre-trained model artifacts from disk."""
        if self._classifier is not None:
            return  # Already loaded
        
        model_path = MODELS_DIR / EMOTION_MODEL["classifier"].split("/")[-1]
        scaler_path = MODELS_DIR / EMOTION_MODEL["scaler"].split("/")[-1]
        encoder_path = MODELS_DIR / EMOTION_MODEL["label_encoder"].split("/")[-1]
        
        # Verify all required files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Missing emotion model: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing emotion scaler: {scaler_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing emotion label encoder: {encoder_path}")
        
        # Load artifacts
        self._classifier = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._encoder = joblib.load(encoder_path)
    
    def predict_emotion(self, audio_path: Path) -> Dict[str, object]:
        """
        Classify emotion from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, ogg, m4a, webm)
            
        Returns:
            Dictionary with:
            - 'label': Predicted emotion class
            - 'probabilities': Dict of {emotion: confidence}
        """
        self._load_artifacts()
        
        # Extract embedding from audio
        embedding = self.extractor.extract_from_file(str(audio_path))
        embedding = embedding.reshape(1, -1)
        
        # Scale the embedding
        embedding_scaled = self._scaler.transform(embedding)
        
        # Predict using SVM
        prediction_idx = int(self._classifier.predict(embedding_scaled)[0])
        prediction_label = str(self._encoder.inverse_transform([prediction_idx])[0])
        
        # Get probability estimates
        labels = list(self._encoder.classes_)
        probabilities = get_probabilities(self._classifier, embedding_scaled, labels)
        
        return {
            "label": prediction_label,
            "probabilities": probabilities
        }
