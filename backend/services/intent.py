"""
Intent Classification Service
Classifies user intent from audio using WavLM embeddings + Classifier.

Model Details:
- Dataset: SLURP (Spoken Language Understanding Researchers' Platform)
- Feature Extractor: WavLM-base-plus
- Classifier: Logistic Regression
- Dimensionality Reduction: PCA

Developer: Sahasra
This module handles intent classification for understanding user commands and requests.
"""

import os
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import sys

# Add parent directory to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INTENT_MODEL, SRC_DIR, MODELS_DIR, DEVICE
from services.utils.audio import (
    load_feature_extractor,
    convert_to_wav,
    get_probabilities,
    l2_normalize
)


class IntentInferenceService:
    """
    Intent classification inference service.
    
    Classifies user intent from audio using WavLM-base-plus embeddings
    with Logistic Regression classifier and PCA dimensionality reduction.
    
    This service is designed by Sahasra for understanding user commands
    and requests from spoken input.
    """
    
    def __init__(self):
        """Initialize the intent inference service."""
        self.device = DEVICE
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._pca = None
        self._encoder = None
        
        # Set artifact paths
        self._model_path = MODELS_DIR / INTENT_MODEL["classifier"].split("/")[-1]
        self._scaler_path = MODELS_DIR / INTENT_MODEL["scaler"].split("/")[-1]
        self._pca_path = MODELS_DIR / INTENT_MODEL["pca"].split("/")[-1]
        self._encoder_path = MODELS_DIR / INTENT_MODEL["label_encoder"].split("/")[-1]
    
    def _load_artifacts(self):
        """Load pre-trained model artifacts from disk."""
        if self._classifier is not None:
            return  # Already loaded
        
        # Verify required files exist
        if not self._model_path.exists():
            raise FileNotFoundError(f"Intent model not found: {self._model_path}")
        if not self._scaler_path.exists():
            raise FileNotFoundError(f"Intent scaler not found: {self._scaler_path}")
        if not self._encoder_path.exists():
            raise FileNotFoundError(f"Intent encoder not found: {self._encoder_path}")
        
        # Load feature extractor
        if self._extractor is None:
            model_name = INTENT_MODEL["feature_extractor"]
            self._extractor = load_feature_extractor(SRC_DIR, model_name)
        
        # Load model artifacts
        self._classifier = joblib.load(self._model_path)
        self._scaler = joblib.load(self._scaler_path)
        self._encoder = joblib.load(self._encoder_path)
        
        # Load PCA if available (optional for intent classification)
        if self._pca_path.exists():
            self._pca = joblib.load(self._pca_path)
        else:
            print(f"Warning: Intent PCA not found at {self._pca_path} - skipping PCA transformation")
            self._pca = None
    
    def predict_intent(self, audio_path: Path) -> Dict[str, object]:
        """
        Classify user intent from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, flac, ogg, m4a, webm)
            
        Returns:
            Dictionary with:
            - 'label': Predicted intent class
            - 'probabilities': Dict of {intent: confidence}
            
        Developer Note (Sahasra):
        This method processes audio through the following pipeline:
        1. Audio normalization to 16kHz WAV
        2. Feature extraction using WavLM-base-plus
        3. L2 normalization of embeddings
        4. Feature scaling and PCA transformation
        5. Intent classification using trained model
        """
        self._load_artifacts()
        
        # Normalize audio to 16kHz WAV
        temp_wav = audio_path.with_suffix(".converted.wav")
        try:
            clean_path = convert_to_wav(str(audio_path), str(temp_wav))
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            clean_path = str(audio_path)
        
        try:
            # Extract embeddings from audio
            embedding = self._extractor.extract_from_file(clean_path, pooling="mean")
            
            # L2 normalize embeddings for consistency
            embedding = l2_normalize(embedding)
            embedding = embedding.reshape(1, -1)
            
            # Apply feature scaling
            embedding_scaled = self._scaler.transform(embedding)
            
            # Apply PCA for dimensionality reduction (if available)
            if self._pca is not None:
                embedding_final = self._pca.transform(embedding_scaled)
            else:
                embedding_final = embedding_scaled
            
            # Predict intent
            prediction_raw = self._classifier.predict(embedding_final)[0]
            
            # Decode the prediction using label encoder
            if self._encoder is not None:
                if isinstance(self._encoder, dict):
                    # If encoder is stored as dict {0: "intent_name", ...}
                    prediction_label = self._encoder.get(int(prediction_raw), str(prediction_raw))
                elif hasattr(self._encoder, 'inverse_transform'):
                    # If encoder is a proper LabelEncoder object
                    try:
                        prediction_label = self._encoder.inverse_transform([prediction_raw])[0]
                    except Exception as e:
                        print(f"Error decoding prediction: {e}")
                        prediction_label = str(prediction_raw)
                else:
                    prediction_label = str(prediction_raw)
            else:
                prediction_label = str(prediction_raw)
            
            # Get probability estimates and decode class labels
            if hasattr(self._classifier, "classes_"):
                class_indices = self._classifier.classes_
                # Decode all class labels for probabilities
                if self._encoder is not None:
                    if isinstance(self._encoder, dict):
                        # Dict encoder: map indices to labels
                        labels = [self._encoder.get(int(idx), str(idx)) for idx in class_indices]
                    elif hasattr(self._encoder, 'inverse_transform'):
                        # LabelEncoder object
                        try:
                            labels = self._encoder.inverse_transform(class_indices)
                        except Exception:
                            labels = [str(idx) for idx in class_indices]
                    else:
                        labels = [str(idx) for idx in class_indices]
                else:
                    labels = [str(idx) for idx in class_indices]
            else:
                labels = [prediction_label]
            
            # Get probabilities with decoded labels
            probabilities = get_probabilities(self._classifier, embedding_final, labels)
            
            return {
                "label": prediction_label,
                "probabilities": probabilities
            }
        
        except Exception as e:
            print(f"Intent prediction error: {e}")
            raise
        
        finally:
            # Clean up temporary file
            if str(temp_wav) != str(audio_path) and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
