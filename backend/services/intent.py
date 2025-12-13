"""
Intent Classification Service
Classifies user intent from audio using WavLM embeddings + Classifier.

Model Details:
- Dataset: SLURP (Spoken Language Understanding Researchers' Platform)
- Feature Extractor: WavLM-base-plus
- Classifier: SVM
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

# Confidence threshold to flag unknown/low-confidence intents
INTENT_UNKNOWN_THRESHOLD = 0.35
TOP_K_PREDICTIONS = 3


class IntentInferenceService:
    """
    Intent classification inference service.
    
    Classifies user intent from audio using WavLM-base-plus embeddings
    with SVM classifier and PCA dimensionality reduction.
    
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
            
            # Predict intent class (raw value from classifier)
            prediction_raw = self._classifier.predict(embedding_final)[0]

            # Resolve labels and decode prediction deterministically
            classifier_classes = getattr(self._classifier, "classes_", None)
            labels = None
            prediction_label = None

            # Preferred path: encoder with classes_ (scikit LabelEncoder)
            if self._encoder is not None and hasattr(self._encoder, "classes_"):
                labels = list(self._encoder.classes_)
                try:
                    prediction_label = self._encoder.inverse_transform([prediction_raw])[0]
                except Exception as e:
                    print(f"Error decoding with encoder.inverse_transform: {e}")
                    prediction_label = str(prediction_raw)

            # Fallback: classifier classes are strings (trained on string labels directly)
            elif classifier_classes is not None and all(isinstance(c, str) for c in classifier_classes):
                labels = [str(c) for c in classifier_classes]
                prediction_label = str(prediction_raw)

            # Fallback: encoder stored as dict (handle both index->label and label->index)
            elif isinstance(self._encoder, dict):
                encoder_dict = self._encoder
                # Detect orientation: values are ints means encoder is label->index
                if all(isinstance(v, (int, np.integer)) for v in encoder_dict.values()):
                    index_to_label = {int(v): str(k) for k, v in encoder_dict.items()}
                else:  # assume keys are indices
                    index_to_label = {int(k): str(v) for k, v in encoder_dict.items()}

                if classifier_classes is not None:
                    labels = [index_to_label.get(int(idx), str(idx)) for idx in classifier_classes]
                else:
                    labels = list(index_to_label.values())

                prediction_label = index_to_label.get(int(prediction_raw), str(prediction_raw))

            # Last resort: use classifier classes as strings or raw prediction
            elif classifier_classes is not None:
                labels = [str(c) for c in classifier_classes]
                prediction_label = str(prediction_raw)
            else:
                labels = [str(prediction_raw)]
                prediction_label = str(prediction_raw)

            # Build probability distribution aligned with resolved labels
            if hasattr(self._classifier, "predict_proba"):
                raw_probs = self._classifier.predict_proba(embedding_final)[0]
                if labels is not None and len(labels) == len(raw_probs):
                    probabilities = {str(lbl): float(p) for lbl, p in zip(labels, raw_probs)}
                else:
                    probabilities = {str(lbl): float(1.0 / len(labels)) for lbl in labels}
            else:
                probabilities = get_probabilities(self._classifier, embedding_final, labels)

            # Unknown/low-confidence handling
            sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_conf = sorted_probs[0]
            if top_conf < INTENT_UNKNOWN_THRESHOLD:
                prediction_label = "unknown"

            return {
                "label": prediction_label,
                "probabilities": probabilities,
                "top_predictions": sorted_probs[:TOP_K_PREDICTIONS],
                "available_labels": labels,
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
