"""
Utility Functions Module
Common utilities for audio processing and model inference.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg

# ============================================================================
# AUDIO CONVERSION & PREPROCESSING
# ============================================================================

def setup_ffmpeg():
    """
    Set FFmpeg path for pydub.
    Required for audio format conversion.
    """
    try:
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"WARNING: FFmpeg setup failed: {e}")


def convert_to_wav(input_path: str, target_path: str) -> str:
    """
    Convert any audio file to 16kHz Mono WAV format.
    
    Args:
        input_path: Path to input audio file
        target_path: Path to save converted WAV file
        
    Returns:
        Path to the converted WAV file (or original if conversion fails)
    """
    try:
        audio = AudioSegment.from_file(input_path)
        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(target_path, format="wav")
        return target_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_path  # Fallback to original


def normalize_audio_to_wav(audio_path: Path, target_path: Path) -> Path:
    """
    Normalize audio to 16kHz mono WAV using torchaudio.
    Ensures consistent preprocessing across all models.
    
    Args:
        audio_path: Input audio file path
        target_path: Output WAV file path
        
    Returns:
        Path to normalized WAV file
    """
    try:
        import torchaudio
        import torchaudio.transforms as T
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Force mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Force 16000 Hz
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        # Save as 16k mono WAV
        torchaudio.save(str(target_path), waveform, 16000)
        return target_path
        
    except Exception as e:
        print(f"Torchaudio conversion failed: {e}. Falling back to pydub.")
        return Path(convert_to_wav(str(audio_path), str(target_path)))


# ============================================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================================

def load_feature_extractor(src_dir: Path, model_name: str):
    """
    Dynamically load WavLMFeatureExtractor from src directory.
    Works around filename constraints (leading digits in filenames).
    
    Args:
        src_dir: Path to src directory
        model_name: Name of the HuggingFace model
        
    Returns:
        Instantiated WavLMFeatureExtractor
    """
    from importlib.machinery import SourceFileLoader
    import types
    
    module_path = src_dir / "2_wavlm_feature_extraction.py"
    loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
    
    return WavLMFeatureExtractor(model_name=model_name)


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize an embedding vector.
    
    Args:
        embedding: Input embedding array
        
    Returns:
        L2 normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ============================================================================
# PROBABILITY UTILITIES
# ============================================================================

def get_probabilities(classifier, embeddings: np.ndarray, classes: list):
    """
    Extract probability estimates from classifier.
    Handles different classifier types (predict_proba, decision_function, etc).
    
    Args:
        classifier: Trained classifier model
        embeddings: Input embeddings
        classes: List of class labels
        
    Returns:
        Dictionary mapping class labels to probabilities
    """
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(embeddings)[0]
    elif hasattr(classifier, "decision_function"):
        # Convert decision function to probabilities using softmax
        decision = classifier.decision_function(embeddings)[0]
        exp_decision = np.exp(decision - np.max(decision))
        probs = exp_decision / exp_decision.sum()
    else:
        # Fallback: uniform probabilities
        probs = np.ones(len(classes)) / len(classes)
    
    return {str(lbl): float(p) for lbl, p in zip(classes, probs)}


# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

def ensure_dependencies():
    """Check and install required dependencies."""
    try:
        import soundfile
    except ImportError:
        print("WARNING: 'soundfile' library not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
        print("Dependency installed successfully.")


# Initialize on module import
setup_ffmpeg()
