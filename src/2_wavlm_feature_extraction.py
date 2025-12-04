"""
Feature Extraction (WavLM / HuBERT)
===================================

Extract fixed-dimensional speech embeddings using self-supervised models.
Default is WavLM-base for IEMOCAP (small subset on CPU). For CREMA-D or
high-capacity runs, set `model_name` to a larger model (e.g., HuBERT-large).
Optimized for CPU-only execution with small batch processing.

Based on the IEEE/ACM 2024 paper:
'From Raw Speech to Fixed Representations: A Comprehensive Evaluation 
of Speech Embedding Techniques'

Author: AI/ML Team
Date: 2024
"""

import os
import logging
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WavLMModel, Wav2Vec2FeatureExtractor, HubertModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WavLMFeatureExtractor:
    """
    Extract speech embeddings (CPU-optimized).

    - Default: WavLM-base (good for quick IEMOCAP validation)
    - Optional: Set `model_name` to large SSL models (e.g., `facebook/hubert-large-ll60k`)
        for CREMA-D experiments.

    Uses final hidden layer with configurable pooling (mean/max/first/last).
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        device: Optional[str] = None,
        embeddings_dir: str = "../embeddings"
    ):
        """
        Initialize the WavLM feature extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cpu only for this setup)
            embeddings_dir: Directory to save extracted embeddings
        """
        self.model_name = model_name
        # Force CPU usage for compatibility
        self.device = "cpu"
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading WavLM model: {model_name}")
        logger.info(f"Using device: {self.device} (CPU-only mode)")
        
        # Detect model type and load appropriate processor/model
        if "hubert" in model_name.lower():
            # HuBERT models use Wav2Vec2FeatureExtractor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name)
        elif "xlsr" in model_name.lower() or "wav2vec2" in model_name.lower():
            # XLSR/Wav2Vec2 models
            from transformers import Wav2Vec2Model
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
        else:
            # WavLM models use Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = WavLMModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model will run on CPU with optimized batch processing")
    
    def load_audio(self, filepath: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load audio file and resample to target sampling rate.
        
        Args:
            filepath: Path to audio file
            target_sr: Target sampling rate (WavLM expects 16kHz)
            
        Returns:
            Audio tensor
        """
        try:
            file_ext = str(filepath).lower().split('.')[-1]
            
            # For WebM or unsupported formats, convert using pydub first
            if file_ext in ['webm', 'ogg', 'opus']:
                    try:
                        import tempfile
                        import subprocess
                        import imageio_ffmpeg
                        
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        
                        # Create temp wav file
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                            tmp_path = tmp_wav.name
                            
                        # Convert using ffmpeg directly: ffmpeg -i input -ar 16000 -ac 1 -y output.wav
                        cmd = [
                            ffmpeg_exe,
                            '-y',
                            '-i', str(filepath),
                            '-ar', str(target_sr),
                            '-ac', '1',
                            tmp_path
                        ]
                        
                        # Run conversion (suppress output)
                        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # Load the WAV file
                        import soundfile as sf
                        waveform_np, sample_rate = sf.read(tmp_path, dtype='float32')
                        
                        # Clean up temp file
                        import os
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        
                        # Convert to tensor
                        waveform = torch.from_numpy(waveform_np)
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0)
                            
                    except Exception as e:
                         # Fallback or error reporting
                         msg = (
                             f"Error converting {file_ext} using ffmpeg: {e}. "
                             "Ensure imageio-ffmpeg is installed."
                         )
                         logger.error(msg)
                         raise RuntimeError(msg)
            else:
                # Try torchaudio first (works for WAV, FLAC)
                try:
                    waveform, sample_rate = torchaudio.load(filepath, backend="soundfile")
                except:
                    # Fallback: use soundfile directly for MP3 and other formats
                    import soundfile as sf
                    waveform_np, sample_rate = sf.read(filepath, dtype='float32')
                    # Convert to tensor and add channel dimension if needed
                    waveform = torch.from_numpy(waveform_np)
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim == 2:
                        # Transpose if channels are in wrong dimension
                        if waveform.shape[0] > waveform.shape[1]:
                            waveform = waveform.T
            
            # Resample if necessary
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze()
        
        except Exception as e:
            logger.error(f"Error loading audio {filepath}: {e}")
            raise
    
    def extract_embedding(
        self,
        audio: torch.Tensor,
        layer: int = -1,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract embedding from audio using WavLM.
        
        Args:
            audio: Audio tensor (1D)
            layer: Which transformer layer to use (-1 for last layer)
            pooling: Pooling strategy ('mean', 'max', 'first', 'last')
            
        Returns:
            Fixed-dimensional embedding as numpy array
        """
        try:
            # Process audio
            inputs = self.processor(
                audio.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[layer]  # Shape: (batch, time, hidden_dim)
            
            # Apply pooling
            if pooling == "mean":
                embedding = hidden_states.mean(dim=1)
            elif pooling == "max":
                embedding = hidden_states.max(dim=1)[0]
            elif pooling == "first":
                embedding = hidden_states[:, 0, :]
            elif pooling == "last":
                embedding = hidden_states[:, -1, :]
            elif pooling == "mean_max":
                mean_emb = hidden_states.mean(dim=1)
                max_emb = hidden_states.max(dim=1)[0]
                embedding = torch.cat([mean_emb, max_emb], dim=1)
            elif pooling == "xlsr_custom":
                # Custom logic from user's training script:
                # 1. Get hidden states from layers 6 to 13 (indices 6 to 12 inclusive)
                # Note: outputs.hidden_states includes embedding layer at index 0, so layer 1 is index 1.
                # The user code: layers = out.hidden_states[6:13]
                # This implies 7 layers.
                
                # We need all hidden states first
                all_hidden = outputs.hidden_states
                
                # Stack layers 6 to 13 (exclusive of 13) -> indices 6,7,8,9,10,11,12
                selected_layers = torch.stack(all_hidden[6:13])
                
                # Mean across layers
                layer_mean = selected_layers.mean(dim=0) # Shape: (batch, time, hidden)
                
                # Pooling: Mean and Std across time
                mean_emb = layer_mean.mean(dim=1)
                std_emb = layer_mean.std(dim=1)
                
                embedding = torch.cat([mean_emb, std_emb], dim=1)
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
            
            # Convert to numpy
            embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_from_file(
        self,
        filepath: str,
        layer: int = -1,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract embedding from audio file.
        
        Args:
            filepath: Path to audio file
            layer: Which transformer layer to use
            pooling: Pooling strategy
            
        Returns:
            Embedding as numpy array
        """
        audio = self.load_audio(filepath)
        return self.extract_embedding(audio, layer=layer, pooling=pooling)
    
    def process_dataset(
        self,
        metadata_path: str,
        dataset_name: str,
        layer: int = -1,
        pooling: str = "mean",
        batch_size: int = 1
    ):
        """
        Process entire dataset and save embeddings (CPU-optimized).
        
        Args:
            metadata_path: Path to metadata CSV file
            dataset_name: Name of the dataset
            layer: Which transformer layer to use
            pooling: Pooling strategy
            batch_size: Batch size for processing (1 for CPU to avoid memory issues)
        """
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Batch size: {batch_size} (optimized for CPU)")
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(df)} samples from {metadata_path}")
        
        embeddings = []
        labels = []
        
        # Process each audio file (one at a time for CPU efficiency)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {dataset_name}"):
            try:
                filepath = row['filepath']
                embedding = self.extract_from_file(filepath, layer=layer, pooling=pooling)
                embeddings.append(embedding)
                
                # Store label (emotion for IEMOCAP)
                if 'emotion' in row:
                    labels.append(row['emotion'])
                else:
                    labels.append('unknown')
                
            except Exception as e:
                logger.warning(f"Skipping {filepath}: {e}")
                continue
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        # Save embeddings as NPZ file (compressed)
        output_path = self.embeddings_dir / f"{dataset_name}_embeddings.npz"
        
        np.savez_compressed(output_path, embeddings=embeddings, labels=labels)
        
        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Unique labels: {np.unique(labels)}")
    
    def extract_multi_layer(
        self,
        filepath: str,
        layers: List[int] = [-4, -3, -2, -1],
        pooling: str = "mean"
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from multiple layers.
        
        Args:
            filepath: Path to audio file
            layers: List of layer indices to extract from
            pooling: Pooling strategy
            
        Returns:
            Dictionary mapping layer indices to embeddings
        """
        audio = self.load_audio(filepath)
        
        embeddings = {}
        for layer in layers:
            embeddings[layer] = self.extract_embedding(audio, layer=layer, pooling=pooling)
        
        return embeddings


if __name__ == "__main__":
    # CPU-optimized feature extraction for IEMOCAP
    logger.info("Starting WavLM feature extraction (CPU mode)")
    extractor = WavLMFeatureExtractor()
    
    # Process IEMOCAP dataset only
    metadata_path = '../data/processed/iemocap_metadata.csv'
    
    if Path(metadata_path).exists():
        logger.info(f"Found metadata file: {metadata_path}")
        extractor.process_dataset(
            metadata_path, 
            'emotion',
            batch_size=1  # Process one sample at a time for CPU
        )
        logger.info("✓ Feature extraction complete!")
    else:
        logger.error(f"✗ Metadata file not found: {metadata_path}")
        logger.error("Please run 1_data_preprocessing.py first")
    
    logger.info("All done!")
