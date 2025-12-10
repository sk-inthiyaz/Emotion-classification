import os
import soundfile as sf
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# The 20 speakers identified by the user
TARGET_SPEAKERS = {
    121, 237, 260, 1089, 1188, 1221, 1284, 1320, 1580,
    1995, 2094, 2300, 2830, 2961, 3570, 3575, 3729, 4077, 4446, 4507
}

OUTPUT_DIR = Path("data/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading samples for {len(TARGET_SPEAKERS)} speakers from LibriSpeech (test-clean)...")

from datasets import load_dataset, Audio

# ...

# Load dataset in streaming mode
ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
# Disable automatic decoding to avoid dependency issues
ds = ds.cast_column("audio", Audio(decode=False))

found_speakers = set()

print("Streaming dataset...")
for sample in tqdm(ds):
    speaker_id = sample['speaker_id']
    
    if speaker_id in TARGET_SPEAKERS and speaker_id not in found_speakers:
        # Get raw audio data
        audio_data = sample['audio']
        bytes_content = audio_data['bytes']
        
        # LibriSpeech is usually FLAC
        filename = f"{speaker_id}_sample.flac"
        filepath = OUTPUT_DIR / filename
        
        # Write raw bytes directly
        with open(filepath, "wb") as f:
            f.write(bytes_content)
        
        found_speakers.add(speaker_id)
        print(f"✓ Saved sample for speaker {speaker_id} to {filepath}")
        
    if len(found_speakers) == len(TARGET_SPEAKERS):
        break

print(f"\nDone! Downloaded {len(found_speakers)}/{len(TARGET_SPEAKERS)} samples.")
if len(found_speakers) < len(TARGET_SPEAKERS):
    missing = TARGET_SPEAKERS - found_speakers
    print(f"Missing speakers: {missing}")
