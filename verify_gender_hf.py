import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("src").resolve()))

try:
    from app.app import GenderInferenceService
    
    print("Initializing GenderInferenceService...")
    service = GenderInferenceService()
    
    audio_path = Path("1221_debug.flac")
    if not audio_path.exists():
        print(f"Test file {audio_path} not found.")
        # Try to find any audio file
        audio_files = list(Path(".").glob("*.flac")) + list(Path(".").glob("*.wav"))
        if audio_files:
            audio_path = audio_files[0]
            print(f"Using {audio_path} instead.")
        else:
            print("No audio files found for testing.")
            sys.exit(1)

    print(f"Predicting gender for {audio_path}...")
    result = service.predict_gender(audio_path)
    print("Prediction Result:")
    print(result)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
