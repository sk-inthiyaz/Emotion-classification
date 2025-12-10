import sys
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from app.app import SpeakerInferenceService
    print("Import successful.")
    
    service = SpeakerInferenceService()
    print("Service initialized.")
    
    # This triggers model loading
    service._load_speaker_artifacts()
    print("Models loaded successfully!")
    
    # Check if attributes are set
    if service._classifier and service._pca and service._encoder:
        print("All artifacts (classifier, pca, encoder) are present.")
    else:
        print("Some artifacts are missing after load.")
        
except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
