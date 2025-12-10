import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("src").resolve()))

try:
    from app.app import GenderInferenceService
    service = GenderInferenceService()
    service._load_gender_artifacts()
    print("Gender artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except Exception as e:
    print(f"Error: {e}")
