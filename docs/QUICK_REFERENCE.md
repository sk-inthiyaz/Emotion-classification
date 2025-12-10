"""
QUICK REFERENCE GUIDE - Emotion Classification System v2.0
===========================================================

Use this guide for quick lookups on how to use the new modular architecture.


BASIC IMPORTS
=============

Services:
    from core.services import (
        EmotionInferenceService,
        GenderInferenceService,
        IntentInferenceService,
        SpeakerInferenceService
    )

Configuration:
    from config import MODELS_DIR, SRC_DIR, ALLOWED_EXTENSIONS

Utilities:
    from core.utils.audio import (
        normalize_audio_to_wav,
        get_probabilities,
        load_feature_extractor
    )


RUNNING THE FLASK APP
=====================

Development:
    python app/app.py

With custom port:
    PORT=8000 python app/app.py

With SSL:
    Place cert.pem and key.pem in project root, then:
    python app/app.py

Debug mode:
    FLASK_DEBUG=True python app/app.py


SERVICE USAGE EXAMPLES
======================

1. EMOTION CLASSIFICATION

    from core.services import EmotionInferenceService
    from pathlib import Path
    
    service = EmotionInferenceService()
    result = service.predict_emotion(Path("audio.wav"))
    
    print(result["label"])                # e.g., "Happy"
    print(result["probabilities"])        # {"Angry": 0.1, "Happy": 0.8, ...}

2. GENDER CLASSIFICATION

    from core.services import GenderInferenceService
    from pathlib import Path
    
    service = GenderInferenceService()
    result = service.predict_gender(Path("audio.wav"))
    
    print(result["label"])                # "Male" or "Female"
    print(result["probabilities"])        # {"Male": 0.95, "Female": 0.05}

3. INTENT CLASSIFICATION (by Sahasra)

    from core.services import IntentInferenceService
    from pathlib import Path
    
    service = IntentInferenceService()
    result = service.predict_intent(Path("audio.wav"))
    
    print(result["label"])                # Intent class
    print(result["probabilities"])        # Intent confidence scores

4. SPEAKER IDENTIFICATION

    from core.services import SpeakerInferenceService
    from pathlib import Path
    
    service = SpeakerInferenceService()
    result = service.predict_speaker(Path("audio.wav"))
    
    print(result["label"])                # Speaker ID
    print(result["probabilities"])        # Speaker confidence scores


AUDIO FORMAT SUPPORT
====================

Supported Formats:
    ✓ WAV
    ✓ MP3
    ✓ FLAC
    ✓ OGG
    ✓ M4A
    ✓ WebM

Automatic Processing:
    All formats are automatically converted to 16kHz mono WAV
    See: core/utils/audio.py


CONFIGURATION SETTINGS
======================

Edit config.py to change:

Paths:
    PROJECT_ROOT = Path(__file__).resolve().parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"

Model Settings:
    EMOTION_MODEL["feature_extractor"] = "facebook/hubert-large-ll60k"
    EMOTION_MODEL["accuracy"] = 0.7914

Audio Settings:
    AUDIO_SETTINGS["sample_rate"] = 16000      # 16kHz
    AUDIO_SETTINGS["n_channels"] = 1           # Mono

Flask:
    FLASK_CONFIG["DEBUG"] = False
    FLASK_CONFIG["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB


FILE PATHS
==========

Project Root:
    /emotion-classification/
    ├── config.py                  (Configuration)
    ├── PROJECT_OVERVIEW.md        (This project info)
    ├── STRUCTURE_GUIDE.md         (Architecture)
    └── MIGRATION_GUIDE.md         (Upgrade guide)

Services:
    /core/services/
    ├── emotion.py                 (Emotion service)
    ├── gender.py                  (Gender service)
    ├── intent.py                  (Intent service - Sahasra)
    └── speaker.py                 (Speaker service)

Utilities:
    /core/utils/
    └── audio.py                   (Audio utilities)

Flask App:
    /app/
    ├── app_new.py                 (NEW refactored app)
    ├── app.py                     (OLD - keep as backup)
    ├── templates/                 (HTML templates)
    └── static/                    (CSS, JS)

Models:
    /models/
    ├── emotion_model_svm.pkl
    ├── gender_classifier.pkl
    ├── intent_classifier.pkl      (Sahasra's model)
    └── xlsr_classifier.pkl


ROUTES (Flask App)
==================

GET Routes:
    GET /                          Home page
    GET /emotion                   Emotion classifier
    GET /gender                    Gender classifier
    GET /intent                    Intent classifier (Sahasra)
    GET /speaker                   Speaker identifier
    GET /about                     About page

POST Routes:
    POST /emotion/predict          Classify emotion
    POST /gender/predict           Classify gender
    POST /intent/predict           Classify intent (Sahasra)
    POST /speaker/predict          Identify speaker


COMMON ISSUES & SOLUTIONS
=========================

Issue: "ModuleNotFoundError: No module named 'core'"
Solution: Ensure config.py and core/ are in project root

Issue: "FileNotFoundError: Missing model"
Solution: Check models/ directory has all required .pkl files
         Verify paths in config.py

Issue: "Gender still predicting wrong"
Solution: Using NEW code? (gender.py from core/services/)
         Check GENDER_MODEL["classes"] = ["Male", "Female"]

Issue: "Feature extractor not loading"
Solution: src/2_wavlm_feature_extraction.py must exist
         Check SRC_DIR path in config.py

Issue: "Audio processing error"
Solution: Install dependencies: pip install -r requirements.txt
         Verify FFmpeg is installed

Issue: "Port 5000 already in use"
Solution: Use different port: PORT=8000 python app/app.py


ENVIRONMENT VARIABLES
=====================

Available Variables:

Flask Configuration:
    export FLASK_SECRET="your-secret-key"
    export FLASK_DEBUG=True
    export PORT=5000

System:
    export PYTHONPATH=/path/to/project:$PYTHONPATH

Example (Windows PowerShell):
    $env:FLASK_DEBUG = "True"
    $env:PORT = "8000"
    python app/app.py


TESTING INDIVIDUAL SERVICES
===========================

Test Emotion Service:
    python -c "
    from core.services import EmotionInferenceService
    from pathlib import Path
    s = EmotionInferenceService()
    r = s.predict_emotion(Path('test.wav'))
    print(f'Emotion: {r[\"label\"]}')
    "

Test Gender Service:
    python -c "
    from core.services import GenderInferenceService
    from pathlib import Path
    s = GenderInferenceService()
    r = s.predict_gender(Path('test.wav'))
    print(f'Gender: {r[\"label\"]}')
    "

Test All Services Load:
    python -c "
    from core.services import (
        EmotionInferenceService,
        GenderInferenceService,
        IntentInferenceService,
        SpeakerInferenceService
    )
    print('✓ All services loaded successfully!')
    "


PERFORMANCE NOTES
=================

First Prediction: Slow (models load for first time)
Subsequent: Fast (models cached in memory)

Typical Performance:
    Emotion: 100-200ms per prediction
    Gender: 50-100ms per prediction
    Intent: 50-100ms per prediction (Sahasra)
    Speaker: 100-200ms per prediction

To Optimize:
    - Run on GPU (change config.py: DEVICE = "cuda")
    - Pre-load services on app startup (already done)
    - Use batch processing for multiple audios


ADDING NEW CLASSIFICATION TASK
===============================

1. Create Service (core/services/new_task.py):
    class NewTaskInferenceService:
        def __init__(self):
            self._classifier = None
        
        def predict_newtask(self, audio_path: Path) -> Dict[str, object]:
            # Load model, extract features, predict
            return {"label": prediction, "probabilities": probs}

2. Add to config.py:
    NEWTASK_MODEL = {
        "feature_extractor": "...",
        "classifier": "models/...",
        ...
    }

3. Add to core/services/__init__.py:
    from core.services.new_task import NewTaskInferenceService

4. Add Flask routes in app/app.py:
    @app.route("/newtask", methods=["GET"])
    def newtask_page():
        ...
    
    @app.route("/newtask/predict", methods=["POST"])
    def newtask_predict():
        ...

5. Create template (app/templates/newtask.html)

That's it! Done in ~1 hour.


DEPENDENCIES
============

Core:
    torch, torchvision, torchaudio
    flask
    scikit-learn
    joblib
    numpy
    scipy

Audio:
    librosa
    pydub
    soundfile
    imageio_ffmpeg

See requirements.txt for full list

Install all:
    pip install -r requirements.txt


VERSION HISTORY
===============

v1.0 - Original monolithic app.py
v2.0 - Refactored modular architecture (current)

Changes in v2.0:
    ✓ Modular services (core/services/)
    ✓ Centralized config (config.py)
    ✓ Utility module (core/utils/)
    ✓ Gender classification fixed
    ✓ Professional documentation
    ✓ Type hints throughout
    ✓ Improved error handling


DOCUMENTATION FILES
===================

README.md ........................ Main project documentation
PROJECT_OVERVIEW.md ............ Project overview (this info)
STRUCTURE_GUIDE.md ............ Architecture documentation
MIGRATION_GUIDE.md ............ Upgrade instructions from v1.0
QUICKSTART.md ................. Getting started guide
SETUP_GUIDE.md ................ Installation guide
MODEL_ARCHITECTURE.md ........ Model details


KEY CONTACTS
============

Restructuring & Architecture:
    - Configuration management
    - Service organization
    - Gender fix

Intent Model Developer:
    - Sahasra
    - Intent classification
    - SLURP dataset


USEFUL COMMANDS
===============

Start app:
    python app/app.py

Test imports:
    python -c "from core.services import *; print('✓ OK')"

Check config paths:
    python -c "from config import *; print(f'Root: {PROJECT_ROOT}')"

List models:
    ls -la models/

Run tests:
    pytest tests/          # (if test suite exists)

Check requirements:
    pip list | grep -E "torch|flask|sklearn"


FINAL NOTES
===========

The system is production-ready with:
    ✓ 4 independent classification tasks
    ✓ Clean modular architecture
    ✓ Comprehensive documentation
    ✓ Professional code organization
    ✓ Easy to maintain and extend

For detailed information, see PROJECT_OVERVIEW.md
For architecture details, see STRUCTURE_GUIDE.md
For migration help, see MIGRATION_GUIDE.md

Questions? Check the relevant documentation file!
"""
