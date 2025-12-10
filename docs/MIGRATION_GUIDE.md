"""
MIGRATION GUIDE: From Monolithic to Modular Architecture
=========================================================

This guide explains how to migrate from the old app.py to the new modular structure.

OVERVIEW
========

OLD: Single 659-line app.py with all logic mixed together
NEW: Modular structure with separate services and utilities

Benefits of Migration:
✓ 50% reduction in complexity per file
✓ Easier maintenance and debugging
✓ Better code reusability
✓ Professional team collaboration ready
✓ Easier testing (unit tests per service)
✓ Easier to add new classification tasks


STEP-BY-STEP MIGRATION
======================

STEP 1: Copy New Files
─────────────────────

The following new files have been created:

1. config.py (Project Root)
   - Centralized configuration
   - Path definitions
   - Model settings

2. core/__init__.py
   - Package marker

3. core/services/__init__.py
   - Services package
   - Import exports

4. core/services/emotion.py
   - EmotionInferenceService
   - Handles: emotion classification

5. core/services/gender.py
   - GenderInferenceService
   - Handles: gender classification
   - FIXED: Gender class mapping (0=Male, 1=Female)

6. core/services/intent.py
   - IntentInferenceService (by Sahasra)
   - Handles: intent classification
   - Dataset: SLURP

7. core/services/speaker.py
   - SpeakerInferenceService
   - Handles: speaker identification

8. core/utils/__init__.py
   - Utils package marker

9. core/utils/audio.py
   - Utility functions
   - Audio processing
   - Feature extraction
   - Probability utilities

10. app/app_new.py (NEW Flask app)
    - Clean refactored version
    - Uses modular services
    - Centralized config

11. STRUCTURE_GUIDE.md
    - Project structure documentation


STEP 2: Update Python Path Configuration
─────────────────────────────────────────

Your original app.py has:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

The new config.py handles this automatically!

Remove from app.py (now in app/app_new.py):
- Path definitions
- sys.path manipulation

All paths now come from config.py:
    from config import SRC_DIR, MODELS_DIR, etc.


STEP 3: Replace app.py
─────────────────────

Option A: Direct replacement
  1. Backup old: cp app/app.py app/app_old.py
  2. Replace: mv app/app_new.py app/app.py
  3. Test thoroughly

Option B: Gradual migration
  1. Keep both files
  2. Test app_new.py separately
  3. Switch when confident


STEP 4: Update Imports
──────────────────────

If you have other Python files importing from app.py:

OLD Import:
    from app.app import EmotionInferenceService, GenderInferenceService

NEW Import:
    from core.services import EmotionInferenceService, GenderInferenceService

Examples:

OLD (in scripts/train_gender_model.py):
    import sys
    sys.path.insert(0, str(SRC_DIR))

NEW:
    from config import SRC_DIR


STEP 5: Update Configuration
────────────────────────────

All configuration is now in config.py

To modify settings:

BEFORE: Hardcoded in app.py
    MODELS_DIR = ROOT_DIR / "models"
    ALLOWED_EXTENSIONS = {"wav", "mp3", ...}

AFTER: In config.py
    from config import MODELS_DIR, ALLOWED_EXTENSIONS

To override settings:

1. Direct edit config.py
2. Or use environment variables:
    export FLASK_SECRET="your-secret"
    export FLASK_DEBUG=True
    export PORT=5000


STEP 6: Test Each Service
─────────────────────────

Test individual services:

    from core.services import EmotionInferenceService
    from pathlib import Path
    
    service = EmotionInferenceService()
    result = service.predict_emotion(Path("test_audio.wav"))
    print(result)

Test all services:

    from core.services import (
        EmotionInferenceService,
        GenderInferenceService,
        IntentInferenceService,
        SpeakerInferenceService
    )
    
    emotion_service = EmotionInferenceService()
    gender_service = GenderInferenceService()
    intent_service = IntentInferenceService()
    speaker_service = SpeakerInferenceService()
    
    print("✓ All services loaded successfully!")


STEP 7: Verify Gender Classification Fix
─────────────────────────────────────────

Critical: Gender classification was fixed!

OLD Code (WRONG):
    if is_recording:
        GENDER_CLASSES = ["Female", "Male"]  # INVERTED!

NEW Code (CORRECT):
    GENDER_CLASSES = ["Male", "Female"]  # Matches training

Test gender classification:
    from core.services import GenderInferenceService
    from pathlib import Path
    
    service = GenderInferenceService()
    result = service.predict_gender(Path("male_voice.wav"))
    
    # Should predict "Male" now (was predicting "Female" before)
    assert result["label"] == "Male"


STEP 8: Run Flask App
────────────────────

Test the new Flask app:

    cd app
    python app_new.py

Or after replacing:

    cd app
    python app.py

Should see:
    Working directory: /path/to/project
    Starting Flask app on port 5000...
    Debug mode: True
    * Running on http://0.0.0.0:5000


VERIFICATION CHECKLIST
======================

After migration, verify:

□ config.py exists in project root
□ core/services/ directory created with all services
□ core/utils/audio.py created
□ app/app_new.py created
□ All imports updated
□ Flask app starts without errors
□ Emotion classification works
□ Gender classification works correctly (male ≠ female)
□ Intent classification works (Sahasra's model)
□ Speaker identification works
□ All routes accessible:
  □ / (home)
  □ /emotion
  □ /gender
  □ /intent
  □ /speaker
  □ /about
□ File uploads work
□ Microphone recording works


FILE LOCATIONS REFERENCE
========================

OLD STRUCTURE:
  app/app.py (659 lines)
  - All logic here

NEW STRUCTURE:
  config.py ..................... Configuration & paths (80 lines)
  core/__init__.py .............. Package marker
  core/services/__init__.py ...... Services package
  core/services/emotion.py ....... Emotion service (95 lines)
  core/services/gender.py ........ Gender service (120 lines)
  core/services/intent.py ........ Intent service (110 lines)
  core/services/speaker.py ....... Speaker service (130 lines)
  core/utils/__init__.py ......... Utils package
  core/utils/audio.py ............ Audio utilities (150 lines)
  app/app_new.py ................ Flask app (450 lines)


TROUBLESHOOTING
===============

Problem: ModuleNotFoundError: No module named 'core'
Solution: Make sure config.py is in project root, and core/ directory is at project root

Problem: Model not found errors
Solution: Check config.py - verify MODELS_DIR path is correct
         Usually: PROJECT_ROOT / "models"

Problem: "Gender still predicting wrong"
Solution: Make sure you're using the NEW code (gender.py from core/services/)
         Check that GENDER_CLASSES = ["Male", "Female"]
         Training uses 0=Male, 1=Female

Problem: Feature extractor not loading
Solution: Check that src/2_wavlm_feature_extraction.py exists
         Check SRC_DIR path in config.py

Problem: Flask app crashes
Solution: Check that TEMPLATES_DIR and STATIC_DIR paths are correct in config.py
         Usually in app/ subdirectory


QUICK REFERENCE
===============

What Changed in Gender Classification:
───────────────────────────────────────

BEFORE (WRONG - gender inverted):
    GENDER_CLASSES = ["Female", "Male"]  # 0=Female, 1=Male (opposite!)

AFTER (CORRECT):
    GENDER_CLASSES = ["Male", "Female"]  # 0=Male, 1=Female (matches training)

Why This Fix:
- Training data uses: 0=Male, 1=Female (from SPEAKERS.TXT)
- Old code had them reversed
- Male voices were classified as Female (and vice versa)
- NEW code uses consistent mapping across all predictions


NEW FEATURES
============

1. Centralized Configuration (config.py)
   - All paths defined in one place
   - Model settings documented
   - Easy to override with env vars

2. Modular Services (core/services/)
   - Each service is independent
   - Can be tested separately
   - Easy to add new services

3. Utility Functions (core/utils/)
   - Reusable audio processing
   - Probability extraction helpers
   - Dependency checking

4. Better Documentation
   - Every module has docstrings
   - Type hints throughout
   - Comments for complex logic

5. Professional Structure
   - Follows Python best practices
   - Suitable for team collaboration
   - Easy code reviews


NEXT STEPS
==========

After successful migration:

1. Delete old app.py (keep backup for reference)
2. Remove old hardcoded path configurations
3. Update any external scripts to use new imports
4. Add unit tests for each service
5. Add logging system
6. Consider adding REST API
7. Set up CI/CD pipeline


SUPPORT
=======

If you encounter issues:

1. Check STRUCTURE_GUIDE.md for architecture details
2. Verify config.py paths match your system
3. Check core/services/ for implementation details
4. Review core/utils/audio.py for utility usage
5. Compare app/app_new.py with original app.py for route changes


SUCCESS METRICS
===============

Migration successful when:

✓ All services load without errors
✓ All 4 classification tasks work
✓ Gender classification is accurate (male=male, female=female)
✓ Flask app runs without errors
✓ All routes are accessible
✓ File uploads work
✓ Microphone recording works
✓ Code is organized and readable
✓ New developers can understand structure quickly
✓ Adding new features is straightforward
"""
