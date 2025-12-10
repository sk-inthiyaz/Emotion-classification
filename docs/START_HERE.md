"""
RESTRUCTURED PROJECT - START HERE!
==================================

Welcome to the professionally restructured Emotion Classification System v2.0!

This document will help you get started with the new modular architecture.


WHAT'S NEW
==========

We've restructured the entire project for professional team collaboration:

✓ Modular architecture (separate services)
✓ Centralized configuration (no hardcoded paths)
✓ Professional documentation (6 comprehensive guides)
✓ Gender classification FIXED (male/female inversion fixed)
✓ Type hints throughout (better IDE support)
✓ Comprehensive docstrings (every module documented)
✓ Clean folder structure (MERN-like organization)


QUICK START (5 MINUTES)
======================

1. FOLDER STRUCTURE
   ✓ docs/ - All documentation
   ✓ backend/ - Flask app and services
   ✓ ml_models/ - Training code and models
   ✓ config.py - Root configuration

2. TO USE IMMEDIATELY
   Option A - Run new app:
     python backend/app/app.py
   
   Option B - From backend directory:
     cd backend && python app/app.py

3. OPEN IN BROWSER
   http://localhost:5000


DIRECTORY STRUCTURE
===================

emotion-classification/
│
├── docs/                           ★ DOCUMENTATION (All README files)
│   ├── START_HERE.md
│   ├── QUICK_REFERENCE.md
│   ├── PROJECT_OVERVIEW.md
│   ├── STRUCTURE_GUIDE.md
│   └── [other documentation]
│
├── backend/                        ★ FLASK WEB APPLICATION
│   ├── app/
│   │   ├── app.py                 Main Flask app
│   │   ├── templates/             HTML templates
│   │   ├── static/                CSS, JS, images
│   │   └── uploads/               Temp storage
│   │
│   ├── services/                  ★ SERVICES (moved from core/)
│   │   ├── emotion.py
│   │   ├── gender.py              (FIXED!)
│   │   ├── intent.py              (by Sahasra)
│   │   ├── speaker.py
│   │   └── utils/
│   │       ├── audio.py
│   │       └── __init__.py
│   │
│   ├── config.py                  ★ BACKEND CONFIG
│   └── requirements.txt
│
├── ml_models/                      ★ MACHINE LEARNING
│   ├── src/                        Feature extraction
│   │   ├── 1_data_preprocessing.py
│   │   ├── 2_wavlm_feature_extraction.py
│   │   ├── 3_train_classifiers.py
│   │   └── [more files]
│   │
│   ├── scripts/                    Training & utility scripts
│   │   ├── train_gender_model.py
│   │   ├── train_intent_model.py
│   │   └── [more scripts]
│   │
│   ├── models/                     Pre-trained models
│   │   ├── emotion_model_svm.pkl
│   │   ├── gender_classifier.pkl
│   │   └── [all model files]
│   │
│   ├── data/                       Datasets
│   │   ├── CREMA-D/
│   │   ├── IEMOCAP/
│   │   └── processed/
│   │
│   └── results/                    Evaluation results
│       ├── confusion_matrix_*.csv
│       └── [metrics]
│
├── config.py                       ★ ROOT CONFIGURATION
├── README.md                       ★ MAIN PROJECT README
├── requirements.txt                Python dependencies
└── .gitignore


DOCUMENTATION ROADMAP
====================

All documentation is in docs/ folder:

START HERE:
  1. docs/START_HERE.md (this file)
  2. docs/QUICK_REFERENCE.md

THEN READ:
  3. docs/PROJECT_OVERVIEW.md
  4. docs/STRUCTURE_GUIDE.md

IF NEEDED:
  5. docs/MIGRATION_GUIDE.md
  6. docs/RESTRUCTURING_SUMMARY.md


KEY CHANGES
===========

FOLDER STRUCTURE:
  ✓ docs/ - All README files organized
  ✓ backend/ - Flask app and services
  ✓ ml_models/ - ML training and models
  ✓ Clean, MERN-like organization

PATHS UPDATED:
  ✓ config.py - Updated to new paths
  ✓ All imports - Fixed for new structure
  ✓ All services - Using new paths

CRITICAL FIX:
  ✓ Gender classification still fixed!


BASIC USAGE
===========

1. RUNNING THE WEB APP

   python backend/app/app.py
   
   Then open: http://localhost:5000

2. USING SERVICES IN CODE

   from backend.services import EmotionInferenceService
   from pathlib import Path
   
   service = EmotionInferenceService()
   result = service.predict_emotion(Path("audio.wav"))
   print(result["label"])

3. TRAINING MODELS

   python ml_models/scripts/train_gender_model.py
   python ml_models/scripts/train_intent_model.py


CONFIGURATION
=============

Main config: backend/config.py

Model settings:
  - EMOTION_MODEL
  - GENDER_MODEL
  - INTENT_MODEL
  - SPEAKER_MODEL

Paths (all updated):
  - MODELS_DIR
  - SRC_DIR
  - DATA_DIR
  - etc.


SERVICES OVERVIEW
================

1. EMOTION CLASSIFICATION
   Location: backend/services/emotion.py
   Model: HuBERT-large + SVM
   Accuracy: 79.14%

2. GENDER CLASSIFICATION ✓ FIXED
   Location: backend/services/gender.py
   Model: WavLM-base-plus + Logistic Regression
   Status: Now correct!

3. INTENT CLASSIFICATION
   Location: backend/services/intent.py
   Developer: Sahasra

4. SPEAKER IDENTIFICATION
   Location: backend/services/speaker.py
   Model: XLSR-53


SUPPORTED AUDIO FORMATS
=======================

✓ WAV, MP3, FLAC, OGG, M4A, WebM
All formats converted to 16kHz mono WAV


QUICK TEST
==========

1. TEST IMPORTS
   python -c "from backend.services import *; print('✓ OK')"

2. TEST APP
   python backend/app/app.py
   Open: http://localhost:5000

3. TEST GENDER (should predict "Male")
   Upload male voice sample


TROUBLESHOOTING
===============

Q: ModuleNotFoundError
A: Make sure working directory is project root

Q: Path not found
A: Check backend/config.py paths

Q: Gender still wrong
A: Using NEW code from backend/services/gender.py


NEXT STEPS
==========

1. Run the app:
   python backend/app/app.py

2. Test classification:
   Upload audio file

3. Verify it works:
   No errors should appear

4. Learn more:
   Read docs/QUICK_REFERENCE.md


FINAL NOTES
===========

✓ Clean folder structure (MERN-like)
✓ All documentation in docs/
✓ All services in backend/services/
✓ All ML code in ml_models/
✓ All paths updated and correct
✓ No hardcoded paths
✓ Gender classification FIXED
✓ Ready for production

Start with: python backend/app/app.py

Questions? Check docs/ folder!
"""
