"""
EMOTION CLASSIFICATION PROJECT - DIRECTORY TREE (v2.0)
======================================================

PROJECT STRUCTURE VISUALIZATION

emotion-classification/
│
├─ 📄 config.py ★                    [NEW] Centralized configuration
├─ 📄 requirements.txt               Python dependencies
├─ 📄 README.md                      Main documentation
│
├─ 📋 DOCUMENTATION FILES
│  ├─ 📄 PROJECT_OVERVIEW.md ★       [NEW] Project overview & info
│  ├─ 📄 STRUCTURE_GUIDE.md ★        [NEW] Architecture documentation
│  ├─ 📄 MIGRATION_GUIDE.md ★        [NEW] Migration instructions
│  ├─ 📄 QUICK_REFERENCE.md ★        [NEW] Quick lookup guide
│  ├─ 📄 RESTRUCTURING_SUMMARY.md ★ [NEW] Restructuring report
│  ├─ 📄 QUICKSTART.md               Getting started
│  ├─ 📄 SETUP_GUIDE.md              Installation guide
│  ├─ 📄 MODEL_ARCHITECTURE.md       Model details
│  └─ 📄 TEAM_LEADER_GUIDE.md        Team information
│
├─ 📁 app/                           FLASK WEB APPLICATION
│  ├─ 📄 app_new.py ★                [NEW] Refactored Flask app (USE THIS!)
│  ├─ 📄 app.py                      [OLD] Original (keep as backup)
│  ├─ 📁 templates/                  HTML Templates
│  │  ├─ 📄 base.html                Base template
│  │  ├─ 📄 index.html               Home page
│  │  ├─ 📄 emotion.html             Emotion classifier UI
│  │  ├─ 📄 gender.html              Gender classifier UI
│  │  ├─ 📄 intent.html              Intent classifier UI (Sahasra)
│  │  ├─ 📄 speaker.html             Speaker identifier UI
│  │  └─ 📄 about.html               About page
│  │
│  ├─ 📁 static/                     Static Files (CSS, JS, Images)
│  │  ├─ 📄 bootstrap.min.css
│  │  ├─ 📄 bootstrap.bundle.min.js
│  │  └─ 📁 css/
│  │     ├─ 📄 styles.css
│  │     └─ 📄 custom.css
│  │
│  └─ 📁 uploads/                    Temporary file storage
│
├─ 📁 core/ ★                        [NEW] CORE MODULES
│  │
│  ├─ 📄 __init__.py
│  │
│  ├─ 📁 services/ ★                 [NEW] INFERENCE SERVICES
│  │  ├─ 📄 __init__.py              Package exports
│  │  │
│  │  ├─ 📄 emotion.py ★             [NEW] Emotion Classification
│  │  │  └─ EmotionInferenceService
│  │  │     ├─ Model: HuBERT-large + SVM
│  │  │     ├─ Accuracy: 79.14%
│  │  │     ├─ Classes: Neutral, Sad, Happy, Angry
│  │  │     └─ ~95 lines
│  │  │
│  │  ├─ 📄 gender.py ★              [NEW] Gender Classification ✓ FIXED
│  │  │  └─ GenderInferenceService
│  │  │     ├─ Model: WavLM-base-plus + Logistic Regression
│  │  │     ├─ Classes: Male (0), Female (1)
│  │  │     ├─ Status: Gender classification FIXED ✓
│  │  │     └─ ~120 lines
│  │  │
│  │  ├─ 📄 intent.py ★              [NEW] Intent Classification (by Sahasra)
│  │  │  └─ IntentInferenceService
│  │  │     ├─ Model: WavLM-base-plus + Classifier
│  │  │     ├─ Dataset: SLURP
│  │  │     ├─ Task: User intent understanding
│  │  │     └─ ~110 lines
│  │  │
│  │  └─ 📄 speaker.py ★             [NEW] Speaker Identification
│  │     └─ SpeakerInferenceService
│  │        ├─ Model: XLSR-53
│  │        ├─ Task: Speaker recognition
│  │        └─ ~130 lines
│  │
│  └─ 📁 utils/ ★                    [NEW] UTILITY MODULES
│     ├─ 📄 __init__.py              Package marker
│     │
│     └─ 📄 audio.py ★               [NEW] Audio Processing
│        ├─ convert_to_wav()
│        ├─ normalize_audio_to_wav()
│        ├─ load_feature_extractor()
│        ├─ l2_normalize()
│        ├─ get_probabilities()
│        ├─ ensure_dependencies()
│        └─ ~150 lines
│
├─ 📁 src/                           FEATURE EXTRACTION (Unchanged)
│  ├─ 📄 1_data_preprocessing.py     Data preparation
│  ├─ 📄 2_wavlm_feature_extraction.py WavLM/HuBERT extraction
│  ├─ 📄 3_train_classifiers.py      Model training
│  ├─ 📄 4_evaluation_metrics.py     Evaluation utilities
│  ├─ 📄 5_visualization_umap.py     UMAP visualization
│  └─ 📁 __pycache__/               Compiled modules
│
├─ 📁 scripts/                       TRAINING SCRIPTS (Unchanged)
│  ├─ 📄 train_gender_model.py       Gender model training
│  ├─ 📄 train_intent_model.py       Intent model training (Sahasra)
│  ├─ 📄 verify_speaker_model.py     Speaker model verification
│  ├─ 📄 download_samples.py         Dataset downloader
│  └─ 📄 [other utility scripts]
│
├─ 📁 models/                        PRE-TRAINED MODELS
│  ├─ 📄 emotion_model_svm.pkl       Emotion classifier
│  ├─ 📄 emotion_scaler.pkl          Emotion feature scaler
│  ├─ 📄 emotion_label_encoder.pkl   Emotion class encoder
│  │
│  ├─ 📄 gender_classifier.pkl       Gender classifier
│  ├─ 📄 gender_scaler.pkl           Gender feature scaler
│  ├─ 📄 gender_pca.pkl              Gender dimensionality reduction
│  ├─ 📄 gender_label_encoder.pkl    Gender class encoder
│  │
│  ├─ 📄 intent_classifier.pkl       Intent classifier (Sahasra)
│  ├─ 📄 intent_scaler.pkl           Intent feature scaler
│  ├─ 📄 intent_pca.pkl              Intent dimensionality reduction
│  ├─ 📄 intent_label_encoder.pkl    Intent class encoder
│  │
│  ├─ 📄 xlsr_classifier.pkl         Speaker identifier
│  ├─ 📄 xlsr_scaler.pkl             Speaker feature scaler
│  ├─ 📄 xlsr_pca.pkl                Speaker dimensionality reduction
│  └─ 📄 xlsr_label_encoder.pkl      Speaker class encoder
│
├─ 📁 data/                          DATASETS
│  ├─ 📁 CREMA-D/                    Emotion dataset
│  │  └─ 📁 CREMA-D-master/
│  │     └─ 📁 AudioWAV/
│  │
│  ├─ 📁 IEMOCAP/                    Emotion dataset
│  │
│  └─ 📁 processed/
│     └─ 📄 cremad_subset.csv
│
├─ 📁 results/                       EVALUATION RESULTS
│  ├─ 📄 confusion_matrix_*.csv      Confusion matrices
│  ├─ 📄 evaluation_results_*.json   Metrics
│  └─ 📄 metrics.json                Summary metrics
│
├─ 📁 embeddings/                    EXTRACTED EMBEDDINGS
│
├─ 📁 .git/                          VERSION CONTROL
│
├─ 📄 .gitignore                     Git ignore rules
├─ 📄 Emotion-classification.code-workspace
└─ 📁 .venv/                         Python virtual environment


STRUCTURE COMPARISON
====================

BEFORE (v1.0 - Monolithic)
──────────────────────────
app/
└─ app.py (659 lines, all logic mixed)
   ├─ EmotionInferenceService
   ├─ GenderInferenceService (with BUG - inverted gender)
   ├─ IntentInferenceService
   ├─ SpeakerInferenceService
   ├─ Flask routes
   ├─ Utility functions
   └─ All hardcoded paths


AFTER (v2.0 - Modular) ★
─────────────────────────
config.py (80 lines)
├─ Centralized configuration
├─ All paths defined once
└─ Model settings

core/services/ (~450 lines total)
├─ emotion.py (95 lines) - Focused service
├─ gender.py (120 lines) - FIXED! Gender now correct
├─ intent.py (110 lines) - Sahasra's model
└─ speaker.py (130 lines) - Speaker identification

core/utils/ (150 lines)
└─ audio.py - Reusable utilities

app/app_new.py (450 lines)
├─ Clean Flask routes
├─ Uses modular services
└─ Professional structure

Documentation (4 files) ★ NEW
├─ PROJECT_OVERVIEW.md
├─ STRUCTURE_GUIDE.md
├─ MIGRATION_GUIDE.md
└─ QUICK_REFERENCE.md


KEY IMPROVEMENTS VISUALIZATION
==============================

BEFORE: Single monolithic file
────────────────────────────────
app.py
├─ 120 lines: Emotion logic
├─ 135 lines: Gender logic (BUGGY)
├─ 125 lines: Intent logic
├─ 145 lines: Speaker logic
├─ 50 lines: Utilities
└─ 84 lines: Flask routes (mixed with logic)
   = 659 lines total (hard to maintain!)


AFTER: Modular structure
─────────────────────────
config.py (80 lines)
core/services/
├─ emotion.py (95 lines) ← Clean, focused
├─ gender.py (120 lines) ← FIXED, focused
├─ intent.py (110 lines) ← Clean, focused
└─ speaker.py (130 lines) ← Clean, focused

core/utils/audio.py (150 lines)
├─ Reusable functions
└─ Used by all services

app/app_new.py (450 lines)
├─ 50 lines: Routes (emotion/gender/intent/speaker)
├─ 30 lines: Utilities
├─ 20 lines: Initialization
└─ 350 lines: Comprehensive documentation


BENEFITS SUMMARY
================

Readability:     ✓✓✓ (each file ~100 lines vs 659 lines)
Maintainability: ✓✓✓ (single responsibility per file)
Testability:     ✓✓✓ (independent service testing)
Scalability:     ✓✓✓ (easy to add new tasks)
Documentation:   ✓✓✓ (comprehensive docs + docstrings)
Team Ready:      ✓✓✓ (clear structure for collaboration)


HOW TO USE NEW STRUCTURE
========================

For Development:
  python app/app_new.py

For Production:
  gunicorn -w 4 app.app:app

In Scripts:
  from core.services import EmotionInferenceService
  service = EmotionInferenceService()

In IDE:
  Type hints provide auto-completion
  Docstrings show in hover


LEGEND
======

★    = New in v2.0 restructuring
✓    = Fixed/Improved in v2.0
📄   = File
📁   = Directory
[NEW] = Created in restructuring
[OLD] = Original file (kept for reference)
"""
