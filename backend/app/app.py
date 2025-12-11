"""
Flask Web Application for Audio Classification
Speech AI Suite with support for Emotion, Gender, Intent, and Speaker identification.

Project Structure (CLEAN - MERN-LIKE):
├── backend/                          # This directory
│   ├── app/                          # Flask application
│   │   ├── app.py                    # Main file (this)
│   │   ├── templates/               # HTML templates
│   │   ├── static/                  # CSS, JavaScript
│   │   └── uploads/                 # Temporary file storage
│   │
│   ├── services/                     # Inference services
│   │   ├── emotion.py               # Emotion classification
│   │   ├── gender.py                # Gender classification (FIXED!)
│   │   ├── intent.py                # Intent classification (Sahasra)
│   │   ├── speaker.py               # Speaker identification
│   │   └── utils/
│   │       └── audio.py             # Audio utilities
│   │
│   └── config.py                     # Backend configuration
│
├── ml_models/                        # ML training & models
│   ├── models/                       # Pre-trained models
│   ├── data/                         # Datasets
│   ├── src/                          # Feature extraction
│   └── scripts/                      # Training scripts
│
└── docs/                             # All documentation

Features:
- Real-time audio classification from microphone
- File upload support (WAV, MP3, FLAC, OGG, M4A, WebM)
- 4 Classification Tasks:
  1. Emotion: Neutral, Sad, Happy, Angry (79.14% accuracy)
  2. Gender: Male, Female (Logistic Regression)
  3. Intent: User intent classification (by Sahasra)
  4. Speaker: Speaker identification (XLSR-53)

Run: python backend/app/app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict

from flask import Flask, render_template, request, redirect, url_for, flash, session

# Add parent backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BACKEND_DIR, TEMPLATES_DIR, STATIC_DIR, UPLOADS_DIR,
    FLASK_CONFIG, ALLOWED_EXTENSIONS
)
from services import (
    EmotionInferenceService,
    GenderInferenceService,
    IntentInferenceService,
    SpeakerInferenceService
)
from services.utils.audio import ensure_dependencies


# ============================================================================
# FLASK APP SETUP
# ============================================================================

def create_app() -> Flask:
    """
    Create and configure Flask application.
    
    Returns:
        Configured Flask app instance
    """
    app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR)
    )
    
    # Configure app
    app.config.update(FLASK_CONFIG)
    
    # Ensure uploads directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # INITIALIZE INFERENCE SERVICES
    # ========================================================================
    emotion_service = EmotionInferenceService()
    gender_service = GenderInferenceService()
    intent_service = IntentInferenceService()
    speaker_service = SpeakerInferenceService()
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def get_file_extension(filename: str) -> str:
        """Extract file extension from filename."""
        return filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
    
    # ========================================================================
    # ROUTES - HOME & ABOUT
    # ========================================================================
    
    @app.route("/")
    def index():
        """Home page - lists all available services."""
        return render_template("index.html")
    
    @app.route("/about")
    def about():
        """About page - project information."""
        return render_template("about.html")
    
    # ========================================================================
    # ROUTES - EMOTION CLASSIFICATION
    # ========================================================================
    
    @app.route("/emotion", methods=["GET"])
    def emotion_page():
        """
        Display emotion classification page.
        Retrieves result from session (Post-Redirect-Get pattern).
        """
        result = session.pop("emotion_result", None)
        return render_template("emotion.html", result=result)
    
    @app.route("/emotion/predict", methods=["POST"])
    def emotion_predict():
        """Process emotion prediction request."""
        file = request.files.get("audio")
        
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("emotion_page"))
        
        file_ext = get_file_extension(file.filename)
        if file_ext not in ALLOWED_EXTENSIONS:
            flash(f"Unsupported file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(url_for("emotion_page"))
        
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOADS_DIR, suffix=f".{file_ext}") as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            result = emotion_service.predict_emotion(tmp_path)
            session["emotion_result"] = result
            return redirect(url_for("emotion_page"))
        
        except Exception as e:
            flash(f"Error during emotion prediction: {str(e)}")
            return redirect(url_for("emotion_page"))
        
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    # ========================================================================
    # ROUTES - GENDER CLASSIFICATION
    # ========================================================================
    
    @app.route("/gender", methods=["GET"])
    def gender_page():
        """Display gender classification page."""
        result = session.pop("gender_result", None)
        return render_template("gender.html", result=result)
    
    @app.route("/gender/predict", methods=["POST"])
    def gender_predict():
        """Process gender prediction request."""
        file = request.files.get("audio")
        
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("gender_page"))
        
        file_ext = get_file_extension(file.filename)
        if file_ext not in ALLOWED_EXTENSIONS:
            flash(f"Unsupported file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(url_for("gender_page"))
        
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOADS_DIR, suffix=f".{file_ext}") as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            is_recording = file.filename.startswith("mic_")
            result = gender_service.predict_gender(tmp_path, is_recording=is_recording)
            session["gender_result"] = result
            return redirect(url_for("gender_page"))
        
        except Exception as e:
            flash(f"Error during gender prediction: {str(e)}")
            return redirect(url_for("gender_page"))
        
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    # ========================================================================
    # ROUTES - INTENT CLASSIFICATION (by Sahasra)
    # ========================================================================
    
    @app.route("/intent", methods=["GET"])
    def intent_page():
        """Display intent classification page."""
        result = session.pop("intent_result", None)
        return render_template("intent.html", result=result)
    
    @app.route("/intent/predict", methods=["POST"])
    def intent_predict():
        """Process intent prediction request."""
        file = request.files.get("audio")
        
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("intent_page"))
        
        file_ext = get_file_extension(file.filename)
        if file_ext not in ALLOWED_EXTENSIONS:
            flash(f"Unsupported file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(url_for("intent_page"))
        
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOADS_DIR, suffix=f".{file_ext}") as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            result = intent_service.predict_intent(tmp_path)
            session["intent_result"] = result
            return redirect(url_for("intent_page"))
        
        except Exception as e:
            flash(f"Error during intent prediction: {str(e)}")
            return redirect(url_for("intent_page"))
        
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    # ========================================================================
    # ROUTES - SPEAKER IDENTIFICATION
    # ========================================================================
    
    @app.route("/speaker", methods=["GET"])
    def speaker_page():
        """Display speaker identification page."""
        result = session.pop("speaker_result", None)
        return render_template("speaker.html", result=result)
    
    @app.route("/speaker/predict", methods=["POST"])
    def speaker_predict():
        """Process speaker prediction request."""
        file = request.files.get("audio")
        
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("speaker_page"))
        
        file_ext = get_file_extension(file.filename)
        if file_ext not in ALLOWED_EXTENSIONS:
            flash(f"Unsupported file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(url_for("speaker_page"))
        
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOADS_DIR, suffix=f".{file_ext}") as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            result = speaker_service.predict_speaker(tmp_path)
            session["speaker_result"] = result
            return redirect(url_for("speaker_page"))
        
        except Exception as e:
            flash(f"Error during speaker prediction: {str(e)}")
            return redirect(url_for("speaker_page"))
        
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    return app


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

app = create_app()


if __name__ == "__main__":
    """
    Application entry point with setup and initialization.
    """
    
    # Ensure we're running from project root
    os.chdir(BACKEND_DIR.parent)
    print(f"Working directory: {os.getcwd()}")
    
    # Ensure required dependencies are installed
    ensure_dependencies()
    
    # Get configuration from environment
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", True)
    
    # Setup SSL context (optional)
    ssl_context = None
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        ssl_context = ("cert.pem", "key.pem")
    
    # Start Flask app
    print(f"Starting Flask app on port {port}...")
    print(f"Debug mode: {debug_mode}")
    print(f"Open: http://localhost:{port}")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        ssl_context=ssl_context
    )
