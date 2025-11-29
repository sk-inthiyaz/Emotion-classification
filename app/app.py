import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict

from flask import Flask, render_template, request, redirect, url_for, flash, session

import torch
import joblib
import numpy as np


# Ensure we can import from the repo's src directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Lazy import to avoid heavy import time before app starts
HuBERTFeatureExtractor = None


class EmotionInferenceService:
    """
    Inference service using HuBERT-large embeddings + SVM classifier.
    Achieves 79.14% accuracy on emotion classification.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._encoder = None

    @property
    def extractor(self):
        global HuBERTFeatureExtractor
        if self._extractor is None:
            # Import here to reduce app startup time and work around leading digit in filename
            if HuBERTFeatureExtractor is None:
                from importlib.machinery import SourceFileLoader
                import types
                module_path = SRC_DIR / "2_wavlm_feature_extraction.py"
                loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
                mod = types.ModuleType(loader.name)
                loader.exec_module(mod)
                # Reuse WavLMFeatureExtractor but switch to HuBERT-large
                HuBERTFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
            # Use HuBERT-large model for best accuracy (79.14%)
            self._extractor = HuBERTFeatureExtractor(model_name="facebook/hubert-large-ll60k")
        return self._extractor

    def _load_emotion_artifacts(self):
        if self._classifier is not None:
            return

        model_path = MODELS_DIR / "emotion_model_svm.pkl"
        scaler_path = MODELS_DIR / "emotion_scaler.pkl"
        encoder_path = MODELS_DIR / "emotion_label_encoder.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing SVM model file: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing label encoder file: {encoder_path}")

        # Load SVM classifier (79.14% accuracy with HuBERT-large)
        self._classifier = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._encoder = joblib.load(encoder_path)

    def predict_emotion(self, audio_path: Path) -> Dict[str, object]:
        self._load_emotion_artifacts()

        # 1) Extract embedding using HuBERT-large
        emb = self.extractor.extract_from_file(str(audio_path))
        emb = emb.reshape(1, -1)

        # 2) Scale
        emb_scaled = self._scaler.transform(emb)

        # 3) Classify using SVM
        pred_idx = int(self._classifier.predict(emb_scaled)[0])
        pred_label = str(self._encoder.inverse_transform([pred_idx])[0])
        
        # Get probability estimates from SVM
        if hasattr(self._classifier, "predict_proba"):
            probs = self._classifier.predict_proba(emb_scaled)[0]
        elif hasattr(self._classifier, "decision_function"):
            # For SVM with decision_function, convert to probabilities using softmax
            decision = self._classifier.decision_function(emb_scaled)[0]
            exp_decision = np.exp(decision - np.max(decision))
            probs = exp_decision / exp_decision.sum()
        else:
            # Fallback: uniform probabilities
            probs = np.ones(len(self._encoder.classes_)) / len(self._encoder.classes_)

        # Return probabilities as dict {label: prob}
        labels = list(self._encoder.classes_)
        prob_map = {str(lbl): float(probs[i]) for i, lbl in enumerate(labels)}
        return {"label": pred_label, "probabilities": prob_map}


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"), static_folder=str(Path(__file__).parent / "static"))
    app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

    uploads_dir = Path(__file__).parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    emotion_service = EmotionInferenceService()

    ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "webm"}

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/emotion", methods=["GET"])
    def emotion_page():
        # Retrieve prediction result from session (PRG pattern)
        result = session.pop("emotion_result", None)
        return render_template("emotion.html", result=result)

    @app.route("/emotion/predict", methods=["POST"])
    def emotion_predict():
        file = request.files.get("audio")
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("emotion_page"))
        
        # Get file extension
        filename = file.filename
        file_ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
        
        # Check if it's an allowed format
        if file_ext not in ALLOWED_EXTENSIONS:
            flash("Unsupported file type. Upload wav, mp3, flac, ogg, m4a, or webm.")
            return redirect(url_for("emotion_page"))

        # Save to a temporary file with correct extension
        suffix = "." + file_ext
        with tempfile.NamedTemporaryFile(delete=False, dir=uploads_dir, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            result = emotion_service.predict_emotion(tmp_path)
            # Store result in session and redirect (PRG) so refresh is safe
            session["emotion_result"] = result
            return redirect(url_for("emotion_page"))
        except Exception as e:
            flash(f"Error during prediction: {e}")
            return redirect(url_for("emotion_page"))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @app.route("/gender", methods=["GET", "POST"])
    def gender_page():
        placeholder = {
            "label": "N/A",
            "note": "Gender model not integrated yet. Page is functional.",
        }
        if request.method == "POST":
            file = request.files.get("audio")
            if file is None or file.filename == "":
                flash("Please upload an audio file.")
                return redirect(url_for("gender_page"))
            flash("Uploaded successfully. Inference not yet implemented.")
            return render_template("gender.html", result=placeholder)
        return render_template("gender.html", result=None)

    @app.route("/intent", methods=["GET", "POST"])
    def intent_page():
        placeholder = {
            "label": "N/A",
            "note": "Intent model not integrated yet. Page is functional.",
        }
        if request.method == "POST":
            file = request.files.get("audio")
            if file is None or file.filename == "":
                flash("Please upload an audio file.")
                return redirect(url_for("intent_page"))
            flash("Uploaded successfully. Inference not yet implemented.")
            return render_template("intent.html", result=placeholder)
        return render_template("intent.html", result=None)

    @app.route("/speaker", methods=["GET", "POST"])
    def speaker_page():
        placeholder = {
            "label": "N/A",
            "note": "Speaker model not integrated yet. Page is functional.",
        }
        if request.method == "POST":
            file = request.files.get("audio")
            if file is None or file.filename == "":
                flash("Please upload an audio file.")
                return redirect(url_for("speaker_page"))
            flash("Uploaded successfully. Inference not yet implemented.")
            return render_template("speaker.html", result=placeholder)
        return render_template("speaker.html", result=None)

    @app.route("/about")
    def about():
        return render_template("about.html")

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
