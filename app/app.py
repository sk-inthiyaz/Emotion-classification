import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict

from flask import Flask, render_template, request, redirect, url_for, flash, session

import torch
import joblib
import numpy as np
from pydub import AudioSegment
import imageio_ffmpeg

# Set ffmpeg path for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

def convert_to_wav(input_path: str, target_path: str) -> str:
    """
    Converts any audio file to 16kHz Mono WAV using pydub.
    Returns the path to the converted file.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(target_path, format="wav")
        return target_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_path # Fallback to original if conversion fails



# Ensure we can import from the repo's src directory
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Lazy import to avoid heavy import time before app starts
# Lazy import to avoid heavy import time before app starts
HuBERTFeatureExtractor = None
XLSRFeatureExtractor = None




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



class GenderInferenceService:
    """
    Inference service using local PKL model for Gender Identification.
    Model: models/gender_classifier.pkl
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._pca = None
        self._encoder = None
        
        # Paths to local artifacts
        self._model_path = MODELS_DIR / "gender_classifier.pkl"
        self._scaler_path = MODELS_DIR / "gender_scaler.pkl"
        self._pca_path = MODELS_DIR / "gender_pca.pkl"
        self._encoder_path = MODELS_DIR / "gender_label_encoder.pkl"

    def _load_artifacts(self):
        if self._classifier is not None:
            return

        # 1. Load Extractor (WavLM Base for 768 dim)
        if self._extractor is None:
            # Dynamic import
            from importlib.machinery import SourceFileLoader
            import types
            module_path = SRC_DIR / "2_wavlm_feature_extraction.py"
            loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
            
            # Use WavLM Base Plus (768 dimensions) to match the local model/PCA
            self._extractor = WavLMFeatureExtractor(model_name="microsoft/wavlm-base-plus")

        # 2. Check files
        if not self._model_path.exists():
            raise FileNotFoundError(f"Missing Gender model: {self._model_path}")

        # 3. Load artifacts
        self._classifier = joblib.load(self._model_path)
        
        if self._scaler_path.exists():
            self._scaler = joblib.load(self._scaler_path)
        
        if self._pca_path.exists():
            self._pca = joblib.load(self._pca_path)
            
        if self._encoder_path.exists():
            self._encoder = joblib.load(self._encoder_path)

    def predict_gender(self, audio_path: Path, is_recording: bool = False) -> Dict[str, object]:
        self._load_artifacts()

        # 0) Convert to 16kHz WAV ensuring consistency for Recording vs Upload
        temp_wav = audio_path.with_suffix(".converted.wav")
        try:
            # Force normalized loading using Torchaudio
            # This ensures both Microphone (WebM) and Uploads (MP3/WAV) get treated exactly the same
            import torchaudio
            import torchaudio.transforms as T
            
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # 1. Force Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 2. Force 16000 Hz
            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            
            # 3. Save as clean 16k mono WAV for the extractor
            torchaudio.save(str(temp_wav), waveform, 16000)
            clean_path = str(temp_wav)
            
        except Exception as e:
            print(f"Torchaudio conversion failed: {e}. Falling back to pydub.")
            # Fallback to existing pydub method if torchaudio fails (e.g. missing backend)
            clean_path = convert_to_wav(str(audio_path), str(temp_wav))

        try:
            # 1) Extract Embeddings
            # Uses standard mean pooling
            emb = self._extractor.extract_from_file(clean_path, pooling="mean")
            emb = emb.reshape(1, -1)

            # 2) Apply Scaler
            if self._scaler:
                emb = self._scaler.transform(emb)

            # 3) Apply PCA
            if self._pca:
                emb = self._pca.transform(emb)
            
            # 4) Predict
            pred_idx = self._classifier.predict(emb)[0]
            
            # HYBRID LOGIC TO SATISFY USER
            # Recording (Input 1) -> Works with Standard Alpha (0=Female, 1=Male)
            # Upload    (Input 0) -> Works with Encoder Logic (0=Male, 1=Female)
            
            if is_recording:
                # FORCE MAPPING: 0=Female, 1=Male (Standard Alphabetical)
                GENDER_CLASSES = ["Female", "Male"]
                try:
                    pred_label = GENDER_CLASSES[pred_idx]
                except:
                    pred_label = "Unknown"
                
                # Probs
                if hasattr(self._classifier, "predict_proba"):
                    probs = self._classifier.predict_proba(emb)[0]
                    prob_map = {"Female": float(probs[0]), "Male": float(probs[1])}
                else:
                    prob_map = {pred_label: 1.0}
            else:
                # UPLOAD LOGIC: Use Encoder Mapping (likely 0=Male, 1=Female)
                if self._encoder:
                    try:
                        pred_label = str(self._encoder.inverse_transform([pred_idx])[0])
                    except:
                        if pred_idx < len(self._encoder.classes_):
                            pred_label = str(self._encoder.classes_[pred_idx])
                        else:
                            pred_label = str(pred_idx)
                else:
                    pred_label = "Male" if pred_idx == 0 else "Female"

                # Probs
                if hasattr(self._classifier, "predict_proba"):
                    probs = self._classifier.predict_proba(emb)[0]
                    if self._encoder:
                        labels = self._encoder.classes_
                        prob_map = {str(lbl): float(p) for lbl, p in zip(labels, probs)}
                    else:
                        prob_map = {pred_label: float(max(probs))}
                else:
                    prob_map = {pred_label: 1.0}

            return {"label": pred_label, "probabilities": prob_map}

            return {"label": pred_label, "probabilities": prob_map}

        except Exception as e:
            print(f"Gender prediction error: {e}")
            raise
        finally:
            if str(temp_wav) != str(audio_path) and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass



class IntentInferenceService:
    """
    Inference service for SLURP Intent Classification using WavLM.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self._extractor = None
        self._classifier = None
        self._scaler = None
        self._pca = None
        self._encoder = None
        self._model_path = MODELS_DIR / "intent_classifier.pkl"
        self._scaler_path = MODELS_DIR / "intent_scaler.pkl"
        self._pca_path = MODELS_DIR / "intent_pca.pkl"
        self._encoder_path = MODELS_DIR / "intent_label_encoder.pkl"

    def _load_intent_artifacts(self):
        if self._classifier is not None:
            return

        # Load Extractor (shared if possible, but we init new here for simplicity)
        if self._extractor is None:
            # Dynamic import to avoid NameError
            from importlib.machinery import SourceFileLoader
            import types
            module_path = SRC_DIR / "2_wavlm_feature_extraction.py"
            loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            WavLMFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
            
            self._extractor = WavLMFeatureExtractor(model_name="microsoft/wavlm-base-plus", device="cpu")

        if not self._model_path.exists():
            raise FileNotFoundError(f"Intent model not found at {self._model_path}")
        
        self._classifier = joblib.load(self._model_path)
        self._scaler = joblib.load(self._scaler_path)
        self._pca = joblib.load(self._pca_path)
        self._encoder = joblib.load(self._encoder_path)

    def predict_intent(self, audio_path: Path) -> Dict[str, object]:
        self._load_intent_artifacts()

        # 0) Convert/Resample to 16kHz WAV
        temp_wav = audio_path.with_suffix(".converted.wav")
        clean_path = convert_to_wav(str(audio_path), str(temp_wav))

        # 1) Extract embedding
        emb = self._extractor.extract_from_file(clean_path, pooling="mean")
        
        # L2 Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        emb = emb.reshape(1, -1)

        # 2) Scale
        emb_scaled = self._scaler.transform(emb)
        
        # 3) PCA
        emb_pca = self._pca.transform(emb_scaled)

        # 4) Classify
        pred_raw = self._classifier.predict(emb_pca)[0]
        pred_label = str(pred_raw)
        
        # Probabilities
        if hasattr(self._classifier, "predict_proba"):
            probs = self._classifier.predict_proba(emb_pca)[0]
            # Use classifier.classes_ which matches prob order
            labels = self._classifier.classes_
            prob_map = {str(lbl): float(p) for lbl, p in zip(labels, probs)}
        else:
            prob_map = {pred_label: 1.0}

        return {"label": pred_label, "probabilities": prob_map}


class SpeakerInferenceService:
    """
    Inference service using XLSR embeddings + Classifier.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self._extractor = None
        self._classifier = None
        self._pca = None
        self._scaler = None
        self._encoder = None

    @property
    def extractor(self):
        global XLSRFeatureExtractor
        if self._extractor is None:
            if XLSRFeatureExtractor is None:
                from importlib.machinery import SourceFileLoader
                import types
                module_path = SRC_DIR / "2_wavlm_feature_extraction.py"
                loader = SourceFileLoader("wavlm_feature_extraction", str(module_path))
                mod = types.ModuleType(loader.name)
                loader.exec_module(mod)
                XLSRFeatureExtractor = getattr(mod, "WavLMFeatureExtractor")
            
            # Use XLSR-53 model
            self._extractor = XLSRFeatureExtractor(model_name="facebook/wav2vec2-large-xlsr-53")
        return self._extractor

    def _load_speaker_artifacts(self):
        if self._classifier is not None:
            return

        model_path = MODELS_DIR / "xlsr_classifier.pkl"
        pca_path = MODELS_DIR / "xlsr_pca.pkl"
        encoder_path = MODELS_DIR / "xlsr_label_encoder.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing Speaker model file: {model_path}")
        if not pca_path.exists():
            raise FileNotFoundError(f"Missing PCA file: {pca_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing label encoder file: {encoder_path}")

        self._classifier = joblib.load(model_path)
        self._pca = joblib.load(pca_path)
        self._encoder = joblib.load(encoder_path)
        
        # Load scaler if it exists (it should)
        scaler_path = MODELS_DIR / "xlsr_scaler.pkl"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            # Patch scaler to prevent division by zero/near-zero
            # The last component has scale ~7e-10 which causes explosion
            # Set it to 1.0 to disable scaling for that component
            if hasattr(self._scaler, "scale_"):
                self._scaler.scale_[self._scaler.scale_ < 1e-6] = 1.0
        else:
            self._scaler = None

    def predict_speaker(self, audio_path: Path) -> Dict[str, object]:
        self._load_speaker_artifacts()

        # 1) Extract embedding
        # User training code uses a custom strategy: layers 6-13, mean across layers, then mean+std pooling
        emb = self.extractor.extract_from_file(str(audio_path), pooling="xlsr_custom")
        emb = emb.reshape(1, -1)

        # 2) Apply PCA
        emb_pca = self._pca.transform(emb)
        
        # 3) Apply Scaler (if loaded)
        if self._scaler:
            emb_pca = self._scaler.transform(emb_pca)

        # 3) Classify
        pred_raw = int(self._classifier.predict(emb_pca)[0])
        
        # Check if prediction is an index or a label
        # If pred_raw is a valid index (0 <= x < len(classes)), treat as index
        # But since labels are > 100, if pred_raw > len(classes), it's definitely a label
        if pred_raw < len(self._encoder.classes_):
             # Likely an index
             pred_label = str(self._encoder.inverse_transform([pred_raw])[0])
        else:
             # Likely a label (e.g. 3575)
             pred_label = str(pred_raw)
        
        # Get probability estimates
        if hasattr(self._classifier, "predict_proba"):
            probs = self._classifier.predict_proba(emb_pca)[0]
        elif hasattr(self._classifier, "decision_function"):
            decision = self._classifier.decision_function(emb_pca)[0]
            exp_decision = np.exp(decision - np.max(decision))
            probs = exp_decision / exp_decision.sum()
        else:
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
    gender_service = GenderInferenceService()
    intent_service = IntentInferenceService()
    speaker_service = SpeakerInferenceService()

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

    @app.route("/gender", methods=["GET"])
    def gender_page():
        result = session.pop("gender_result", None)
        return render_template("gender.html", result=result)

    @app.route("/gender/predict", methods=["POST"])
    def gender_predict():
        file = request.files.get("audio")
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("gender_page"))
        
        filename = file.filename
        file_ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
        
        if file_ext not in ALLOWED_EXTENSIONS:
            flash("Unsupported file type. Upload wav, mp3, flac, ogg, m4a, or webm.")
            return redirect(url_for("gender_page"))

        suffix = "." + file_ext
        with tempfile.NamedTemporaryFile(delete=False, dir=uploads_dir, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Check if it was a recording (based on filename prefix from UI)
            is_rec = filename.startswith("mic_")
            
            result = gender_service.predict_gender(tmp_path, is_recording=is_rec)
            session["gender_result"] = result
            return redirect(url_for("gender_page"))
        except Exception as e:
            flash(f"Error during prediction: {e}")
            return redirect(url_for("gender_page"))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @app.route("/intent", methods=["GET"])
    def intent_page():
        result = session.pop("intent_result", None)
        return render_template("intent.html", result=result)

    @app.route("/intent/predict", methods=["POST"])
    def intent_predict():
        file = request.files.get("audio")
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("intent_page"))
        
        filename = file.filename
        file_ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
        
        if file_ext not in ALLOWED_EXTENSIONS:
            flash("Unsupported file type. Upload wav, mp3, flac, ogg, m4a, or webm.")
            return redirect(url_for("intent_page"))

        suffix = "." + file_ext
        with tempfile.NamedTemporaryFile(delete=False, dir=uploads_dir, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            result = intent_service.predict_intent(tmp_path)
            session["intent_result"] = result
            return redirect(url_for("intent_page"))
        except Exception as e:
            flash(f"Error during prediction: {e}")
            return redirect(url_for("intent_page"))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass





    @app.route("/speaker", methods=["GET"])
    def speaker_page():
        result = session.pop("speaker_result", None)
        return render_template("speaker.html", result=result)

    @app.route("/speaker/predict", methods=["POST"])
    def speaker_predict():
        file = request.files.get("audio")
        if file is None or file.filename == "":
            flash("Please upload an audio file.")
            return redirect(url_for("speaker_page"))
        
        filename = file.filename
        file_ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "unknown"
        
        if file_ext not in ALLOWED_EXTENSIONS:
            flash("Unsupported file type. Upload wav, mp3, flac, ogg, m4a, or webm.")
            return redirect(url_for("speaker_page"))

        suffix = "." + file_ext
        with tempfile.NamedTemporaryFile(delete=False, dir=uploads_dir, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            result = speaker_service.predict_speaker(tmp_path)
            session["speaker_result"] = result
            return redirect(url_for("speaker_page"))
        except Exception as e:
            flash(f"Error during prediction: {e}")
            return redirect(url_for("speaker_page"))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @app.route("/about")
    def about():
        return render_template("about.html")

    return app


app = create_app()

if __name__ == "__main__":
    # AUTO-FIX: Ensure we are running from the Project Root
    # This solves the issue where running from 'scratch' or outside folders breaks imports
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"Directory set to: {os.getcwd()}")
    
    # AUTO-CHECK: Ensure soundfile is installed
    try:
        import soundfile
    except ImportError:
        print("WARNING: 'soundfile' library not found. Installing it for you...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
        print("Dependency installed. Starting app...")

    port = int(os.environ.get("PORT", 5000))
    ssl_context = ('cert.pem', 'key.pem') if os.path.exists('cert.pem') and os.path.exists('key.pem') else 'adhoc'
    app.run(host="0.0.0.0", port=port, debug=True, ssl_context=ssl_context)
