# Application Architecture and How to Run

## Why "Live Server" on index.html doesn’t work
- The app is a Flask web application, not a static site.
- Pages like `/emotion` submit files to the backend (`/emotion/predict`), which runs Python code (HuBERT + SVM) and returns results.
- A static Live Server only serves HTML/CSS/JS. It cannot execute Flask routes or Python inference.
- Therefore, opening `app/templates/index.html` directly won’t show dynamic data and uploads won’t be processed.

## How we run the website
- We start the Flask server (`app/app.py`) which:
  - Serves templates from `app/templates/` and static assets from `app/static/`.
  - Exposes routes like `/`, `/emotion`, `/emotion/predict`.
  - Handles uploaded audio, extracts HuBERT embeddings, applies the SVM, and renders results.

### Commands (Windows PowerShell)
```pwsh
# From the project root
python app\app.py
# Open http://127.0.0.1:5000 in your browser
```

## High-level Architecture

### Frontend (HTML/CSS/JS)
- Files: `app/templates/*.html`, `app/static/css/styles.css`, local Bootstrap.
- Pages:
  - `/` Home with task cards
  - `/emotion` Upload/Record audio with playback UI
- JS handles recording (MediaRecorder), playback UI (time/progress), and form submission.

### Backend (Flask)
- Entry: `app/app.py`
- Routes:
  - `GET /` → renders `index.html`
  - `GET /emotion` → renders `emotion.html` (reads `session['emotion_result']`)
  - `POST /emotion/predict` → accepts file, runs inference, stores result in session, redirects (PRG)
- Uses session to implement Post/Redirect/Get so refresh doesn’t resubmit.

### Model Pipeline (Python)
- Feature extraction: `src/2_wavlm_feature_extraction.py`
  - Loads HuBERT-large (`facebook/hubert-large-ll60k`) via Transformers
  - Handles WAV/MP3/FLAC and WebM recordings (converted via pydub + FFmpeg when present)
  - Resamples to 16kHz, converts to mono, returns 1024-dim embeddings
- Classifier: `models/emotion_model_svm.pkl` (with scaler + label encoder)
  - Scales embeddings, predicts one of 6 emotions

### Data flow for Emotion prediction
1. User selects or records audio on `/emotion`.
2. Browser uploads the file to `POST /emotion/predict`.
3. Server saves a temp file in `app/uploads/`.
4. Extractor loads audio, produces HuBERT embeddings.
5. Scaler + SVM predict probabilities; label chosen.
6. Result stored in `session` and redirected to `/emotion`.
7. Page renders result and probability bars.

## Common Questions
- "Why doesn’t index.html load data on Live Server?"
  - Because the dynamic data is served by Flask endpoints. Use `python app\app.py` instead.
- "Recording upload fails on Windows?"
  - For WebM recordings, install FFmpeg and add it to PATH so pydub can convert:
  - Download: https://www.gyan.dev/ffmpeg/builds/
  - Add `ffmpeg\bin` to PATH and restart the terminal.

## Project Structure (key folders)
```
app/
  app.py                # Flask app and routes
  templates/            # Jinja2 HTML templates
  static/               # CSS and local Bootstrap
src/
  2_wavlm_feature_extraction.py  # HuBERT/WavLM embedding extraction
models/
  emotion_model.pt / pkl files   # SVM + scaler + label encoder
```

## Quick Start
```pwsh
# Install dependencies
pip install -r requirements.txt

# Optional: recording support via FFmpeg for WebM
# Install FFmpeg and ensure ffmpeg.exe is in PATH

# Run the app
python app\app.py
# Visit http://127.0.0.1:5000/emotion
```
