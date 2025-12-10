@echo off
echo ===================================================
echo   Starting Emotion/Gender Classification App
echo ===================================================

:: 1. Force navigate to the correct Permanent Project Directory
cd /d "C:\Users\J SHULAMITHI\Emotion-classification"
echo Current Directory: %CD%

:: 2. Install critical dependency for audio
echo Installing audio dependencies...
pip install soundfile

:: 3. Run the App
echo Starting Flask Server...
python app\app.py

:: 4. Pause if it crashes so user can read error
pause
