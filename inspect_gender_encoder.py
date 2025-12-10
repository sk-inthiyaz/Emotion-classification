import joblib
import os

try:
    encoder = joblib.load('models/gender_label_encoder.pkl')
    print(f"Classes: {encoder.classes_}")
except Exception as e:
    print(f"Error: {e}")
