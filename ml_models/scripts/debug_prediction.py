import joblib
import numpy as np
import sys
from pathlib import Path

MODELS_DIR = Path("models")

def debug_prediction():
    print("Loading artifacts...")
    try:
        classifier = joblib.load(MODELS_DIR / "xlsr_classifier.pkl")
        encoder = joblib.load(MODELS_DIR / "xlsr_label_encoder.pkl")
        pca = joblib.load(MODELS_DIR / "xlsr_pca.pkl")
        
        print(f"Classifier type: {type(classifier)}")
        print(f"Encoder type: {type(encoder)}")
        
        if hasattr(classifier, "classes_"):
            print(f"Classifier classes ({len(classifier.classes_)}): {classifier.classes_}")
        else:
            print("Classifier has no classes_ attribute")
            
        print(f"Encoder classes ({len(encoder.classes_)}): {encoder.classes_}")
        
        # Simulate a prediction result
        # Let's assume the classifier returns the label 3575
        mock_pred = 3575
        print(f"\nMock prediction from classifier: {mock_pred}")
        
        # Try to interpret it as an index for inverse_transform
        try:
            print(f"Attempting inverse_transform([{mock_pred}])...")
            res = encoder.inverse_transform([mock_pred])
            print(f"Result: {res}")
        except Exception as e:
            print(f"inverse_transform failed: {e}")

        # Check if 3575 is in encoder classes
        if mock_pred in encoder.classes_:
            print(f"{mock_pred} is in encoder classes.")
            # Try transform
            try:
                idx = encoder.transform([mock_pred])
                print(f"transform([{mock_pred}]) -> {idx}")
            except Exception as e:
                print(f"transform failed: {e}")
        else:
            print(f"{mock_pred} is NOT in encoder classes.")

    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    debug_prediction()
