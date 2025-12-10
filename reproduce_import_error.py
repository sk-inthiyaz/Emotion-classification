try:
    from transformers import Wav2Vec2Processor
    print("Successfully imported Wav2Vec2Processor")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
